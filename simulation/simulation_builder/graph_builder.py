from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys

import tensorflow as tf
import json

from simulation.simulation_builder.graph_duplicator import copy_and_duplicate
from simulation.simulation_builder.optimizers import GDOptimizer
from simulation.simulation_builder.optimizers import NormalNoiseGDOptimizer
from simulation.simulation_builder.summary import Summary
from simulation.simulator_exceptions import InvalidDatasetTypeError
from simulation.simulator_exceptions import InvalidArchitectureFuncError

__DEBUG__ = True

'export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/'
class GraphBuilder(object):

	def __init__(self, architecture, learning_rate, noise_list, name, 
		noise_type='random_normal', summary_type=None):
		
		self._architecture = architecture
		self._learning_rate = learning_rate
		self._n_workers = len(noise_list)
		self._noise_type = noise_type
		self._name = name
		self._graph = tf.Graph()
		self._noise_list = (sorted(noise_list) 
			if noise_type == 'random_normal' 
			else sorted(noise_list, reverse=True))
		self._summary_type = summary_type
		#
		# create graph with duplicates based on architecture function 
		# and noise typetrain_collect
		res = []
		try:
			res = self._architecture(tf.Graph())
			if (len(res) == 3 and 
				noise_type == 'random_normal'):
				X, y, logits = res
				self.X, self.y, logits_list = copy_and_duplicate(X, y, logits, 
					self._n_workers, self._graph)

				# _noise_plcholders will be used to store noise vals for summaries
				with self._graph.as_default():
					self._noise_plcholders = {i:tf.placeholder_with_default(0.0, shape=[])
						for i in range(self._n_workers)}

				# curr_noise_dict stores {worker_id:current noise stddev VALUE}  
				self._curr_noise_dict = {i:n for i, n in enumerate(self._noise_list)}

			elif (len(res) == 4 and 
				noise_type == 'dropout'):
				X, y, prob_placeholder, logits = res
				self.X, self.y, probs, logits_list = copy_and_duplicate(X, y, logits, 
					self._n_workers, self._graph, prob_placeholder)

				# _noise_plcholders stores dropout plcholders: {worker_id:plcholder}
				# it is used also to store summaries
				self._noise_plcholders = {i:p for i, p in enumerate(probs)}
				
				# in case of noise_type == dropout, _curr_noise_dict stores
				# probabilities for keeping optimization parameters 
				# (W's and b's): {worker_id:keep_proba}
				self._curr_noise_dict = {i:n 
					for i, n in enumerate(sorted(self._noise_list, reverse=True))}

			else: 
				raise InvalidArchitectureFuncError(len(res), self.noise_type)

		except:
			raise

		#
		# from here, whole net that goes after logits is created
		self.__DEBUG__logits_list = logits_list
		self._loss_dict = {}
		self._acc_dict = {}
		self._optimizer_dict = {}

		with self._graph.as_default():
			for i in range(self._n_workers):
				
				with tf.name_scope('LossAccuracy_' + str(i)):
					
					self._loss_dict[i] = self._cross_entropy_loss(self.y, 
						logits_list[i])
					
					self._acc_dict[i] = self._accuracy(self.y, 
						logits_list[i])

				with tf.name_scope('Optimizer_' + str(i)):

					Optimizer = (NormalNoiseGDOptimizer 
						if noise_type == 'random_normal' else GDOptimizer)

					optimizer = Optimizer(self._learning_rate, i,
						noise_list=self._noise_list)
					
					self._optimizer_dict[i] = optimizer
					optimizer.minimize(self._loss_dict[i])
			
			self._summary = Summary(self._graph, self._n_workers, self._name, 
				self._loss_dict, self._acc_dict, self._noise_list, 
				self._noise_plcholders, summary_type=self._summary_type)

			self.variable_initializer = tf.global_variables_initializer()


	def create_feed_dict(self, X_batch, y_batch, dataset_type='train'):
		"""Creates feed_dict for session run.

		Args:
			X_batch: input X training batch
			y_batch: input y training batch
			dataset_type: 'train', 'test' or 'validation'
		
		Returns:
			A dictionary to feed into session run.
			If dataset_type=='train', adds to feed_dict placeholders 
			to store noise (for summary).
			If dataset_type=='validation'/'test', then doesn't add 
			this placeholder (since we don't add noise for test or 
			validation).
			If noise_type is 'dropout' and dataset_type is 'train', 
			adds values for keeping parameters during optimization 
			(placeholders keep_prob for each worker).
		"""

		feed_dict = {self.X:X_batch, self.y:y_batch}

		if dataset_type == 'test' or dataset_type == 'validation':
			d = {self._noise_plcholders[i]:1.0
				for i in range(self._n_workers)}
			
		elif dataset_type == 'train':
			d = {self._noise_plcholders[i]:self._curr_noise_dict[i]
				for i in range(self._n_workers)}
		
		else:
			raise InvalidDatasetTypeError()
		
		feed_dict.update(d)
		
		return feed_dict

	def get_train_ops(self, dataset_type='train'):
		"""Returns train ops for session's run.

		The returned list should be used as:
		# evaluated = sess.run(get_train_ops(), feed_dict=...)
		
		Args:
			test: if True, doesn't include ops for optimizing gradients 
				in the returned list.

		Returns:
			train_ops for session run.
		"""

		loss = [self._loss_dict[i] for i in range(self._n_workers)]
		accuracy = [self._acc_dict[i] for i in range(self._n_workers)]
		summary = self._summary.get_summary_ops(dataset_type)

		if dataset_type == 'test' or dataset_type == 'validation':
			return loss + accuracy + summary
		elif dataset_type == 'train':
			train_op = [self._optimizer_dict[i].get_train_op() 
				for i in range(self._n_workers)]
			if __DEBUG__: train_op = train_op + self.__DEBUG__logits_list
			return loss + accuracy + summary + train_op
		else:
			raise InvalidDatasetTypeError()

	def add_summary(self, evaluated, step, dataset_type='train'):
		summs = self.extract_evaluated_tensors(evaluated, 'summary')
		self._summary.add_summary(summs, step, dataset_type)

		
	
	def extract_evaluated_tensors(self, evaluated, tensor_type):
		
		if tensor_type == 'loss':
			return evaluated[:self._n_workers]
		
		elif tensor_type == 'accuracy':
			return evaluated[self._n_workers:2*self._n_workers]

		elif tensor_type == 'summary':
			end_mult = (4 if self._summary_type is None else 3)
			return evaluated[2*self._n_workers:end_mult*self._n_workers]

	def update_noise_vals(self, evaluated):
		"""Updates noise values based on loss function.

		If the noise_type is random_normal then the optimizaiton 
		route is updated. If noise_type is dropout, then the dropout
		placeholders are updated.

		Args:
			evaluated: a list as returned by sess.run(get_train_ops())
		"""
		loss_list = self.extract_evaluated_tensors(evaluated, 'loss')
		losses_and_ids = [(l, i) for i, l in enumerate(loss_list)]
		
		losses_and_ids.sort(key=lambda x: x[0])
		
		for i, li in enumerate(losses_and_ids):
			self._curr_noise_dict[li[1]] = self._noise_list[i]
		
		if self._noise_type == 'random_normal':
			for i, li in enumerate(losses_and_ids):
				self._optimizer_dict[li[1]].set_train_route(i)

	def get_tf_graph(self): return self._graph
		
	def _store_tensorflow_graph(self, path): tf.summary.FileWriter(path, self.graph).close()
	
	def _cross_entropy_loss(self, y, logits):
		with tf.name_scope('CrossEntropyLoss'):
			with tf.device('/cpu:0'):
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=y, logits=logits)
				loss = tf.reduce_mean(cross_entropy, name='loss')
		return loss

	def _accuracy(self, y, logits):
		with tf.name_scope('Accuracy'):
			with tf.device('/cpu:0'):
				y_pred = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
				
				accuracy = tf.reduce_mean(tf.cast(x=y_pred, dtype=tf.float32), 
					name='accuracy')
		return accuracy

	




