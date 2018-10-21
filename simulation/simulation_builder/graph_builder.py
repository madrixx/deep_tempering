from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import sys

import tensorflow as tf
import numpy as np
import json
import random

from simulation.simulation_builder.graph_duplicator import copy_and_duplicate
from simulation.simulation_builder.optimizers import GDOptimizer
from simulation.simulation_builder.optimizers import NormalNoiseGDOptimizer
from simulation.simulation_builder.optimizers import GDLDOptimizer
from simulation.simulation_builder.optimizers import LDSampler
from simulation.simulation_builder.summary import Summary
from simulation.simulator_exceptions import InvalidDatasetTypeError
from simulation.simulator_exceptions import InvalidArchitectureFuncError
from simulation.simulator_exceptions import InvalidLossFuncError
from simulation.simulator_utils import __DEBUG__

'export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/'
class GraphBuilder(object):

	def __init__(self, architecture, learning_rate, noise_list, name, 
		noise_type='random_normal', summary_type=None, simulation_num=None, 
		surface_view='information', loss_func_name='cross_entropy'):
		
		self._architecture = architecture
		self._learning_rate = learning_rate
		self._n_replicas = len(noise_list)
		self._noise_type = noise_type
		self._name = name
		self._surface_view = surface_view
		self._graph = tf.Graph()
		self._noise_list = (sorted(noise_list) 
			if noise_type == 'random_normal' or noise_type == 'betas' or 'LDSamper'
			else sorted(noise_list, reverse=True))
		self._summary_type = summary_type
		self._simulation_num = '' if simulation_num is None else str(simulation_num)
		self._loss_func_name = loss_func_name
		# create graph with duplicates based on architecture function 
		# and noise type 
		res = []
		try:
			res = self._architecture(tf.Graph())
			if (len(res) == 3 and 
				(noise_type == 'random_normal' or noise_type == 'betas' or noise_type == 'LDSampler')):
				X, y, logits = res
				self.X, self.y, logits_list = copy_and_duplicate(X, y, logits, 
					self._n_replicas, self._graph)

				# _noise_plcholders will be used to store noise vals for summaries
				with self._graph.as_default():
					self._noise_plcholders = {i:tf.placeholder(tf.float32, shape=[])
						for i in range(self._n_replicas)}

				# curr_noise_dict stores {replica_id:current noise stddev VALUE}  
				self._curr_noise_dict = {i:n for i, n in enumerate(self._noise_list)}

			elif (len(res) == 4 and 
				noise_type == 'dropout'):
				X, y, prob_placeholder, logits = res
				self.X, self.y, probs, logits_list = copy_and_duplicate(X, y, logits, 
					self._n_replicas, self._graph, prob_placeholder)

				# _noise_plcholders stores dropout plcholders: {replica_id:plcholder}
				# it is used also to store summaries
				self._noise_plcholders = {i:p for i, p in enumerate(probs)}
				
				# in case of noise_type == dropout, _curr_noise_dict stores
				# probabilities for keeping optimization parameters 
				# (W's and b's): {replica_id:keep_proba}
				self._curr_noise_dict = {i:n 
					for i, n in enumerate(sorted(self._noise_list, reverse=True))}

			else: 
				raise InvalidArchitectureFuncError(len(res), self._noise_type)

		except:
			raise

		# from here, whole net that goes after logits is created
		self.__DEBUG__logits_list = logits_list
		self.__DEBUG__routes = [] # remove this !
		self._cross_entropy_loss_dict = {}
		self._zero_one_loss_dict = {}
		self._optimizer_dict = {}
		
		# special vals for summary:
		self.swap_accept_ratio = 0.0
		self.n_swap_attempts = 0
		self.latest_accept_proba = 1.0
		self.latest_swapped_pair = -1 
		self.replica_swap_ratio = {i:0.0 for i in range(self._n_replicas)}
		self.ordered_swap_ratio = {i:0.0 for i in range(self._n_replicas)}
		self.replica_n_swap_attempts = {i:0 for i in range(self._n_replicas)}
		self.ordered_n_swap_attempts = {i:0 for i in range(self._n_replicas)}
		
		with self._graph.as_default():
			for i in range(self._n_replicas):
				
				with tf.name_scope('Metrics' + str(i)):
					
					self._cross_entropy_loss_dict[i] = self._cross_entropy_loss(self.y, 
						logits_list[i])
					
					self._zero_one_loss_dict[i] = self._zero_one_loss(self.y, 
						logits_list[i])

				with tf.name_scope('Optimizer_' + str(i)):

					if noise_type == 'random_normal':
						Optimizer = NormalNoiseGDOptimizer
					elif noise_type == 'betas':
						Optimizer = GDLDOptimizer
					elif noise_type == 'LDSampler':
						Optimizer = LDSampler
					else:
						Optimizer = GDOptimizer
					"""	
					Optimizer = (NormalNoiseGDOptimizer 
						if noise_type == 'random_normal' else GDOptimizer)
					"""
					optimizer = Optimizer(self._learning_rate, i,
						noise_list=self._noise_list)
					
					self._optimizer_dict[i] = optimizer

					if (self._loss_func_name == 'cross_entropy' or 
						self._loss_func_name == 'crossentropy'):
						optimizer.minimize(self._cross_entropy_loss_dict[i])
					elif self._loss_func_name == 'zero_one_loss':
						optimizer.minimize(self._zero_one_loss_dict[i])
					else:
						raise ValueError('Invalid loss function name. ',
							'Available functions are: cross_entropy/zero_one_loss,',
							'But given:', self._loss_func_name)
			
			self._summary = Summary(self._graph, self._n_replicas, self._name, 
				self._cross_entropy_loss_dict, self._zero_one_loss_dict, self._noise_list, 
				self._noise_plcholders, simulation_num, 
				self._optimizer_dict, summary_type=self._summary_type)

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
			(placeholders keep_prob for each replica).
		"""

		feed_dict = {self.X:X_batch, self.y:y_batch}

		if dataset_type == 'test':
			d = {self._noise_plcholders[i]:1.0
				for i in range(self._n_replicas)}

		elif dataset_type == 'validation':
			d = {self._noise_plcholders[i]:1.0
				for i in range(self._n_replicas)}
			d.update({
				self._summary.swap_accept_ratio_plcholder:self.swap_accept_ratio,
				self._summary.accept_proba_plcholder:self.latest_accept_proba,
				self._summary.swap_replica_pair_plcholder:self.latest_swapped_pair})
			temp_dict1 = {self._summary.replica_accept_ratio_plcholders[i]:self.replica_swap_ratio[i]
				for i in range(self._n_replicas)}
			temp_dict2 = {self._summary.ordered_accept_ratio_plcholders[i]:self.ordered_swap_ratio[i]
				for i in range(self._n_replicas)}
			d.update(temp_dict1)
			d.update(temp_dict2)
			
		elif dataset_type == 'train':
			d = {self._noise_plcholders[i]:self._curr_noise_dict[i]
				for i in range(self._n_replicas)}
		
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

		loss = [self._cross_entropy_loss_dict[i] for i in range(self._n_replicas)]
		zero_one_loss = [self._zero_one_loss_dict[i] for i in range(self._n_replicas)]
		summary = self._summary.get_summary_ops(dataset_type)

		if dataset_type == 'test' or dataset_type == 'validation':
			return loss + zero_one_loss + summary
		elif dataset_type == 'train':
			train_op = [self._optimizer_dict[i].get_train_op() 
				for i in range(self._n_replicas)]
			#if __DEBUG__: train_op = train_op + self.__DEBUG__logits_list
			return loss + zero_one_loss + summary + train_op
		else:
			raise InvalidDatasetTypeError()

	def add_summary(self, evaluated, step, dataset_type='train'):
		summs = self.extract_evaluated_tensors(evaluated, 'summary')
		self._summary.add_summary(summs, step, dataset_type)

	def extract_evaluated_tensors(self, evaluated, tensor_type):
		
		if tensor_type == 'cross_entropy':
			return evaluated[:self._n_replicas]
		
		elif tensor_type == 'zero_one_loss':
			return evaluated[self._n_replicas:2*self._n_replicas]

		elif tensor_type == 'summary':
			end_mult = (4 if self._summary_type is None else 3)
			if len(evaluated) % self._n_replicas == 0:
				return evaluated[2*self._n_replicas:end_mult*self._n_replicas]
			else:
				# special summary case
				return evaluated[2*self._n_replicas:end_mult*self._n_replicas + 1]
		else:
			raise InvalidLossFuncError() 

	def update_noise_vals(self, evaluated):
		"""Updates noise values based on loss function.

		If the noise_type is random_normal then the optimizaiton 
		route is updated. If noise_type is dropout, then the dropout
		placeholders are updated.

		Args:
			evaluated: a list as returned by sess.run(get_train_ops())
		"""

		loss_list = self.extract_evaluated_tensors(evaluated, 'cross_entropy')
		losses_and_ids = [(l, i) for i, l in enumerate(loss_list)]
		
		losses_and_ids.sort(key=lambda x: x[0])
		
		for i, li in enumerate(losses_and_ids):
			self._curr_noise_dict[li[1]] = self._noise_list[i]
		
		if self._noise_type == 'random_normal':
			for i, li in enumerate(losses_and_ids):
				self._optimizer_dict[li[1]].set_train_route(i)

	def swap_replicas(self, evaluated):
		"""Swaps between replicas.

		Swaps according to:
			1. Uniformly randomly select a pair of adjacent temperatures
				1/beta_i and 1/beta_i+1, for which swap move is proposed.
			2. Compute the acceptance ratio for the proposed swap. 
			If surface_view is 'information', accept with probability:
				min{1, exp((beta_i-beta_i+1)*(loss_i/beta_i-loss_i+1/beta_i+1)}
			if surface_view is 'energy', accept with probability:
				min{1, exp((beta_i-beta_i+1)*(loss_i-loss_i+1)} 
		
		"""
		random_pair = random.choice(range(self._n_replicas - 1)) # pair number
		

		beta = [self._curr_noise_dict[x] for x in range(self._n_replicas)]
		beta_id = [(b, i) for i, b in enumerate(beta)]
		beta_id.sort(key=lambda x: x[0], reverse=True)

		i = beta_id[random_pair][1]
		j = beta_id[random_pair+1][1]
	
		loss_list = self.extract_evaluated_tensors(evaluated, 'cross_entropy')

		#losses_and_ids = [(l, x) for x, l in enumerate(loss_list)]
		#losses_and_ids.sort(key=lambda x: x[0])
		sorted_losses = sorted(loss_list)

		sorted_i = sorted_losses.index(loss_list[i])
		sorted_j = sorted_losses.index(loss_list[j])

		self.n_swap_attempts += 1
		self.replica_n_swap_attempts[i] += 1
		self.replica_n_swap_attempts[j] += 1
		self.ordered_n_swap_attempts[sorted_i] += 1
		self.ordered_n_swap_attempts[sorted_j] += 1

		if self._surface_view == 'information':
			l1, l2 = loss_list[i]/beta[i], loss_list[j]/beta[j]
		else:
			l1, l2 = loss_list[i], loss_list[j] # energy
		
		accept_proba = np.exp((l1-l2)*(beta[i] - beta[j]))
		self.latest_accept_proba = accept_proba
		

		if np.random.uniform() < accept_proba:
			self._curr_noise_dict[i] = beta[j]
			self._curr_noise_dict[j] = beta[i]

			self._optimizer_dict[i].set_train_route(j)
			self._optimizer_dict[j].set_train_route(i)

			self.swap_accept_ratio = (((self.n_swap_attempts - 1)/self.n_swap_attempts)*self.swap_accept_ratio 
				+ (1/self.n_swap_attempts))
			self.latest_swapped_pair = i
			
			for x in [i, j]:
				n = self.replica_n_swap_attempts[x]
				ratio = self.replica_swap_ratio[x]
				self.replica_swap_ratio[x] = ((n - 1)/n)*ratio + (1/n)
			for x in [sorted_j, sorted_i]:
				n = self.ordered_n_swap_attempts[x]
				ratio = self.ordered_swap_ratio[x]
				self.ordered_swap_ratio[x] = ((n - 1)/n)*ratio + (1/n)
		else:
			
			self.latest_swapped_pair = -1
			self.swap_accept_ratio = (((self.n_swap_attempts - 1)/self.n_swap_attempts)*self.swap_accept_ratio)
			
			for x in [i, j]:
				n = self.replica_n_swap_attempts[x]
				ratio = self.replica_swap_ratio[x]
				self.replica_swap_ratio[x] = ((n - 1)/n)*ratio 
			for x in [sorted_j, sorted_i]:
				n = self.ordered_n_swap_attempts[x]
				ratio = self.ordered_swap_ratio[x]
				self.ordered_swap_ratio[x] = ((n - 1)/n)*ratio 

	def get_tf_graph(self): return self._graph
		
	def _store_tf_graph(self, path): tf.summary.FileWriter(path, self.graph).close()
	
	def _cross_entropy_loss(self, y, logits, clip_value_max=100.0):
		with tf.name_scope('cross_entropy'):
			with tf.device('/cpu:0'):
				cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
					labels=y, logits=logits)
				loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
				if clip_value_max is not None:
					loss = tf.clip_by_value(loss, 0.0, clip_value_max)
		return loss

	def _zero_one_loss(self, y, logits):
		with tf.name_scope('zero_one_loss'):
			with tf.device('/cpu:0'):
				y_pred = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
				
				zero_one_loss = 1.0 - tf.reduce_mean(tf.cast(x=y_pred, dtype=tf.float32), 
					name='zero_one_loss')
		return zero_one_loss


	


	




