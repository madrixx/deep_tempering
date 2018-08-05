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

# if SUMMARY_TYPE is None, then stores per worker summary and 
# ordered summaries. Can be only 'ordered_summary' or 'worker_summary'
# or None
SUMMARY_TYPE = None 

class GraphBuilder(object):

	def __init__(self, architecture, learning_rate, noise_list, name, noise_type='random_normal'):
		
		self._architecture = architecture
		self._learning_rate = learning_rate
		self._noise_list = (sorted(noise_list) 	if noise_type == 'random_normal'
												else sorted(noise_list, reverse=True))
		self._n_workers = len(noise_list)
		self._noise_type = noise_type
		self._name = name
		self._graph = tf.Graph()

		# create graph with duplicates based on architecture function 
		# and noise type

		try:
			res = self._architecture(tf.Graph())
			if (len(res) == 3 and 
				noise_type == 'random_normal'):
				X, y, logits = res
				self.X, self.y, logits_list = copy_and_duplicate(X, y, logits, 
					self._n_workers, self._graph)

				# curr_noise_dict stores {worker_id:current noise stddev VALUE}  
				self._curr_noise_dict = {i:n for i, n in enumerate(self._noise_list)}

			elif (len(res) == 4 and 
				noise_type == 'dropout'):
				X, y, prob_placeholder, logits = res
				self.X, self.y, probs, logits_list = copy_and_duplicate(X, y, logits, 
					self._n_workers, self._graph, prob_placeholder)

				# _probs_dict dropout placeholder for each key==worker_id
				self._probs_dict = {i:p for i, p in enumerate(probs)}
				
				# in case of noise_type == dropout, _curr_noise_dict stores
				# probabilities for keeping parameters:
				# {worker_id:keep_proba}
				self._curr_noise_dict = {i:n 
					for i, n in enumerate(sorted(self._noise_list, reverse=True))}


			else: 
				raise ValueError()

		except ValueError:
			raise ValueError('`architecture` function must return' + \
			 	' 4 variables for noise_type `dropout` and' + \
				' 3 variables for noise_type `random_normal`')

		except:
			raise

		# from here, whole net that goes after logits is created
			
		self._loss_dict = {}
		self._acc_dict = {}
		self._optimizer_dict = {}

		with self._graph.as_default():
			for i in range(self._n_workers):
				
				with tf.name_scope('LossAccuracy_' + str(i)):
					
					self._loss_dict[i] = self._cross_entropy_loss(	self.y, 
																	logits_list[i])
					
					self._acc_dict[i] = self._accuracy(	self.y, 
														logits_list[i])

				with tf.name_scope('Optimizer_' + str(i)):

					Optimizer = (NormalNoiseGDOptimizer if noise_type == 'random_normal'
														else GDOptimizer)
					optimizer = Optimizer(	self._learning_rate, 
											i,
											noise_list=self._noise_list)
					self._optimizer_dict[i] = optimizer
					optimizer.minimize(self._loss_dict[i])
			
			self.summary = Summary(self.graph, self._n_workers, self._name, 
				self._loss_dict, self._acc_dict, self._noise_list)

			self.variable_initializer = tf.global_variables_initializer()


	def create_feed_dict(self, X_batch, y_batch, test=False):

		feed_dict = {self.X:X_batch, self.y:y_batch}

		if self._noise_type == 'dropout':
			
			if test:
				d = {self._probs_dict[i]:1.0 for i in range(self._n_workers)}
			
			else:
				d = {self._probs_dict[i]:self._curr_noise_dict[i]
					for i in range(self._n_workers)}
			
			feed_dict.update(d)
		
		return feed_dict

	def get_train_ops(self, test=False):
		
		loss = [self._loss_dict[i] for i in range(self._n_workers)]
		accuracy = [self._acc_dict[i] for i in range(self._n_workers)]
		summary = self.summary.get_summary_ops()

		if test:
			return loss + accuracy + summary
		
		train_op = [self._optimizer_dict[i].get_train_op() 
			for i in range(self._n_workers)]

		return loss + accuracy + summary + train_op

	def add_summary(self, evaluated, step, dataset_type='train'):
		

		def add_worker_summary():

			summary = self.extract_evaluated_tensors(evaluated, 'summary')
			writer = self.summary.get_summary_writer(dataset_type, 
				'worker_summary')
			
			noise_dict = self.summary.worker_noise_dict

			feed_dict = {noise_dict[i]:self._curr_noise_dict[i]
				for i in range(self._n_workers)}
			for i in range(self._n_workers):
				#writer[i].add_summary(summary[i], step)
				feed_dict[i]
		def add_ordered_summary():

			accuracy = self.extract_evaluated_tensors(evaluated, 'accuracy')
			loss = self.extract_evaluated_tensors(evaluated, 'loss')

			loss_acc_id = [(l, a, i) 
				for l, a, i in zip(loss, accuracy, range(self._n_workers))]
			loss_acc_id.sort(key=lambda x: x[0])

			feed_dict = {}
			for i in range(self._n_workers):
				feed_dict[]
	def extract_evaluated_tensors(self, evaluated, tensor_type):
		
		if tensor_type == 'loss':
			return evaluated[:self._n_workers]
		
		elif tensor_type == 'accuracy':
			return evaluated[self._n_workers:2*self._n_workers]

		elif tensor_type == 'summary':
			return evaluated[2*self._n_workers:3*self._n_workers]

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

		sys.exit()
		
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

	

	#def _create_summaries(self):



class Summary(object):
	"""Helper class for creating summaries."""

	def __init__(self, graph, n_workers, name, loss_dict, acc_dict, 
		noise_list, summary_type=None):
		self.graph = graph
		self.n_workers = n_workers
		self.log_dir = os.path.abspath(
			'/'.join(__file__.split('/')[:-2])) + '/summaries/' + name + '/'
		self.name = name
		self.dir = Dir(self.log_dir)
		self.summary_type = summary_type
		
		self.writer_dict = {
			'train_ordered':{},
			'test_ordered':{},
			'valid_ordered':{},
			'train_worker':{},
			'test_worker':{},
			'valid_worker':{}
		}

		self.ordered_summary = {}
		self.worker_summary = {}
	
	def get_summary_ops(self):
		#ordered_summ = [self.ordered_summary[i] for i in range(self.n_workers)]
		if SUMMARY_TYPE == 'ordered_summary':
			return []

		worker_summ = [self.worker_summary[i] for i in range(self.n_workers)]
		return worker_summ

	def get_summary_writer(dataset_type, summary_type):
		if summary_type == 'worker_summary':
			if dataset_type == 'train':
				return self.writer_dict['train_worker']
			elif dataset_type == 'test':
				return self.writer_dict['test_worker']
			elif dataset_type == 'validation':
				return self.writer_dict['valid_worker']
			else:
				raise ValueError("""dataset_type can be one of `train`, `test`
					or `validation`.""")
		elif summary_type == 'ordered_summary':
			if dataset_type == 'train':
				return self.writer_dict['train_ordered']
			elif dataset_type == 'test':
				return self.writer_dict['test_ordered']
			elif dataset_type == 'validation':
				return self.writer_dict['valid_ordered']
			else:
				raise ValueError("""dataset_type can be one of `train`, `test`
					or `validation`.""")
		else:
			raise ValueError("""`summary_type can be `worker_summary` or
				ordered_summary'""")

	def create_worker_summaries(self, loss_dict, acc_dict):
		# create per-worker summary scalars and per worker summary writers

		# worker_noise_dict stores placeholdes for values for specific 
		# worker
		self.worker_noise_dict = {}
		with tf.name_scope('Summary'):
			for i in range(self.n_workers):
				tf.summary.scalar('loss', loss_dict[i], collections=[i])
				tf.summary.scalar('accuracy', acc_dict[i], collections=[i])
				self.worker_noise_dict[i] = tf.placeholder_with_default(0.0, 
					shape=[])
				tf.summary.scalar('noise', self.worker_noise_dict[i], 
					collections=[i])
				self.worker_summary[i] = tf.summary.merge_all(i)

				self.writer_dict['train_worker'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_train_dir(i),
					graph=self.graph,
					filename_suffix=self.dir.get_filename_suffix())
				self.writer_dict['test_worker'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_test_dir(i),
					graph=self.graph,
					filename_suffix=self.dir.get_filename_suffix())
				self.writer_dict['valid_worker'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_validation_dir(i),
					graph=self.graph,
					filename_suffix=self.dir.get_filename_suffix())

	def create_ordered_summaries(self, ):

		# create ordered summary scalars and ordered summary writers
		with tf.name_scope('Summary'):
			self.ordered_summary = ({'pholders':{'accuracy':{}, 'loss':{}, 
				'noise':{} }, 'summaries':{}})

			self.ordered_summary['pholders']['accuracy'] = {
				i:tf.placeholder_with_default(0.0, shape=[]) 
				for i in range(self.n_workers)
			}
			self.ordered_summary['pholders']['loss'] = {
				i:tf.placeholder_with_default(0.0, shape=[]) 
				for i in range(self.n_workers)
			}
			self.ordered_summary['pholders']['noise'] = {
				i:tf.placeholder_with_default(0.0, shape=[]) 
				for i in range(self.n_workers)
			}

			for i in range(self.n_workers):
				
				tf.summary.scalar('loss', 
					self.ordered_summary['pholders']['loss'][i], 
					collections=[i+self.n_workers])

				tf.summary.scalar('accuracy', 
					self.ordered_summary['pholders']['accuracy'][i],
					collections=[i+self.n_workers])

				tf.summary.scalar('noise',
					self.ordered_summary['pholders']['noise'][i],
					collections=[i+self.n_workers])
				
				self.ordered_summary['summaries'][i] = 
					(tf.summary.merge_all(i+self.n_workers))
			
				self.writer_dict['train_ordered'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_ordered_train_dir(i),
					graph=self.graph, 
					filename_suffix=self.dir.get_filename_suffix())
				self.writer_dict['test_ordered'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_ordered_test_dir(i),
					graph=self.graph, 
					filename_suffix=self.dir.get_filename_suffix())
				self.writer_dict['valid_ordered'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_ordered_validation_dir(i),
					graph=self.graph, 
					filename_suffix=self.dir.get_filename_suffix())

class Dir(object):
	"""Helper class for generating directory names."""

	def __init__(self, summary_path, name):
		self.summary_path = summary_path
		self.name = name

	def get_train_dir(self, worker_id):
		"""Returns the name of a train dir to store summaries.

		Args:
			worker_id:  An integer. The worker_id to which the result summaries
						belongs to.
			"""
		return self.summary_path + 'train_' + str(worker_id)

	def get_test_dir(self, worker_id):
		"""Returns the name of a validation_dir to store summaries.

		Args:
			worker_id:  An integer. The worker_id to which the result summaries
						belongs to.
		"""
		return self.summary_path + 'test_' + str(worker_id)

	def get_validation_dir(self, worker_id):
		"""Returns the name of a validation dir to store summaries.

		Args:
			worker_id:  An integer. The worker_id to which the result summaries
						belongs to.
		"""
		return self.summary_path + 'validation_' + str(worker_id)

	def get_ordered_train_dir(self, k):
		"""Returns the name of the ordered train dir to store summaries.

		Args:
			k: An integer. Corresponds to the k'th lowest loss.
		"""
		return self.summary_path + 'train_ordered_' + str(k)

	def get_ordered_test_dir(self, k):
		"""Returns the name of the ordered test dir to store summaries.

		Args:
			k: An integer. Corresponds to the k'th lowest loss.
		"""
		return self.summary_path + 'test_ordered_' + str(k)

	def get_ordered_validation_dir(self, k):
		"""Returns the name of the ordered validation dir to store summaries.

		Args:
			k: An integer. Corresponds to the k'th lowest loss.
		"""
		return self.summary_path + 'validation_ordered_' + str(k)

	def get_filename_suffix(self): return self.name

	def clean_dirs(self, dir):
		"""Recursively removes all train, test and validation summary files \
				and folders from previos training life cycles."""
		try:
			for file in os.listdir(dir):
				if os.path.isfile(os.path.join(dir, file)):
					os.remove(os.path.join(dir, file))
				else:
					self.clean_dirs(dir=os.path.join(dir, file))
			
			if dir == self.summary_path:
				for file in os.listdir(dir):
					os.rmdir(os.path.join(dir, file))
		except OSError:
			# if first simulation, nothing to delete
			return
