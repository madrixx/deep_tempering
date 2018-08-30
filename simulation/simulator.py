import sys
import os
import gc

import tensorflow as tf
import json
import sys

from simulation.simulation_builder.graph_builder import GraphBuilder
from simulation.simulation_builder.summary import Dir

class Simulator(object):

	def __init__(self, architecture, learning_rate, noise_list, noise_type, 
		batch_size, n_epochs, name, n_simulations=1, summary_type=None, 
		test_step=200, swap_attept_step=500, description=None):
		"""
		self.graph = GraphBuilder(architecture, learning_rate, noise_list, 
			name, noise_type, summary_type)
		"""
		self.architecture = architecture
		self.learning_rate = learning_rate
		self.noise_type = noise_type
		self.noise_list = noise_list
		self.summary_type = summary_type
		self.learning_rate = learning_rate
		self.name = name
		self.n_simulations = n_simulations

		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.test_step = test_step
		self.swap_attept_step = swap_attept_step
		tf.logging.set_verbosity(tf.logging.ERROR)
		self.delim = "\\" if 'win' in sys.platform else "/"
		self._dir = Dir(name)
		if description:
			self._log_params(description)
		

	def train_n_times(self, train_func, *args, **kwargs):
		for i in range(self.n_simulations):
			self.graph = GraphBuilder(self.architecture, self.learning_rate, 
				self.noise_list, self.name + '_s_' + str(i), self.noise_type, self.summary_type)
			train_func(kwargs)
			gc.collect()
	
	def train_PTLD(self, kwargs):
	
		try:
			g = self.graph
			train_data = kwargs.get('train_data', None)
			train_labels = kwargs.get('train_labels', None)
			test_data = kwargs.get('test_data', None)
			test_labels = kwargs.get('test_labels', None)
			valid_data = kwargs.get('validation_data', None)
			valid_labels = kwargs.get('validation_labels', None)
			test_feed_dict = g.create_feed_dict(test_data, test_labels, 
				dataset_type='test')
			with g.get_tf_graph().as_default():
				data = tf.data.Dataset.from_tensor_slices({
					'X':train_data,
					'y':train_labels
					}).batch(self.batch_size)
				iterator = data.make_initializable_iterator()
		except:
			raise

		with g.get_tf_graph().as_default():

			step = 0
			
			with tf.Session() as sess:
				sess.run(iterator.initializer)
				sess.run(g.variable_initializer)
				next_batch = iterator.get_next()

				# validation first time
				valid_feed_dict = g.create_feed_dict(
					valid_data, valid_labels, 'validation')
				evaluated = sess.run(g.get_train_ops('validation'),
					feed_dict=valid_feed_dict)
				g.add_summary(evaluated, step, dataset_type='validation')
				g.swap_replicas(evaluated)
				g._summary.flush_summary_writer()
				
				for epoch in range(self.n_epochs):
					
					while True:
						try:
							step += 1

							# train
							batch = sess.run(next_batch)
							feed_dict = g.create_feed_dict(batch['X'], batch['y'])
							evaluated = sess.run(g.get_train_ops(), 
								feed_dict=feed_dict)
							if step % 100 == 0:
								g.add_summary(evaluated, step=step)

							# test
							if step % self.test_step == 0:
								evaluated = sess.run(g.get_train_ops('test'),
									feed_dict=test_feed_dict)
								g.add_summary(evaluated, step, dataset_type='test')
								loss = g.extract_evaluated_tensors(evaluated, 'cross_entropy')
								buff = 'epoch:' + str(epoch) + ', step:' + str(step) + ', '
								buff = buff + ','.join([str(l) for l in loss]) + ', '
								buff = buff + 'accept_ratio:' + str(g.swap_accept_ratio)
								buff = buff + ', proba:' + str(g.latest_accept_proba) + ', '
								buff = buff + str(g.latest_swapped_pair)
								self.stdout_write(buff)

							if step % self.swap_attept_step == 0:
								# validation
								valid_feed_dict = g.create_feed_dict(
									valid_data, valid_labels, 'validation')
								evaluated = sess.run(g.get_train_ops('validation'),
									feed_dict=valid_feed_dict)
								g.add_summary(evaluated, step, dataset_type='validation')
								g.swap_replicas(evaluated)

								g._summary.flush_summary_writer()
								#g._summary.close_summary_writer()
						except tf.errors.OutOfRangeError:
							sess.run(iterator.initializer)
							break

				
				g._summary.close_summary_writer()

	

	def train(self, *args, **kwargs):

		def _prepare_data():
			with g.get_tf_graph().as_default():
				train_data = kwargs.get('train_data', None)
				train_labels = kwargs.get('train_labels', None)
				test_data = kwargs.get('test_data', None)
				test_labels = kwargs.get('test_labels', None)
				valid_data = kwargs.get('validation_data', None)
				valid_labels = kwargs.get('validation_labels', None)

				test_feed_dict = g.create_feed_dict(test_data, test_labels, 
					dataset_type='test')
				valid_feed_dict = g.create_feed_dict(valid_data, valid_labels,
					dataset_type='validation')

				data = tf.data.Dataset.from_tensor_slices({
					'X':train_data,
					'y':train_labels
					}).batch(self.batch_size)
				train_iter = data.make_initializable_iterator()

			return test_feed_dict, valid_feed_dict, train_iter

		try:
			g = self.graph
			test_feed_dict, valid_feed_dict, iterator = _prepare_data()

		except:
			raise
		

		with g.get_tf_graph().as_default():

			step = 0
			
			with tf.Session() as sess:
				sess.run(iterator.initializer)
				sess.run(g.variable_initializer)
				next_batch = iterator.get_next()

				# validation first time
				evaluated = sess.run(g.get_train_ops('validation'),
					feed_dict=valid_feed_dict)
				g.add_summary(evaluated, step, dataset_type='validation')
				g.update_noise_vals(evaluated)

				for epoch in range(self.n_epochs):
					
					while True:
						try:
							step += 1

							# train
							batch = sess.run(next_batch)
							feed_dict = g.create_feed_dict(batch['X'], batch['y'])
							evaluated = sess.run(g.get_train_ops(), 
								feed_dict=feed_dict)
							g.add_summary(evaluated, step=step)

							# test
							if step % self.test_step == 0:
								evaluated = sess.run(g.get_train_ops('test'),
									feed_dict=test_feed_dict)
								g.add_summary(evaluated, step, dataset_type='test')
								loss = g.extract_evaluated_tensors(evaluated, 'loss')
								buff = 'epoch:' + str(epoch) + ', step:' + str(step) + ', '
								buff = buff + ','.join([str(l) for l in loss])
								self.stdout_write(buff)
						except tf.errors.OutOfRangeError:
							sess.run(iterator.initializer)
							break

					# validation
					evaluated = sess.run(g.get_train_ops('validation'),
						feed_dict=valid_feed_dict)
					g.add_summary(evaluated, step, dataset_type='validation')
					g.update_noise_vals(evaluated)
					g._summary.flush_summary_writer()

				g._summary.close_summary_writer()

	
	def _log_params(self, desciption):
		dirpath = self._dir.log_dir
		filepath = os.path.join(dirpath, 'desciption.json')
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		_log = {
			'name':self.name,
			'noise_type': self.noise_type,
			'noise_list': self.noise_list,
			'learning_rate':self.learning_rate,
			'n_epochs':self.n_epochs,
			'batch_size':self.batch_size,
			'swap_attept_step': self.swap_attept_step,
			'description':desciption

		}
		with open(filepath, 'w') as fo:
			json.dump(_log, fo, indent=4)

	def stdout_write(self, buff):
		sys.stdout.write('\r' + buff)
		sys.stdout.flush()

	def chunks(self, l):
		"""Yield successive batch-sized chunks from l."""
		for i in range(0, len(l), self.batch_size):
			yield l[i:i + n]
