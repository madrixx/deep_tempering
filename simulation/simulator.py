import sys
import os
import gc

import tensorflow as tf
import json
import sys

from simulation.simulation_builder.graph_builder import GraphBuilder
from simulation.simulation_builder.summary import Dir
from simulation import simulator_utils as s_utils

class Simulator(object):
	"""
	
	Args:
		architecture:
		learning_rate:
		noise_list:
		noise_type:
		batch_size:
		n_epochs:
		n_simulatins:
		summary_type:
		test_step:
		swap_attempt_step:
		temp_factor:
		tuning_parameter_name:
		surface_view: 'information' or 'energy'. See 
			GraphBuilder.swap_replicas() for detailed explanation.
		description: 

	"""
	def __init__(self, architecture, learning_rate, noise_list, noise_type, 
		batch_size, n_epochs, name, n_simulations=1, summary_type=None, 
		test_step=500, swap_attempt_step=500, temp_factor=None, 
		tuning_parameter_name=None, burn_in_period=None, 
		loss_func_name='cross_entropy', 
		surface_view='information', description=None):
		
		if n_simulations == 1:
			self.graph = GraphBuilder(architecture, learning_rate, noise_list, 
				name, noise_type, summary_type, loss_func_name=loss_func_name)
		
		self.architecture = architecture
		self.learning_rate = learning_rate
		self.noise_type = noise_type
		self.noise_list = noise_list
		self.summary_type = summary_type
		self.learning_rate = learning_rate
		self.name = name
		self.n_simulations = n_simulations
		self.burn_in_period = burn_in_period 
		self.loss_func_name = loss_func_name
			

		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.test_step = test_step
		self.swap_attempt_step = swap_attempt_step
		self.temp_factor = temp_factor
		self.tuning_parameter_name = tuning_parameter_name
		self.surface_view = surface_view
		tf.logging.set_verbosity(tf.logging.ERROR)
		self.delim = "\\" if 'win' in sys.platform else "/"
		self._dir = Dir(name)
		if description:
			self._log_params(description)
		

	def train_n_times(self, train_func, *args, **kwargs):
		"""Trains `n_simulations` times using the same setup."""
		sim_names = []
		for i in range(self.n_simulations):
			self.graph = GraphBuilder(self.architecture, self.learning_rate, 
				self.noise_list, self.name, self.noise_type, 
				self.summary_type, simulation_num=i, surface_view=self.surface_view,
				loss_func_name=self.loss_func_name)
			
			sim_names.append(self.graph._summary.dir.name)
			train_func(kwargs)

			gc.collect()
		
		[s_utils.extract_and_remove_simulation(n) for n in sim_names]
	
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
				"""
				valid_feed_dict = g.create_feed_dict(
					valid_data, valid_labels, 'validation')
				evaluated = sess.run(g.get_train_ops('validation'),
					feed_dict=valid_feed_dict)
				g.add_summary(evaluated, step, dataset_type='validation')
				g.swap_replicas(evaluated)
				g._summary.flush_summary_writer()
				"""
				for epoch in range(self.n_epochs):
					
					while True:
						try:
							step += 1

							### train ###
							batch = sess.run(next_batch)
							feed_dict = g.create_feed_dict(batch['X'], batch['y'])
							evaluated = sess.run(g.get_train_ops(), 
								feed_dict=feed_dict)
							if step % 100 == 0:
								g.add_summary(evaluated, step=step)

							### test ###
							if step % self.test_step == 0:
								evaluated = sess.run(g.get_train_ops('test'),
									feed_dict=test_feed_dict)
								g.add_summary(evaluated, step, dataset_type='test')
								loss = g.extract_evaluated_tensors(evaluated, self.loss_func_name)
								
								self.print_log(loss, epoch, g.swap_accept_ratio, g.latest_accept_proba, step)
								
							### validation + swaps ###
							if step % self.swap_attempt_step == 0:
								
								valid_feed_dict = g.create_feed_dict(
									valid_data, valid_labels, 'validation')
								evaluated = sess.run(g.get_train_ops('validation'),
									feed_dict=valid_feed_dict)
								g.add_summary(evaluated, step, dataset_type='validation')
								if step > self.burn_in_period:
									
									g.swap_replicas(evaluated)


								g._summary.flush_summary_writer()
						
						except tf.errors.OutOfRangeError:
							sess.run(iterator.initializer)
							break

				
				g._summary.close_summary_writer()

	def train(self, **kwargs):
	
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
				"""
				# validation first time
				valid_feed_dict = g.create_feed_dict(
					valid_data, valid_labels, 'validation')
				evaluated = sess.run(g.get_train_ops('validation'),
					feed_dict=valid_feed_dict)
				g.add_summary(evaluated, step, dataset_type='validation')
				g.swap_replicas(evaluated)
				g._summary.flush_summary_writer()
				"""
				
				for epoch in range(self.n_epochs):
					
					while True:
						try:
							step += 1

							### train ###
							batch = sess.run(next_batch)
							feed_dict = g.create_feed_dict(batch['X'], batch['y'])
							evaluated = sess.run(g.get_train_ops(), 
								feed_dict=feed_dict)
							if step % 100 == 0:
								g.add_summary(evaluated, step=step)

							### test ###
							if step % self.test_step == 0:
								evaluated = sess.run(g.get_train_ops('test'),
									feed_dict=test_feed_dict)
								g.add_summary(evaluated, step, dataset_type='test')
								loss = g.extract_evaluated_tensors(evaluated, 'cross_entropy')
								
								self.print_log(loss, epoch, g.swap_accept_ratio, g.latest_accept_proba, step)
								
							### validation ###
							if step % self.swap_attempt_step == 0:
								
								valid_feed_dict = g.create_feed_dict(
									valid_data, valid_labels, 'validation')
								evaluated = sess.run(g.get_train_ops('validation'),
									feed_dict=valid_feed_dict)
								g.add_summary(evaluated, step, dataset_type='validation')
								if step > self.burn_in_period:
									g.swap_replicas(evaluated)
								else:
									g.swap_accept_ratio = 0
								g._summary.flush_summary_writer()
						
						except tf.errors.OutOfRangeError:
							sess.run(iterator.initializer)
							break

				
				g._summary.close_summary_writer()

	

	
	def _log_params(self, desciption):
		dirpath = self._dir.log_dir
		filepath = os.path.join(dirpath, 'description.json')
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		_log = {
			'name':self.name,
			'noise_type': self.noise_type,
			'noise_list': self.noise_list,
			'learning_rate':self.learning_rate,
			'n_epochs':self.n_epochs,
			'batch_size':self.batch_size,
			'swap_attempt_step': self.swap_attempt_step,
			'temp_factor': self.temp_factor,
			'n_simulations': self.n_simulations,
			'tuning_parameter_name':self.tuning_parameter_name,	
			'surface_view':self.surface_view,
			'description':desciption,
			'burn_in_period':self.burn_in_period

		}
		with open(filepath, 'w') as fo:
			json.dump(_log, fo, indent=4)

	def print_log(self, loss, epoch, swap_accept_ratio, latest_accept_proba, step):
		buff = 'epoch:' + str(epoch) + ', step:' + str(step) + ', '
		buff = buff + ','.join([str(l) for l in loss]) + ', '
		buff = buff + 'accept_ratio:' + str(swap_accept_ratio)
		buff = buff + ', proba:' + str(latest_accept_proba) + '         '
		self.stdout_write(buff)
		
	def stdout_write(self, buff):
		sys.stdout.write('\r' + buff)
		sys.stdout.flush()

	def chunks(self, l):
		# TODO: remove eventually
		"""Yield successive batch-sized chunks from l."""
		for i in range(0, len(l), self.batch_size):
			yield l[i:i + n]
