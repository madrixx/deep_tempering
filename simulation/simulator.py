import sys
import os
import gc

import tensorflow as tf
import json
import numpy as np

from simulation.simulation_builder.graph_builder import GraphBuilder
from simulation.simulation_builder.summary import Dir
from simulation import simulator_utils as s_utils

class Simulator(object):
	"""Performs single/multiple simulation for calculating averages.

	This class defines the API for performing simulations. This class 
	trains models (possibly multiple times), while class GraphBuilder 
	creates dataflow graphs with duplicated replicas. More functions
	can be added to train models in different setups.
	
	### Usage

	```python
	from tensorflow.examples.tutorials.mnist import input_data
	import numpy as np

	from simulation.simulator import Simulator
	from simulation.summary_extractor2 import SummaryExtractor
	import simulation.simulator_utils as s_utils
	from simulation.architectures.mnist_architectures import nn_mnist_architecture_dropout
	MNIST_DATAPATH = 'simulation/data/mnist/'

	mnist = input_data.read_data_sets(MNIST_DATAPATH)
	train_data = mnist.train.images
	train_labels = mnist.train.labels
	test_data = mnist.test.images
	test_labels = mnist.test.labels
	valid_data = mnist.validation.images
	valid_labels = mnist.validation.labels

	n_replicas = 8
	separation_ratio = 1.21

	# set simulation parameters
	architecture_func = nn_mnist_architecture_dropout
	learning_rate = 0.01
	noise_list = [1/separation_ratio**i for i in range(n_replicas)]
	noise_type = 'dropout_rmsprop'
	batch_size = 200
	n_epochs = 50
	name = 'test_simulation' # simulation name
	test_step = 300 # 1 step==batch_size
	swap_attempt_step = 300
	burn_in_period = 400
	loss_func_name = 'cross_entropy'
	description = 'RMSProp with dropout.'
	rmsprop_decay = 0.9
	rmsprop_momentum = 0.001
	rmsprop_epsilon=1e-6

	# make sure that there are no directories that were previously created with same name
	# otherwise, there will be problems extracting simulated results
	s_utils.clean_dirs('simulation/summaries/' + name)
	s_utils.clean_dirs('simulation/summaries/compressed/' + name)

	# create and run simulation

	sim = Simulator(
		architecture=architecture_func,
		learning_rate=learning_rate,
		noise_list=noise_list,
		noise_type='dropout_rmsprop',
		batch_size=batch_size,
		n_epochs=n_epochs,
	    test_step=test_step,
		name=name,
		swap_attempt_step=swap_attempt_step,
		burn_in_period=burn_in_period,
		loss_func_name='cross_entropy',
	    description=description,
	    rmsprop_decay=rmsprop_decay,
	    rmsprop_epsilon=rmsprop_epsilon,
	    rmsprop_momentum=rmsprop_momentum
		)

	sim.train(train_data=train_data, train_labels=train_labels,
		test_data=test_data, test_labels=test_labels, 
		validation_data=valid_data, validation_labels=valid_labels)


	# plot results
	se = SummaryExtractor(name)
	se.print_report(mixing_log_y=separation_ratio)

	```
	"""
	
	def __init__(self, architecture, learning_rate, noise_list, noise_type, 
		batch_size, n_epochs, name, n_simulations=1, summary_type=None, 
		test_step=500, swap_attempt_step=500, temp_factor=None, 
		tuning_parameter_name=None, burn_in_period=None, 
		loss_func_name='cross_entropy', 
		surface_view='energy', description=None,
		rmsprop_decay=0.9, rmsprop_momentum=0.001, rmsprop_epsilon=1e-6):
		"""Creates a new simulator object.
		
		Args:
			architecture: A function that creates inference model (e.g. 
				see simulation.architectures.nn_mnist_architecture)
			learning_rate: Learning rate for optimizer
			noise_list: A list (not np.array!) for noise/temperatures/dropout
				values
			noise_type: A string specifying the noise type and optimizer to apply.
				Possible values could be seen at 
				simulation.simulation_builder.graph_builder.GraphBuilder.__noise_types
			batch_size: Batch Size
			n_epochs: Number of epochs for each simulation
			n_simulatins: Number of simulation to run.
			summary_type: Specifies what summary types to store. Detailed possibilities
				could be seen in 
				simulation.simulation_builder.graph_builder.Summary.
				Default is None (if None stores all summaries)
			test_step: An integer specifing an interval of steps to perform until
				running a test dataset (1 step equals batch_size)
			swap_attempt_step: An integer specifying an interval to perform until
				attempting to swap between replicas based on validation dataset.
			temp_factor: A separation ratio between two adjacent temperatures. 
				This value is not important for simulation because the 
				noise_list already contains the separated values. This value is
				(as well as some others) are stored in the simulation 
				description file (this file is created by _log_params() 
				function).
			tuning_parameter_name: As the temp_factor value, this argument is 
				also not important for simulation. It is stored in the description
				file as well.
			burn_in_period: A number of steps until the swaps start to be 
				proposed.
			loss_func_name: A function which we want to optimize. Currently, 
				only cross_entropy and stun (stochastic tunneling) are 
				supported.
			surface_view: 'information' or 'energy'. See 
				GraphBuilder.swap_replicas() for detailed explanation.
			description: A custom string that is stored in the description file.
			rmsprop_decay: Used in  
				simulation.simulation_builder.optimizers.RMSPropOptimizer 
				for noise type 'dropout_rmsprop'. This value is ignored for 
				other noise_types.
			rmsprop_momentum: Used in  
				simulation.simulation_builder.optimizers.RMSPropOptimizer 
				for noise type 'dropout_rmsprop'. This value is ignored for 
				other noise_types.
			rmsprop_epsilon: Used in  
				simulation.simulation_builder.optimizers.RMSPropOptimizer 
				for noise type 'dropout_rmsprop'. This value is ignored for 
				other noise_types.
			
		"""
		
		
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

		self.rmsprop_decay = rmsprop_decay
		self.rmsprop_momentum = rmsprop_momentum
		self.rmsprop_epsilon = rmsprop_epsilon

		if n_simulations == 1:
			self.graph = GraphBuilder(self.architecture, self.learning_rate, 
				self.noise_list, self.name, self.noise_type, 
				self.summary_type, simulation_num=0, surface_view=self.surface_view,
				loss_func_name=self.loss_func_name,
				rmsprop_decay=self.rmsprop_decay, rmsprop_momentum=self.rmsprop_momentum, 
				rmsprop_epsilon=self.rmsprop_epsilon)
		

	def train_n_times(self, train_func, *args, **kwargs):
		"""Trains `n_simulations` times using the same setup."""
		sim_names = []
		for i in range(self.n_simulations):
			self.graph = GraphBuilder(self.architecture, self.learning_rate, 
				self.noise_list, self.name, self.noise_type, 
				self.summary_type, simulation_num=i, surface_view=self.surface_view,
				loss_func_name=self.loss_func_name,
				rmsprop_decay=self.rmsprop_decay, rmsprop_momentum=self.rmsprop_momentum, 
				rmsprop_epsilon=self.rmsprop_epsilon)
			
			sim_names.append(self.graph._summary.dir.name)
			train_func(kwargs)

			gc.collect()
		
		[s_utils.extract_and_remove_simulation(n) for n in sim_names]

	def explore_heat_capacity(self, train_func, betas, *args, **kwargs):
		"""Used for exploring heat capacity function."""

		sim_names = []
		i = 0
		for beta_0 in betas:
			for j in range(self.n_simulations):
				noise_list = [beta_0, temp_factor*beta_0]
				self.graph = GraphBuilder(self.architecture, self.learning_rate, 
					noise_list, self.name, self.noise_type, 
					self.summary_type, simulation_num=i, surface_view=self.surface_view,
					loss_func_name=self.loss_func_name,
					rmsprop_decay=self.rmsprop_decay, rmsprop_momentum=self.rmsprop_momentum, 
					rmsprop_epsilon=self.rmsprop_epsilon)
			
				sim_names.append(self.graph._summary.dir.name)
				train_func(kwargs)
				gc.collect()
				i += 1

	
	def train_PTLD(self, kwargs):
		"""Trains and swaps between replicas"""
	
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
							if step % self.test_step == 0 or step == 1:
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
		self.train_PTLD(kwargs)
		"""
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
				#valid_feed_dict = g.create_feed_dict(
				#	valid_data, valid_labels, 'validation')
				#evaluated = sess.run(g.get_train_ops('validation'),
				#	feed_dict=valid_feed_dict)
				#g.add_summary(evaluated, step, dataset_type='validation')
				#g.swap_replicas(evaluated)
				#g._summary.flush_summary_writer()
				
				valid_summary_step = (self.swap_attempt_step 
					if self.swap_attempt_step >= self.test_step else np.ceil(self.test_step/self.swap_attempt_step))

								
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
								
							### validation ###
							if step % self.swap_attempt_step == 0:
								
								valid_feed_dict = g.create_feed_dict(
									valid_data, valid_labels, 'validation')
								evaluated = sess.run(g.get_train_ops('validation'),
									feed_dict=valid_feed_dict)
								
								g.add_summary(evaluated, step, dataset_type='validation')
								#valid_inc = (valid_inc + 1) % valid_summary_step
								if step > self.burn_in_period:
									g.swap_replicas(evaluated)
								else:
									g.swap_accept_ratio = 0
								g._summary.flush_summary_writer()
						
						except tf.errors.OutOfRangeError:
							sess.run(iterator.initializer)
							break

				
				g._summary.close_summary_writer()
				"""

	

	
	def _log_params(self, desciption):
		"""Creates a description file."""
		dirpath = self._dir.log_dir
		filepath = os.path.join(dirpath, 'description.json')
		if not os.path.exists(dirpath):
			os.makedirs(dirpath)
		_log = {
			'name':self.name,
			'noise_type': self.noise_type,
			'noise_list': self.noise_list,
			'n_replicas': len(self.noise_list),
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
		"""Helper for logs during training."""
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
