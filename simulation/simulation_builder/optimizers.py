from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops
import collections
import random
import sys

from simulation.simulation_builder.device_placer import _gpu_device_name

class Optimizer(object):
	"""Wrapper for tf.train.GradientDescentOptimizer"""
	def __init__(self, learning_rate, worker_id, noise_list=None):
		self.learning_rate = learning_rate
		self.worker_id = worker_id
		self.noise_list = noise_list
		self.tf_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

	def minimize(self, loss):
		grads_and_vars = self.compute_gradients(loss)
		train_op = self.apply_gradients(grads_and_vars)
		return train_op

	def compute_gradients(self, loss):
		var_list = self._get_dependencies(loss)
		with tf.device(_gpu_device_name(self.worker_id)):
			grads_and_vars = self.tf_optimizer.compute_gradients(loss, var_list)
		return grads_and_vars

	def apply_gradients(self, grads_and_vars):
		"""Applies gradients.
		
		Args:
			grads_and_vars:	list of tuples as returned by optimizer.compute_gradients()

		Returns:
			An op for gradient computation.
		"""
		with tf.device(_gpu_device_name(self.worker_id)):
			op = [	tf.assign(v, v - self.learning_rate*g) 
					for g, v in grads_and_vars]

			train_op = tf.group(op)
		return train_op

	def _get_dependencies(self, tensor):
		
		_dict = {v.op: v for v in tf.trainable_variables()}
		
		start = tensor.op
		queue = collections.deque()
		queue.append(start)
		visited = set([start])
		variables = []
		while queue:
			op = queue.popleft()
			if op in _dict:
				variables.append(_dict[op])
			else:
				for op_in in op.inputs:
					if op_in.op not in visited:
						queue.append(op_in.op)
						visited.add(op_in.op)
						
		return variables 

class NormalNoiseGDOptimizer(Optimizer):

	def __init__(self, learning_rate, worker_id, noise_list):
		super(NormalNoiseGDOptimizer, self).__init__(	learning_rate, 
														worker_id, 
														noise_list)
		self.noise_list = noise_list
		self.n_routes = len(noise_list)
		self.train_route_dict = {}
		self.current_route = worker_id

	def minimize(self, loss):
		grads_and_vars = self.compute_gradients(loss)
		for route, stddev in enumerate(self.noise_list):
			with tf.name_scope('Route_' + str(route)):
				self.train_route_dict[route] = self.apply_gradients(
													grads_and_vars, 
													stddev)

		return self.train_route_dict[self.current_route]

	def apply_gradients(self, grads_and_vars, stddev):
		"""Applies gradients and adds normal noise with `stddev`.
		
		Args:
			grads_and_vars:	list of tuples as returned by optimizer.compute_gradients()
			stddev:			standard deviation of normal noise

		Returns:
			An op for gradient computation.
		"""

		with tf.device(_gpu_device_name(self.worker_id)):
			op = [	tf.assign(v, v - self.learning_rate*g + tf.random_normal(v.shape, stddev=stddev)) 
					for g, v in grads_and_vars]
			train_op = tf.group(op)
		return train_op

	def set_train_route(self, route):
		self.current_route = route

	def get_train_op(self,):
		
		return self.train_route_dict[self.current_route]

class GDOptimizer(Optimizer):

	def __init__(self, learning_rate, worker_id, noise_list=None):
		super(GDOptimizer, self).__init__(	learning_rate, 
											worker_id)


