"""Defines wrappers for tensorflow optimizers."""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import collections

import tensorflow as tf
from tensorflow.python.framework import ops # pylint: disable=no-name-in-module, unused-import
import numpy as np

from simulation.simulation_builder.device_placer import _gpu_device_name

class Optimizer:
  """Wrapper for tf.train.GradientDescentOptimizer"""
  def __init__(self, learning_rate, replica_id, noise_list=None, # pylint: disable=too-many-arguments
               decay=None, momentum=None, epsilon=None, # pylint: disable=unused-argument
               use_locking=None, centered=None): # pylint: disable=unused-argument
    """decay, momentum, epsilon, use_locking, centered args are for
      RMSPropOptimizer only."""
    self._initializer(learning_rate, replica_id, noise_list)


  def _initializer(self, learning_rate, replica_id, noise_list, # pylint: disable=too-many-arguments
                   decay=None, momentum=None, epsilon=None, # pylint: disable=unused-argument
                   use_locking=None, centered=None): # pylint: disable=unused-argument
    """Initializes optimizer."""
    self.learning_rate = learning_rate
    self.replica_id = replica_id
    self.noise_list = noise_list
    self.tf_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

    self.train_op = None
    self.trainable_variables = None

  def minimize(self, loss):
    """Wrapper for tf.train.Optimizer.minimize()"""
    grads_and_vars = self.compute_gradients(loss)
    train_op = self.apply_gradients(grads_and_vars)
    self.train_op = train_op # pylint: disable=attribute-defined-outside-init
    return train_op

  def compute_gradients(self, loss):
    """Wrapper for tf.train.Optimizer.minimize()"""
    var_list = self._get_dependencies(loss)
    with tf.device(_gpu_device_name(self.replica_id)):
      grads_and_vars = self.tf_optimizer.compute_gradients(loss, var_list)
    return grads_and_vars

  def apply_gradients(self, grads_and_vars):
    """Applies gradients.

    Args:
      grads_and_vars: list of tuples as returned by
        optimizer.compute_gradients()

    Returns:
      An op for gradient computation.
    """
    with tf.device(_gpu_device_name(self.replica_id)):
      ops_ = [tf.assign(v, v - self.learning_rate*g)
              for g, v in grads_and_vars]

      train_op = tf.group(ops_)
    return train_op

  def get_train_op(self,):
    """Returns the current training op."""
    if self.train_op is None:
      raise ValueError('train_op is not set. Call minimize() to set.')
    return self.train_op

  def _get_dependencies(self, tensor):
    """Returns all vars that `tensor` is dependent on."""
    _dict = {v.op: v for v in tf.trainable_variables()}

    start = tensor.op
    queue = collections.deque()
    queue.append(start)
    visited = set([start])
    variables = []
    while queue:
      op_ = queue.popleft()
      if op_ in _dict:
        variables.append(_dict[op_])
      else:
        for op_in in op_.inputs:
          if op_in.op not in visited:
            queue.append(op_in.op)
            visited.add(op_in.op)

    # trainable vars for calculation of displacement from original vals
    if self.trainable_variables is None:
      self.trainable_variables = variables # pylint: disable=attribute-defined-outside-init

    return variables

class NormalNoiseGDOptimizer(Optimizer):
  """Optimizer that adds random noise during training."""
  def __init__(self, learning_rate, replica_id, noise_list):
    super(NormalNoiseGDOptimizer, self).__init__(
        learning_rate, replica_id, noise_list)
    self.noise_list = noise_list
    self.n_routes = len(noise_list)
    self.train_route_dict = {}
    self.current_route = replica_id

  def minimize(self, loss):
    grads_and_vars = self.compute_gradients(loss)
    for route, stddev in enumerate(self.noise_list):
      with tf.name_scope('Route_' + str(route)):
        self.train_route_dict[route] = self.apply_gradients(
            grads_and_vars, stddev)

    return self.train_route_dict[self.current_route]

  def apply_gradients(self, grads_and_vars, stddev): # pylint: disable=arguments-differ
    """Applies gradients and adds normal noise with `stddev`.

    Args:
      grads_and_vars: list of tuples as returned by
        optimizer.compute_gradients()
      stddev:     standard deviation of normal noise

    Returns:
      An op for gradient computation.
    """

    with tf.device(_gpu_device_name(self.replica_id)):
      ops_ = [tf.assign(
          var,
          (var
           - self.learning_rate*grad
           + tf.random_normal(var.shape, stddev=stddev)))
              for grad, var in grads_and_vars]
      train_op = tf.group(ops_)
    return train_op

  def set_train_route(self, route):
    """Sets training route to `route`."""
    self.current_route = route

  def get_train_op(self,):
    if not list(self.train_route_dict.keys()):
      raise ValueError('train_op is not set for Optimizer.',
                       'Call minimize() to set.')
    return self.train_route_dict[self.current_route]

class GDLDOptimizer(NormalNoiseGDOptimizer):
  """Gradient Descent Langevin Dynamics Optimizer"""

  def __init__(self, learning_rate, replica_id, noise_list): # pylint: disable=useless-super-delegation
    super(GDLDOptimizer, self).__init__(learning_rate, replica_id, noise_list)

  def apply_gradients(self, grads_and_vars, beta): # pylint: disable=arguments-differ
    with tf.device(_gpu_device_name(self.replica_id)):

      c = tf.sqrt(np.float32(2*self.learning_rate/beta)) # pylint: disable=invalid-name
      ops_ = [tf.assign(
          v,
          v - self.learning_rate*g + c*tf.random_normal(v.shape, stddev=1))
              for g, v in grads_and_vars]
    return tf.group(ops_)

class GDOptimizer(Optimizer):
  """Wrapper for gradient descent optimizer."""
  def __init__(self, learning_rate, replica_id, noise_list=None):
    super(GDOptimizer, self).__init__(
        learning_rate, replica_id, noise_list)

  def set_train_route(self, route): # pylint: disable=unused-argument, no-self-use
    """Don't do anything. Added for consistency with other optimizers."""
    return

class RMSPropOptimizer(Optimizer):
  """Wrapper for tf.train.RMSPropOptimizer."""
  def __init__( # pylint: disable=too-many-arguments
      self, learning_rate, replica_id, noise_list=None,
      decay=0.9, momentum=0.001, epsilon=1e-6,
      use_locking=False, centered=False):
    super(RMSPropOptimizer, self).__init__(
        learning_rate, replica_id, noise_list, decay, momentum, epsilon,
        use_locking, centered)


  def _initializer( # pylint: disable=too-many-arguments, attribute-defined-outside-init
      self, learning_rate, replica_id, noise_list,
      decay=0.9, momentum=0.001, epsilon=1e-6,
      use_locking=None, centered=None):

    self.replica_id = replica_id
    self.tf_optimizer = tf.train.RMSPropOptimizer(
        learning_rate, decay=decay, momentum=momentum,
        epsilon=epsilon, use_locking=use_locking, centered=centered)

    self.train_op = None
    self.trainable_variables = None


  def minimize(self, loss):
    self.trainable_variables = self._get_dependencies(loss) # pylint: disable=attribute-defined-outside-init
    with tf.device(_gpu_device_name(self.replica_id)):
      self.train_op = self.tf_optimizer.minimize(loss) # pylint: disable=attribute-defined-outside-init
    return self.train_op

  def set_train_route(self, route): # pylint: disable=unused-argument, no-self-use
    """Don't do anything. Added for consistency with other optimizers."""
    return
