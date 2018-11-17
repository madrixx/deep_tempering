"""NN models for mnist"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module

from simulation.simulation_builder.device_placer import _gpu_device_name

def nn_layer(X, n_neurons, name, activation=None): # pylint: disable=invalid-name
  """Creates NN layer.

  Creates NN layer with W's initialized as truncated normal and
  b's as zeros.

  Args:
    X: Input tensor.
    n_neurons: An integer. Number of neurons in the layer.
    name: A string. Name of the layer.
    activation: Activation function. Optional.

  """
  with tf.name_scope(name):
    # dimension of each x in X
    n_inputs = int(X.get_shape()[1])

    stddev = 2.0 / np.sqrt(n_inputs)
    init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)

    with tf.device(_gpu_device_name(0)):
      W = tf.Variable(init, name='W') # pylint: disable=invalid-name

      b = tf.Variable(tf.zeros([n_neurons]), name='b') # pylint: disable=invalid-name

      Z = tf.matmul(X, W) + b # pylint: disable=invalid-name

      if activation is not None: # pylint: disable=no-else-return
        return activation(Z)
      else:
        return Z

############################ WORKING ARCHITECTURES ############################

def nn_mnist_model_075(graph):
  n_inputs = 28*28
  n_hidden1 = int(300*0.75)
  n_hidden2 = int(100*0.75)
  n_outputs = 10
  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2 = nn_layer(
        hidden1,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, logits

def nn_mnist_model_125(graph):
  n_inputs = 28*28
  n_hidden1 = int(300*1.25)
  n_hidden2 = int(100*1.25)
  n_outputs = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2 = nn_layer(
        hidden1,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, logits


def nn_mnist_model(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = 300
  n_hidden2 = 100
  n_outputs = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2 = nn_layer(
        hidden1,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    logits = nn_layer(
        hidden2,
        n_outputs,
        name='logits')

  return X, y, logits

def nn_mnist_model2(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = 1024
  n_hidden2 = 1024
  n_hidden3 = 2048
  n_outputs = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2 = nn_layer(
        hidden1,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden3 = nn_layer(
        hidden2,
        n_hidden3,
        name='hidden2',
        activation=tf.nn.relu)# pylint: disable=no-member

    logits = nn_layer(
        hidden3,
        n_outputs,
        name='logits')

  return X, y, logits

def nn_mnist_model2_05(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = int(1024*0.5)
  n_hidden2 = int(1024*0.5)
  n_hidden3 = int(2048*0.5)
  n_outputs = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2 = nn_layer(
        hidden1,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden3 = nn_layer(
        hidden2,
        n_hidden3,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    logits = nn_layer(
        hidden3,
        n_outputs,
        name='logits')

  return X, y, logits

def nn_mnist_model2_150(graph):
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = int(1024*1.5)
  n_hidden2 = int(1024*1.5)
  n_hidden3 = int(2048*1.5)
  n_outputs = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2 = nn_layer(
        hidden1,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden3 = nn_layer(
        hidden2,
        n_hidden3,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    logits = nn_layer(
        hidden3,
        n_outputs,
        name='logits')

  return X, y, logits


def nn_mnist_model_dropout(graph): # pylint: disable=too-many-locals
  """Creates model for NN mnist.

  Returns:
    logits
  """
  n_inputs = 28*28
  n_hidden1 = 1024
  n_hidden2 = 1024
  n_hidden3 = 2048
  n_outputs = 10

  with graph.as_default():
    with tf.name_scope('Inputs'):
      with tf.name_scope('X'):
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X') # pylint: disable=invalid-name
      with tf.name_scope('y'):
        y = tf.placeholder(tf.int64, shape=(None), name='y') # pylint: disable=invalid-name
    #with tf.name_scope('NN'):

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    hidden1 = nn_layer(
        X,
        n_hidden1,
        name='hidden1',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

    hidden2 = nn_layer(
        hidden1_dropout,
        n_hidden2,
        name='hidden2',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden2_dropout = tf.nn.dropout(hidden2, keep_prob)

    hidden3 = nn_layer(
        hidden2_dropout,
        n_hidden3,
        name='hidden3',
        activation=tf.nn.relu) # pylint: disable=no-member

    hidden3_dropout = tf.nn.dropout(hidden3, keep_prob)

    logits = nn_layer(
        hidden3_dropout,
        n_outputs,
        name='logits')
  return X, y, keep_prob, logits
