from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import collections
#from simulation.device_placer import DevicePlacer
#PLACER = DevicePlacer()
from tensorflow.python.client import device_lib
from simulation.simulation_builder.device_placer import _gpu_device_name


def nn_mnist_architecture(graph):
	"""Creates architecture for NN mnist.

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
				X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
			with tf.name_scope('y'):
				y = tf.placeholder(tf.int64, shape=(None), name='y')
		#with tf.name_scope('NN'):

			
		hidden1 = nn_layer(	
			X, 
			n_hidden1, 
			name='hidden1', 
			activation=tf.nn.relu)
		
		hidden2 = nn_layer(
			hidden1,
			n_hidden2,
			name='hidden2',
			activation=tf.nn.relu)
		
		logits = nn_layer(
			hidden2,
			n_outputs,
			name='logits')
	return X, y, logits



def nn_layer(X, n_neurons, name, activation=None):
	"""Creates NN layer.

	Creates NN layer with W's initialized as truncated normal and
	b's as zeros.

	Args:
		`X` 		: Input tensor.
		`n_neurons`	: An integer. Number of neurons in the layer.
		`name`		: A string. Name of the layer.
		`activation`: Activation function. Optional.

	"""
	gpu_device_name = _get_default_gpu_name()

	with tf.name_scope(name):
		# dimension of each x in X
		n_inputs = int(X.get_shape()[1])

		stddev = 2.0 / np.sqrt(n_inputs)
		init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)

		with tf.device(_gpu_device_name(0)):
			W = tf.Variable(init, name='W')
			
			
			b = tf.Variable(tf.zeros([n_neurons]), name='b')
			
			
			Z = tf.matmul(X, W) + b
			

			if activation is not None:
				return activation(Z)
			else:
				return Z
def conv2d_layer(X, filter_shape, strides, name, padding='SAME', activation=None):

	with tf.device(_gpu_device_name(0)):
		with tf.name_scope(name):
			init = tf.truncated_normal(filter_shape, stddev=1e-4)
			kernel = tf.Variable(init, name='W')
			conv = tf.nn.conv2d(X, kernel, strides, padding=padding, name=name)
			biases = tf.Variable(tf.zeros([filter_shape[-1]]))
			Z = tf.nn.bias_add(conv, biases)
			if activation is not None:
				return activation(Z)
			else:
				return Z


def cnn_mnist_architecture(graph):
	height = 28
	width = 28
	channels = 1
	n_inputs = height * width * channels

	conv1_fmaps = 32
	conv1_ksize = 3
	conv1_stride = 1
	conv1_pad = "SAME"

	conv2_fmaps = 64
	conv2_ksize = 3
	conv2_stride = 2
	conv2_pad = "SAME"

	pool3_fmaps = conv2_fmaps

	n_fc1 = 64
	n_outputs = 10

	gpu_device_name = _get_default_gpu_name()

	with graph.as_default():
		with tf.name_scope('Input'):
			with tf.name_scope('X'):
				X = tf.placeholder(tf.float32, shape=[None, n_inputs], 
					name='X')
				X_reshaped = tf.reshape(X, 
					shape=[-1, height, width, channels])
			with tf.name_scope('y'):
				y = tf.placeholder(tf.int32, shape=[None], name='y')
		with tf.device(gpu_device_name):
			with tf.name_scope('conv1'):
				conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, 
					kernel_size=conv1_ksize, strides=conv1_stride, 
					padding=conv1_pad, activation=tf.nn.relu, name='conv1')

			with tf.name_scope('conv2'):
				conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, 
					kernel_size=conv2_ksize, strides=conv2_stride, 
					padding=conv2_pad, activation=tf.nn.relu, name='conv2')

			with tf.name_scope('pool3'):
				pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], 
					strides=[1, 2, 2, 1], padding='VALID')
				pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])

		with tf.name_scope('fully_connected'):
			fc = nn_layer(pool3_flat, n_fc1, activation=tf.nn.relu, name='fc')

		with tf.device(gpu_device_name):
			with tf.name_scope('dropout'):
				keep_prob = tf.placeholder(tf.float32, name='keep_prob')
				fc_dropout = tf.nn.dropout(fc, keep_prob)

			with tf.name_scope('logits'):
				logits = tf.layers.dense(fc_dropout, n_outputs, name='logits')

	return X, y, keep_prob, logits

def rnn_mnist_architecture(graph):
	n_inputs = 28
	n_hidden = 128
	n_classes = 10
	time_steps = 28

	from tensorflow.contrib import rnn

	def RNN(x, weights, biases, time_steps):
		x = tf.unstack(x, time_steps, 1)
		lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		return tf.matmul(outputs[-1], weights['out'] + biases['out'])



	with graph.as_default():
		with tf.name_scope('Inputs/X'):
			X = tf.placeholder('float', [None, time_steps, n_inputs])

		with tf.name_scope('Inputs/y'):
			y = tf.placeholder(tf.int32, [None, n_classes])

		weights = {
			'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
		}

		biases = {
			'out': tf.Variable(tf.random_normal([n_classes]))
		}
		logits = RNN(X, weights, biases, time_steps)
	return X, y, logits


def _get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	return [x.name for x in local_device_protos if x.device_type == 'GPU']

def _get_default_gpu_name():
	gpus = _get_available_gpus()
	
	return gpus[0] if len(gpus) > 0 else '/cpu:0'

