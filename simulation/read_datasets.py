from __future__ import print_function, division, absolute_import
import os
from six.moves import urllib
import tarfile
import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random

mnist_dataset_path = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/data/'
mnist_dataset_path = os.path.join(mnist_dataset_path, 'mnist')

class Cifar10(object):

	def __init__(self, batch_size=50):

		self.DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
		self.data_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/data/'
		self.cifar10_dir = os.path.join(self.data_dir, 'cifar10')
		self.batch_size = batch_size
		self._download_and_extract()
		self._prepare_dataset()
		'''
		self.X = X
		self.y = y
		self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
		self.train_dataset = self.train_dataset.batch(self.batch_size)

		self.iterator = self.train_dataset.make_initializable_iterator()
		'''
		'''
		self.train_dataset = tf.data.Dataset.from_tensor_slices({
			'X':self.train_data,
			'y':self.train_labels}).batch(self.batch_size)

		self.test_dataset = tf.data.Dataset.from_tensor_slices({
			'X':self.test_data,
			'y':self.test_labels})
		self.valid_dataset = tf.data.Dataset.from_tensor_slices({
			'X':self.valid_data,
			'y':self.valid_labels})

		self.iterator = self.train_dataset.make_initializable_iterator()
		'''
		



	def _prepare_dataset(self):
		"""
		# PYTHON 2.7.x
		import cPickle as pickle
		def unpickle(file):
			
			with open(file , 'rb') as fo:
				res = pickle.load(fo)
			return res
		"""
		# PYTHON 3.x.x
		import pickle
		def unpickle(file):
			with open(file, 'rb') as fo:
				u = pickle._Unpickler(fo)
				u.encoding = 'latin1'
				res = u.load()
			return res

		def extract_data(datatype):
			"""Returns tuple numpy arrays of data, labels

			Args:
				`datatype`: A string, 'test', 'train', or 'validation'
			"""
			batches = []
			if datatype == 'train':
				str2search = 'batch_'
			elif datatype == 'test':
				str2search = 'test'
			elif datatype == 'validation':
				'''Figure out from where to take validation data'''
				str2search = 'test'
 
			for file in os.listdir(self.cifar10_dir):
				file_path = os.path.join(self.cifar10_dir, file)
				if os.path.isfile(file_path) and str2search in file:
					batches.append(unpickle(file_path))
			data = np.concatenate(tuple(a['data'] for a in batches))
			labels = np.concatenate(tuple(a['labels'] for a in batches))
			return data, labels
		
		
		self.train_data, self.train_labels  = extract_data('train')
		self.test_data, self.test_labels = extract_data('test')
		self.valid_data, self.valid_labels = extract_data('validation')
		
	def _download_and_extract(self):
		"""Download from https://www.cs.toronto.edu/~kriz/cifar.html if the
		the file is not located in the path"""
		if not os.path.exists(self.data_dir):
			os.makedirs(self.data_dir)
		dest_directory = self.cifar10_dir 
		
		if not os.path.exists(dest_directory):
			os.makedirs(dest_directory)

		filename = self.DATA_URL.split('/')[-1]
		filepath = os.path.join(dest_directory, filename)
		

		if not os.path.exists(filepath):
			def _progress(count, block_size, total_size):
				sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
						float(count * block_size) / float(total_size) * 100.0))
				sys.stdout.flush()
			filepath, _ = urllib.request.urlretrieve(self.DATA_URL, filepath, _progress)
			print()
			statinfo = os.stat(filepath)
			print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
		extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
		if not os.path.exists(extracted_dir_path):
			tarfile.open(filepath, 'r:gz').extractall(dest_directory)
		self.cifar10_dir = extracted_dir_path
	
	
def get_cifar10_data():
	c = Cifar10()
	X_test, y_test = c.test_data, c.test_labels

	X_train, X_valid, y_train, y_valid = train_test_split(c.train_data, 
		c.train_labels, test_size=0.1, random_state=random.randint(1, 42))

	return X_train, y_train, X_test, y_test, X_valid, y_valid