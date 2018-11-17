"""Downloads and returns cifar10 dataset."""
from __future__ import print_function, division, absolute_import
import os
import tarfile
import random
import sys
from six.moves import urllib

import numpy as np
from sklearn.model_selection import train_test_split


MNIST_DATASET_PATH = ('/'.join(
    os.path.abspath(__file__).split('/')[:-1]) + '/data/')
MNIST_DATASET_PATH = os.path.join(MNIST_DATASET_PATH, 'mnist')

class Cifar10: # pylint: disable=too-many-instance-attributes, too-few-public-methods, missing-docstring

  def __init__(self, batch_size=50):

    self.DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' # pylint: disable=invalid-name, line-too-long
    self.data_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1]) + '/data/' # pylint: disable=line-too-long
    self.cifar10_dir = os.path.join(self.data_dir, 'cifar10')
    self.batch_size = batch_size
    self._download_and_extract()
    self._prepare_dataset()

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
      with open(file, 'rb') as file_:
        unpick = pickle._Unpickler(file_) # pylint: disable=protected-access
        unpick.encoding = 'latin1'
        res = unpick.load()
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
        str2search = 'test'

      for file in os.listdir(self.cifar10_dir):
        file_path = os.path.join(self.cifar10_dir, file)
        if os.path.isfile(file_path) and str2search in file:
          batches.append(unpickle(file_path))
      data = np.concatenate(tuple(a['data'] for a in batches))
      labels = np.concatenate(tuple(a['labels'] for a in batches))
      return data, labels

    self.train_data, self.train_labels = extract_data('train')
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
        sys.stdout.write(
            '\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

      filepath, _ = urllib.request.urlretrieve(
          self.DATA_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir_path):
      tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    self.cifar10_dir = extracted_dir_path

def get_cifar10_data(validation_size=0.1, random_state=None):
  """Returns cifar10 data. If not on disk, downloads."""
  cif = Cifar10()
  X_test, y_test = cif.test_data, cif.test_labels # pylint: disable=invalid-name

  X_train, X_valid, y_train, y_valid = train_test_split( # pylint: disable=invalid-name
      cif.train_data,
      cif.train_labels, test_size=validation_size,
      random_state=(random_state
                    if random_state is not None else random.randint(1, 42)))

  return X_train, y_train, X_test, y_test, X_valid, y_valid
