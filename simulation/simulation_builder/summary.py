import os
import tensorflow as tf


def InvalidDatasetTypeError(Exception):
	pass
	def __str__(self, ):
		return """The dataset_type must be one of: 'train', 'test' or 
			'validation'"""

def sort_py_func(a):
    return a[a[:,0].argsort()]

class Summary(object):
	def __init__(self, graph, n_workers, name, loss_dict, acc_dict, noise_list, 
		noise_plcholders, summary_type=None):

		self.graph = graph
		self.n_workers = n_workers
		self.name = name
		self.loss_dict = loss_dict
		self.acc_dict = acc_dict
		self.noise_list = noise_list
		self.noise_plcholders = noise_plcholders 
		self.summary_type = summary_type
		self.log_dir = os.path.abspath(
			'/'.join(__file__.split('/')[:-2])) + '/summaries/' + name + '/'
		self.dir = Dir(self.log_dir, self.name)
		self.dir.clean_dirs(self.log_dir)

		self.writer_dict = {
			'train_ordered':{},
			'test_ordered':{},
			'valid_ordered':{},
			'train_worker':{},
			'test_worker':{},
			'valid_worker':{}
		}

		self.summ_worker = {}
		self.summ_ordered = {}
		self.train_summ_worker = {}
		self.train_summ_ordered = {}
		self.test_summ_worker = {}
		self.test_summ_ordered = {}
		self.valid_summ_worker = {}
		self.valid_summ_ordered = {}

		if (summary_type is None
			or summary_type == 'worker_summary'):
			self.create_worker_summary()
		if (summary_type is None 
			or summary_type == 'ordered_summary'):
			self.create_ordered_summary()

	def create_worker_summary(self,):
		
		with tf.name_scope('Summary'):
			for i in range(self.n_workers):
				train_collect = ['train'+str(i)]
				test_collect = ['test'+str(i)]
				valid_collect = ['validation'+str(i)]

				tf.summary.scalar('loss', self.loss_dict[i], 
					collections=train_collect+test_collect+valid_collect)
				tf.summary.scalar('accuracy', self.acc_dict[i], 
					collections=train_collect+test_collect+valid_collect)
				
				tf.summary.scalar('noise', self.noise_plcholders[i],
					collections=train_collect)

				self.train_summ_worker[i] = tf.summary.merge_all(train_collect[0])
				self.test_summ_worker[i] = tf.summary.merge_all(test_collect[0])
				self.valid_summ_worker[i] = tf.summary.merge_all(valid_collect[0])

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

	def create_ordered_summary(self, ):
		loss = self.loss_dict
		acc = self.acc_dict
		noise = self.noise_plcholders
		with tf.name_scope('Summary'):
			# sort using python function
			list_ = [(loss[i], acc[i], noise[i]) 
				for i in range(self.n_workers)]

			sorted_ = tf.py_func(sort_py_func, [list_], tf.float32, 
				stateful=False)

			for i in range(self.n_workers):
				train_collect = ['train'+str(i+self.n_workers)]
				test_collect = ['test'+str(i+self.n_workers)]
				valid_collect = ['validation'+str(i+self.n_workers)]

				tf.summary.scalar('loss', sorted_[i][0], 
					collections=train_collect+test_collect+valid_collect)
				tf.summary.scalar('accuracy', sorted_[i][1], 
					collections=train_collect+test_collect+valid_collect)
				
				tf.summary.scalar('noise', sorted_[i][2],
					collections=train_collect)

				self.train_summ_ordered[i] = tf.summary.merge_all(train_collect[0])
				self.test_summ_ordered[i] = tf.summary.merge_all(test_collect[0])
				self.valid_summ_ordered[i] = tf.summary.merge_all(valid_collect[0])

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

	def get_summary_ops(self, dataset_type):
		summs = []
		N = self.n_workers
		if (self.summary_type is None 
			or self.summary_type=='worker_summary'):
			
			if dataset_type == 'train':
				summs = summs + [self.train_summ_worker[i] for i in range(N)]
			elif dataset_type == 'test':
				summs = summs + [self.test_summ_worker[i] for i in range(N)]
			elif dataset_type == 'validation':
				summs = summs + [self.valid_summ_worker[i] for i in range(N)]
			else:
				raise InvalidDatasetTypeError()

		if (self.summary_type is None
			or self.summary_type=='ordered_summary'):

			if dataset_type == 'train':
				summs = summs + [self.train_summ_ordered[i] for i in range(N)]
			elif dataset_type == 'test':
				summs = summs + [self.test_summ_ordered[i] for i in range(N)]
			elif dataset_type == 'validation':
				summs = summs + [self.valid_summ_ordered[i] for i in range(N)]
			else:
				raise InvalidDatasetTypeError()

		return summs

	def add_summary(self, evaluated_summ, step, dataset_type):
		if (self.summary_type is None
			or self.summary_type == 'worker_summary'):
			
			writer = self.get_summary_writer(dataset_type, 'worker_summary')
			for i in range(self.n_workers):
				writer[i].add_summary(evaluated_summ[i], step)
		
		if (self.summary_type is None
			or self.summary_type == 'ordered_summary'):
			
			writer = self.get_summary_writer(dataset_type, 'ordered_summary')
			start_indx = (0 if self.summary_type is not None else self.n_workers)
			for i in range(self.n_workers):
				writer[i].add_summary(evaluated_summ[start_indx+i], step)

	def get_summary_writer(self, dataset_type, summary_type):
		if summary_type == 'worker_summary':
			if dataset_type == 'train':
				return self.writer_dict['train_worker']
			elif dataset_type == 'test':
				return self.writer_dict['test_worker']
			elif dataset_type == 'validation':
				return self.writer_dict['valid_worker']
			else:
				raise InvalidDatasetTypeError()
		elif summary_type == 'ordered_summary':
			if dataset_type == 'train':
				return self.writer_dict['train_ordered']
			elif dataset_type == 'test':
				return self.writer_dict['test_ordered']
			elif dataset_type == 'validation':
				return self.writer_dict['valid_ordered']
			else:
				raise InvalidDatasetTypeError()
		else:
			raise ValueError("""`summary_type can be `worker_summary` or
				ordered_summary'""")

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


