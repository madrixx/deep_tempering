import os
import tensorflow as tf


from simulation.simulator_exceptions import InvalidDatasetTypeError

def sort_py_func(a):
    return a[a[:,0].argsort()]

class Summary(object):
	def __init__(self, graph, n_replicas, name, loss_dict, zero_one_loss_dict, noise_list, 
		swaps_dictnoise_plcholders, summary_type=None):

		self.graph = graph
		self.n_replicas = n_replicas
		self.name = name
		self.loss_dict = loss_dict
		self.zero_one_loss_dict = zero_one_loss_dict
		self.noise_list = noise_list
		self.noise_plcholders = noise_plcholders 
		self.swap_accept_ratio_plcholder = tf.placeholder(tf.float32, shape=[]) 
		self.accept_proba_plcholder = tf.placeholder(tf.float32, shape=[])
		self.pairs_swap_dict = {i:tf.placeholder(tf.float32, shape=[])
			for i in range(n_replicas - 1)}
		self.summary_type = summary_type
		self.log_dir = os.path.abspath(
			'/'.join(__file__.split('/')[:-2])) + '/summaries/' + name + '/'
		self.dir = Dir(self.log_dir, self.name)
		self.dir.clean_dirs(self.log_dir)

		self.writer_dict = {
			'train_ordered':{},
			'test_ordered':{},
			'valid_ordered':{},
			'train_replica':{},
			'test_replica':{},
			'valid_replica':{},
		}
		self.special_writer = None
		self.summ_replica = {}
		self.summ_ordered = {}
		self.train_summ_replica = {}
		self.train_summ_ordered = {}
		self.test_summ_replica = {}
		self.test_summ_ordered = {}
		self.valid_summ_replica = {}
		self.valid_summ_ordered = {}
		self.special_summ = None # returned together with validation

		if (summary_type is None
			or summary_type == 'replica_summary'):
			self.create_replica_summary()
		if (summary_type is None 
			or summary_type == 'ordered_summary'):
			self.create_ordered_summary()
		self.create_special_summary()

	def create_special_summary(self):
		with tf.name_scope('Summary'):
			tf.summary.scalar('accept_ratio', self.swap_accept_ratio_plcholder,
				collections=['special'])
			tf.summary.scalar('accept_proba', 
				tf.clip_by_value(self.accept_proba_plcholder, 0.0, 1.0),
				collections=['special'])

			self.special_writer = tf.summary.FileWriter(
				logdir=self.dir.get_special_dir(),
				graph=self.graph)

			self.special_summ = tf.summary.merge_all('special')

	def create_replica_summary(self,):
		
		with tf.name_scope('Summary'):
			for i in range(self.n_replicas):
				train_collect = ['train'+str(i)]
				test_collect = ['test'+str(i)]
				valid_collect = ['validation'+str(i)]

				tf.summary.scalar('cross_entropy', self.loss_dict[i], 
					collections=train_collect+test_collect+valid_collect)
				tf.summary.scalar('zero_one_loss', self.zero_one_loss_dict[i], 
					collections=train_collect+test_collect+valid_collect)
				
				tf.summary.scalar('noise', self.noise_plcholders[i],
					collections=train_collect)
				
				

				self.train_summ_replica[i] = tf.summary.merge_all(train_collect[0])
				self.test_summ_replica[i] = tf.summary.merge_all(test_collect[0])
				self.valid_summ_replica[i] = tf.summary.merge_all(valid_collect[0])

				self.writer_dict['train_replica'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_train_dir(i),
					graph=self.graph,
					filename_suffix=self.dir.get_filename_suffix())
				self.writer_dict['test_replica'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_test_dir(i),
					graph=self.graph,
					filename_suffix=self.dir.get_filename_suffix())
				self.writer_dict['valid_replica'][i] = tf.summary.FileWriter(
					logdir=self.dir.get_validation_dir(i),
					graph=self.graph,
					filename_suffix=self.dir.get_filename_suffix())

	def create_ordered_summary(self, ):
		loss = self.loss_dict
		acc = self.zero_one_loss_dict
		noise = self.noise_plcholders
		with tf.name_scope('Summary'):
			# sort using python function
			list_ = [(loss[i], acc[i], noise[i]) 
				for i in range(self.n_replicas)]

			sorted_ = tf.py_func(sort_py_func, [list_], tf.float32, 
				stateful=False)

			for i in range(self.n_replicas):
				train_collect = ['train'+str(i+self.n_replicas)]
				test_collect = ['test'+str(i+self.n_replicas)]
				valid_collect = ['validation'+str(i+self.n_replicas)]

				tf.summary.scalar('cross_entropy', sorted_[i][0], 
					collections=train_collect+test_collect+valid_collect)
				tf.summary.scalar('zero_one_loss', sorted_[i][1], 
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
		N = self.n_replicas
		
		if (self.summary_type is None 
			or self.summary_type=='replica_summary'):
			
			if dataset_type == 'train':
				summs = summs + [self.train_summ_replica[i] for i in range(N)]
			elif dataset_type == 'test':
				summs = summs + [self.test_summ_replica[i] for i in range(N)]
			elif dataset_type == 'validation':
				summs = summs + [self.valid_summ_replica[i] for i in range(N)]
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
		
		if dataset_type == 'validation':
			summs = summs + [self.special_summ]
		
		return summs

	def add_summary(self, evaluated_summ, step, dataset_type):
		if (self.summary_type is None
			or self.summary_type == 'replica_summary'):
			
			writer = self.get_summary_writer(dataset_type, 'replica_summary')
			for i in range(self.n_replicas):
				writer[i].add_summary(evaluated_summ[i], step)
		
		if (self.summary_type is None
			or self.summary_type == 'ordered_summary'):
			
			writer = self.get_summary_writer(dataset_type, 'ordered_summary')
			start_indx = (0 if self.summary_type is not None else self.n_replicas)
			for i in range(self.n_replicas):
				writer[i].add_summary(evaluated_summ[start_indx+i], step)

		if dataset_type == 'validation':
			self.special_writer.add_summary(evaluated_summ[-1], step)

	def get_summary_writer(self, dataset_type, summary_type):
		if summary_type == 'replica_summary':
			if dataset_type == 'train':
				return self.writer_dict['train_replica']
			elif dataset_type == 'test':
				return self.writer_dict['test_replica']
			elif dataset_type == 'validation':
				return self.writer_dict['valid_replica']
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
			raise ValueError("""`summary_type can be `replica_summary` or
				ordered_summary'""")

	def flush_summary_writer(self):
		for i in self.writer_dict:
			for k in self.writer_dict[i]:
				self.writer_dict[i][k].flush()
		self.special_writer.flush()

	def close_summary_writer(self):
		for i in self.writer_dict:
			for k in self.writer_dict[i]:
				self.writer_dict[i][k].close()
		self.special_writer.close()



class Dir(object):
	"""Helper class for generating directory names."""

	def __init__(self, summary_path, name):
		self.summary_path = summary_path
		self.name = name

	def get_special_dir(self):
		return self.summary_path + 'special_summary'

	def get_train_dir(self, replica_id):
		"""Returns the name of a train dir to store summaries.

		Args:
			replica_id:  An integer. The replica_id to which the result summaries
						belongs to.
			"""
		return self.summary_path + 'train_replica_' + str(replica_id)

	def get_test_dir(self, replica_id):
		"""Returns the name of a validation_dir to store summaries.

		Args:
			replica_id:  An integer. The replica_id to which the result summaries
						belongs to.
		"""
		return self.summary_path + 'test_replica_' + str(replica_id)

	def get_validation_dir(self, replica_id):
		"""Returns the name of a validation dir to store summaries.

		Args:
			replica_id:  An integer. The replica_id to which the result summaries
						belongs to.
		"""
		return self.summary_path + 'valid_replica' + str(replica_id)

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
		return self.summary_path + 'valid_ordered_' + str(k)

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


