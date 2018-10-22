import os
import sys

import tensorflow as tf

from simulation.simulator_exceptions import InvalidDatasetTypeError

def sort_py_func(a):
    return a[a[:,0].argsort()]

class Summary(object):
	def __init__(self, graph, n_replicas, name, loss_dict, zero_one_loss_dict, 
		stun_loss_dict, noise_list, noise_plcholders, simulation_num, 
		optimizer_dict, summary_type=None):

		self.graph = graph
		self.n_replicas = n_replicas
		self.name = name
		self.loss_dict = loss_dict
		self.zero_one_loss_dict = zero_one_loss_dict
		self.stun_loss_dict = stun_loss_dict
		self.noise_list = noise_list
		self.noise_plcholders = noise_plcholders
		self.optimizer_dict = optimizer_dict # diffusion summary 
		self.swap_accept_ratio_plcholder = tf.placeholder(tf.float32, shape=[]) 
		self.accept_proba_plcholder = tf.placeholder(tf.float32, shape=[])
		self.swap_replica_pair_plcholder = tf.placeholder(tf.int8, shape=[])
		self.swap_ordered_pair_plcholder = tf.placeholder(tf.int8, shape=[])

		self.replica_accept_ratio_plcholders = {i:tf.placeholder(tf.float32, shape=[]) 
			for i in range(self.n_replicas)}
		self.ordered_accept_ratio_plcholders = {i:tf.placeholder(tf.float32, shape=[]) 
			for i in range(self.n_replicas)}

		self.pairs_swap_dict = {i:tf.placeholder(tf.float32, shape=[])
			for i in range(n_replicas - 1)}
		self.summary_type = summary_type
		
		self.dir = Dir(self.name, simulation_num=simulation_num)
		self.dir.clean_dirs()
		
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
		self.special_summ = None # used only together with validation

		if (summary_type is None
			or summary_type == 'replica_summary'):
			self.create_replica_summary()
		if (summary_type is None 
			or summary_type == 'ordered_summary'):
			self.create_ordered_summary()
		self.create_diffusion_vals() # MUST BE CALLED BEFORE special_summary
		self.create_special_summary()
		
	def create_diffusion_vals(self):
		""" Diffusion vals for summary.
		
		Stores the initial values of a trainable 
		variables as a single vector to later calculation of a 
		displacement of trainable variables relative to the initial 
		values of those trainable parameters. 
		"""
		with tf.name_scope('Diffusion'):
			opt = self.optimizer_dict
			
			curr_vars = {i:sorted(opt[i].trainable_variables, key=lambda x: x.name) 
				for i in range(self.n_replicas)}

			init_vars = {i:[tf.Variable(v.initialized_value()) for v in curr_vars[i]]
				for i in range(self.n_replicas)}

			#print(curr_vars[0])

			curr_vars_reshaped = {i:[tf.reshape(v, [-1]) for v in curr_vars[i]]
				for i in range(self.n_replicas)}


			init_vars_reshaped = {i:[tf.reshape(v, [-1]) for v in init_vars[i]]
				for i in range(self.n_replicas)}
			#print(curr_vars_reshaped[0])
			
			curr_vars_concat = {i:tf.concat(curr_vars_reshaped[i], axis=0)
				for i in range(self.n_replicas)}

			init_vars_concat = {i:tf.concat(init_vars_reshaped[i], axis=0)
				for i in range(self.n_replicas)}

			#print(curr_vars_concat[0])
			#print(init_vars_concat)

			self.diffusion_tensors = {i:tf.norm(curr_vars_concat[i] - init_vars_concat[i])
				for i in range(self.n_replicas)}


			#self.curr_train_vars = train_vars_concat
			#self.init_train_vars = init_train_vars
			

	def create_special_summary(self):
		"""Special summary is everything except cross_validation, 
			zero_one_loss and noise."""
		with tf.name_scope('special_summary'):
			tf.summary.scalar('accept_ratio', self.swap_accept_ratio_plcholder,
				collections=['special'])
			tf.summary.scalar('accept_proba', 
				tf.clip_by_value(self.accept_proba_plcholder, 0.0, 1.0),
				collections=['special'])
			tf.summary.scalar('swapped_replica_pair', 
				self.swap_replica_pair_plcholder,
				collections=['special'])

			for i in range(self.n_replicas):
				tf.summary.scalar('accept_ratio_replica_' + str(i),
					self.replica_accept_ratio_plcholders[i],
					collections=['special'])
				tf.summary.scalar('accept_ratio_ordered_' + str(i),
					self.ordered_accept_ratio_plcholders[i],
					collections=['special'])
				tf.summary.scalar('diffusion_' + str(i),
					self.diffusion_tensors[i], 
					collections=['special'])



			self.special_writer = tf.summary.FileWriter(
				logdir=self.dir.get_special_dir(),
				graph=self.graph)

			self.special_summ = tf.summary.merge_all('special')

	def create_replica_summary(self,):
		
		
		for i in range(self.n_replicas):
			with tf.name_scope('replica_summary_' + str(i)):
				train_collect = ['train'+str(i)]
				test_collect = ['test'+str(i)]
				valid_collect = ['validation'+str(i)]

				tf.summary.scalar('cross_entropy', self.loss_dict[i], 
					collections=train_collect+test_collect+valid_collect)
				
				tf.summary.scalar('zero_one_loss', self.zero_one_loss_dict[i], 
					collections=train_collect+test_collect+valid_collect)

				tf.summary.scalar('stun', self.stun_loss_dict[i],
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
		stun = self.stun_loss_dict
		noise = self.noise_plcholders
		
		# sort using python function
		list_ = [(loss[i], acc[i], stun[i], noise[i]) 
			for i in range(self.n_replicas)]

		sorted_ = tf.py_func(sort_py_func, [list_], tf.float32, 
			stateful=False)

		for i in range(self.n_replicas):
			train_collect = ['train'+str(i+self.n_replicas)]
			test_collect = ['test'+str(i+self.n_replicas)]
			valid_collect = ['validation'+str(i+self.n_replicas)]
			with tf.name_scope('ordered_summary_' + str(i)):

				tf.summary.scalar('cross_entropy', sorted_[i][0], 
					collections=train_collect+test_collect+valid_collect)

				tf.summary.scalar('zero_one_loss', sorted_[i][1], 
					collections=train_collect+test_collect+valid_collect)

				tf.summary.scalar('stun', sorted_[i][2],
					collections=train_collect+test_collect+valid_collect)

				tf.summary.scalar('noise', sorted_[i][3],
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
		"""Returns summary ops.

		For train, test and validation tensor evaluation different
		summary variables are stored. Hence, `dataset_type` must
		be specified.

		Args:
			dataset_type: One of `train`, `test`, `validation`.

		Raises:
			InvalidDatasetTypeError if `dataset_type` is incorrect.

		Returns:
			Summary ops 
		"""
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
		"""Adds evaluated summary to summary writer.

		Args:
			evaluated_summ: A list of evaluated summary (by sess.run()).
			step: An int for second argument of tf.Summary.add_summary()
			dataset_type: One of `train`, `test`, `validation`.

		"""
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
		"""Returns summary writer.

		Args:
			dataset_type: One of `train`, `test`, `validation`.
			summary_type: One of `replica_summary`, `ordered_summary`.
				`replica_summary` is the summary that stores metrics 
				for each one of the replicas. `ordered_summary` is the
				summary that stores metrics for each replica but in 
				sorted form (best loss is 0, second best is 1 etc.)
		
		Raises:
			InvalidDatasetTypeError if `dataset_type` is incorrect. 
			ValueError if `summary_type` is incorrect.

		Returns:
			Summary writer

		"""

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

	def __init__(self, name, simulation_num=''):
		
		self.name = name
		self.delim = "\\" if "win" in sys.platform else "/"
		log_dir = os.path.abspath(
			self.delim.join(__file__.split(self.delim)[:-2]))
		self.log_dir = os.path.join(log_dir, 'summaries', name) 
		if (simulation_num is not None and
			simulation_num != ''):
			self.log_dir = os.path.join(self.log_dir, str(simulation_num))
		
	
	def get_special_dir(self):
		#return self.log_dir + 'special_summary'
		return os.path.join(self.log_dir, 'special_summary')

	def get_train_dir(self, replica_id):
		"""Returns the name of a train dir to store summaries.

		Args:
			replica_id:  An integer. The replica_id to which the result summaries
						belongs to.
			"""
		#return self.log_dir + 'train_replica_' + str(replica_id)
		return os.path.join(self.log_dir, 'train_replica_' + str(replica_id))
	def get_test_dir(self, replica_id):
		"""Returns the name of a validation_dir to store summaries.

		Args:
			replica_id:  An integer. The replica_id to which the result summaries
						belongs to.
		"""
		#return self.log_dir + 'test_replica_' + str(replica_id)
		return os.path.join(self.log_dir, 'test_replica_' + str(replica_id))

	def get_validation_dir(self, replica_id):
		"""Returns the name of a validation dir to store summaries.

		Args:
			replica_id:  An integer. The replica_id to which the result summaries
						belongs to.
		"""
		#return self.log_dir + 'valid_replica_' + str(replica_id)
		return os.path.join(self.log_dir, 'valid_replica_' + str(replica_id))

	def get_ordered_train_dir(self, k):
		"""Returns the name of the ordered train dir to store summaries.

		Args:
			k: An integer. Corresponds to the k'th lowest loss.
		"""
		#return self.log_dir + 'train_ordered_' + str(k)
		return os.path.join(self.log_dir, 'train_ordered_' + str(k))

	def get_ordered_test_dir(self, k):
		"""Returns the name of the ordered test dir to store summaries.

		Args:
			k: An integer. Corresponds to the k'th lowest loss.
		"""
		#return self.log_dir + 'test_ordered_' + str(k)
		return os.path.join(self.log_dir, 'test_ordered_' + str(k))

	def get_ordered_validation_dir(self, k):
		"""Returns the name of the ordered validation dir to store summaries.

		Args:
			k: An integer. Corresponds to the k'th lowest loss.
		"""
		#return self.log_dir + 'valid_ordered_' + str(k)
		return os.path.join(self.log_dir, 'valid_ordered_' + str(k))

	def get_filename_suffix(self): return self.name

	def clean_dirs(self, dir_=None):
		"""Recursively removes all train, test and validation summary files \
				and folders from previos training life cycles."""
		if dir_ is None:
			dir_ = self.log_dir
		try:
			for file in os.listdir(dir_):
				if os.path.isfile(os.path.join(dir_, file)):
					os.remove(os.path.join(dir_, file))
				else:
					self.clean_dirs(os.path.join(dir_, file))
			
			if dir_ == self.log_dir:
				for file in os.listdir(dir_):
					os.rmdir(os.path.join(dir_, file))
		except OSError:
			# if first simulation, nothing to delete
			return


