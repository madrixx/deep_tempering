import sys
import os

import tensorflow as tf
import json

from simulation.simulation_builder.graph_builder import GraphBuilder

class Simulator(object):

	def __init__(self, architecture, learning_rate, noise_list, noise_type, 
		batch_size, n_epochs, name, summary_type=None, test_step=200, 
		description=None):
		
		self.graph = GraphBuilder(architecture, learning_rate, noise_list, 
			name, noise_type, summary_type)

		self.noise_type = noise_type
		self.noise_list = noise_list
		self.summary_type = summary_type
		self.learning_rate = learning_rate
		self.name=name

		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.test_step = test_step
		tf.logging.set_verbosity(tf.logging.ERROR)
		if description:
			self._log_params(description)
	def train(self, *args, **kwargs):

		def _prepare_data():
			with g.get_tf_graph().as_default():
				train_data = kwargs.get('train_data', None)
				train_labels = kwargs.get('train_labels', None)
				test_data = kwargs.get('test_data', None)
				test_labels = kwargs.get('test_labels', None)
				valid_data = kwargs.get('validation_data', None)
				valid_labels = kwargs.get('validation_labels', None)

				test_feed_dict = g.create_feed_dict(test_data, test_labels, 
					dataset_type='test')
				valid_feed_dict = g.create_feed_dict(valid_data, valid_labels,
					dataset_type='validation')

				data = tf.data.Dataset.from_tensor_slices({
					'X':train_data,
					'y':train_labels
					}).batch(self.batch_size)
				train_iter = data.make_initializable_iterator()

			return test_feed_dict, valid_feed_dict, train_iter

		try:
			g = self.graph
			test_feed_dict, valid_feed_dict, iterator = _prepare_data()

		except:
			raise
		

		with g.get_tf_graph().as_default():

			step = 0
			#n_iters = len(train_data) // self.batch_size
			
			with tf.Session() as sess:
				sess.run(iterator.initializer)
				sess.run(g.variable_initializer)
				next_batch = iterator.get_next()

				# validation first time
				evaluated = sess.run(g.get_train_ops('validation'),
					feed_dict=valid_feed_dict)
				g.add_summary(evaluated, step, dataset_type='validation')
				g.update_noise_vals(evaluated)

				for epoch in range(self.n_epochs):
					
					while True:
						try:
							step += 1

							# train
							batch = sess.run(next_batch)
							feed_dict = g.create_feed_dict(batch['X'], batch['y'])
							evaluated = sess.run(g.get_train_ops(), 
								feed_dict=feed_dict)
							g.add_summary(evaluated, step=step)

							# test
							if step % self.test_step == 0:
								evaluated = sess.run(g.get_train_ops('test'),
									feed_dict=test_feed_dict)
								g.add_summary(evaluated, step, dataset_type='test')
								loss = g.extract_evaluated_tensors(evaluated, 'loss')
								buff = 'epoch:' + str(epoch) + ', step:' + str(step) + ', '
								buff = buff + ','.join([str(l) for l in loss])
								self.stdout_write(buff)
						except tf.errors.OutOfRangeError:
							sess.run(iterator.initializer)
							break

					# validation
					evaluated = sess.run(g.get_train_ops('validation'),
						feed_dict=valid_feed_dict)
					g.add_summary(evaluated, step, dataset_type='validation')
					g.update_noise_vals(evaluated)
					#print()
					#print(evaluated[:g._n_workers])
					#print('*******************************')
					#break
				g._summary.close_summary_writer()

	def _train_(self, *args, **kwargs):

		def _prepare_data():
			with g.get_tf_graph().as_default():
				train_data = kwargs.get('train_data', None)
				train_labels = kwargs.get('train_labels', None)
				test_data = kwargs.get('test_data', None)
				test_labels = kwargs.get('test_labels', None)
				valid_data = kwargs.get('validation_data', None)
				valid_labels = kwargs.get('validation_labels', None)

				test_feed_dict = g.create_feed_dict(test_data, test_labels, 
					dataset_type='test')
				valid_feed_dict = g.create_feed_dict(valid_data, valid_labels,
					dataset_type='validation')

				data = tf.data.Dataset.from_tensor_slices({
					'X':train_data,
					'y':train_labels
					}).batch(self.batch_size)
				train_iter = data.make_initializable_iterator()

			return test_feed_dict, valid_feed_dict, train_iter

		try:
			g = self.graph
			test_feed_dict, valid_feed_dict, iterator = _prepare_data()

		except:
			raise
		
		with g.get_tf_graph().as_default():

			step = 0
			#n_iters = len(train_data) // self.batch_size
			
			with tf.Session() as sess:
				sess.run(iterator.initializer)
				sess.run(g.variable_initializer)
				next_batch = iterator.get_next()

				# validation first time
				evaluated = sess.run(g.get_train_ops('validation'),
					feed_dict=valid_feed_dict)
				g.add_summary(evaluated, 1, dataset_type='validation')
				g.update_noise_vals(evaluated)

				for epoch in range(self.n_epochs):
					
					while True:
						try:
							step += 1

							# train
							batch = sess.run(next_batch)
							feed_dict = g.create_feed_dict(batch['X'], batch['y'])
							evaluated = sess.run(g.get_train_ops(), 
								feed_dict=feed_dict)
							#sys.exit()
							g.add_summary(evaluated, step=step)
							
							# test
							if step % self.test_step == 0:
								evaluated = sess.run(g.get_train_ops('test'),
									feed_dict=test_feed_dict)
								g.add_summary(evaluated, step, dataset_type='test')
								loss = g.extract_evaluated_tensors(evaluated, 'loss')
								buff = 'epoch:' + str(epoch) + ', step:' + str(step) + ', '
								buff = buff + ','.join([str(l) for l in loss])
								self.stdout_write(buff)
						except tf.errors.OutOfRangeError:
							sess.run(iterator.initializer)
							break

					# validation
					evaluated = sess.run(g.get_train_ops('validation'),
						feed_dict=valid_feed_dict)
					g.add_summary(evaluated, step, dataset_type='validation')
					g.update_noise_vals(evaluated)
					#print()
					#print(evaluated[:g._n_workers])
					#print('*******************************')
					#break
				g._summary.close_summary_writer()
	
	
	
	def _log_params(self, desciption):
		dirpath = self.graph._summary.dir.summary_path
		filepath = os.path.join(dirpath, 'desciption.json')
		_log = {
			'name':self.name,
			'noise_type': self.noise_type,
			'noise_list': self.noise_list,
			'learning_rate':self.learning_rate,
			'n_epochs':self.n_epochs,
			'description':desciption

		}
		with open(filepath, 'w') as fo:
			json.dump(_log, fo, indent=4)

	def stdout_write(self, buff):
		sys.stdout.write('\r' + buff)
		sys.stdout.flush()

	def chunks(self, l):
		"""Yield successive batch-sized chunks from l."""
		for i in range(0, len(l), self.batch_size):
			yield l[i:i + n]
