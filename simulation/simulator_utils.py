import os
import sys

import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt

from simulation.simulation_builder.summary import Dir

__DEBUG__ = False

class SummaryExtractor(object):

	def __init__(self, name, disable_log=True):
		if disable_log:
			tf.logging.set_verbosity(tf.logging.ERROR)

		self._dir = Dir(name)
		#self.all_summs_dict = extract_summary(self._dir.log_dir)
		self.all_summs_dict = {}
		
		for i in range(100):
			try:
				print(self._dir.log_dir + self._dir.delim + str(i))
				self.all_summs_dict.update(extract_summary(
					self._dir.log_dir + self._dir.delim + str(i)))
			except: 
				print(3, 'simulations')
				break

	def get_summary(self, summ_name):
		"""Returns numpy arrays (x, y) of summaries.

		Args:
			summary_type: Name of the scalar summary
			

		Returns:
			(x, y) numpy array
		"""
		return np.hsplit(self.all_summs_dict[summ_name], 2)

	def list_available_summaries(self):
		return sorted([k for k in self.all_summs_dict.keys()])
		#return sorted(self.all_summs_dict.keys())

	def create_subplot(self, keys=['valid']):
		#font_prop = FontProperties()
		#font_prop.set_size('small')
		n_col = 0
		fig, ax = plt.subplots()
		for s in self.list_available_summaries():
			if all(k in s for k in keys):
				x, y = self.get_summary(s)
				ax.plot(x, y, label=s)
				n_col += 1
			
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.5, 0.5))
		return fig

"""
def extract_summary(log_dir, tag):
	'''Extracts summaries from simulation `name`

	Args:
		log_dir: directory
		tag: summary name (e.g. cross_entropy, zero_one_loss ...)

	Returns:
		A dict where keys are names of the tf.summary.FileWriter folders
		and vals are lists with summary values. 
	'''
	
	#current_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
	#log_dir = os.path.join(log_dir, 'summaries', name)
	res = {}
	for file in os.listdir(log_dir):
		fullpath = os.path.join(log_dir, file)
		if os.path.isdir(fullpath):
			res[file] = []
			for _file in os.listdir(fullpath):
				#print(_file)
				for e in tf.train.summary_iterator(os.path.join(fullpath, _file)):
					for v in e.summary.value:
						if tag in v.tag:
							res[file].append(v.simple_value)
							#return v
							#print(v.step)
							#sys.exit()

	return res
"""

def extract_summary2(log_dir):
	res = {} 
	for f in os.listdir(log_dir):
		dirname = os.path.join(log_dir, f)
		if os.path.isdir(dirname):
		
			
			#print(dirname)
			res.update(extract_summary(dirname))

	return res

def extract_summary(log_dir, delim="\\"):
	"""
	Extracts summaries from simulation `name`

	Args:
		log_dir: directory
		tag: summary name (e.g. cross_entropy, zero_one_loss ...)

	Returns:
		A dict where keys are names of the summary scalars and
		vals are numpy arrays of tuples (step, value)
	""" 
	sim_num = log_dir.split(delim)[-1]
	res = {}
	for file in os.listdir(log_dir):
		fullpath = os.path.join(log_dir, file)
		#print(file)

		if os.path.isdir(fullpath):
		
			for _file in os.listdir(fullpath):
				
				filename = os.path.join(fullpath, _file)
				
				ea = event_accumulator.EventAccumulator(filename)
				ea.Reload()
				for k in ea.scalars.Keys():
					lc = np.stack(
						[np.asarray([scalar.step, scalar.value])
						for scalar in ea.Scalars(k)])

					res[sim_num + '/' + file + '/' +  k.split('/')[-1]] = lc
		
	return res
