import os
import sys

import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import json

from simulation.simulation_builder.summary import Dir

__DEBUG__ = False


class MultiExperimentSummaryExtractor(object):

	def __init__(self, experiments):
		experiments = list(set(experiments))
		self.summary_extractors = {e:SummaryExtractor(e)
			for e in experiments}
		
	def plot(self, keys, match=None):
		param_names_list = []
		fig, ax = plt.subplots()
		keys = keys + ['mean']
		for k in self.summary_extractors:
			extractor = self.summary_extractors[k]
			for s in extractor.list_available_summaries():
				summ_name = s.split('/') if match == 'exact' else s
				if all(x in summ_name for x in keys):
					x, y = extractor.get_summary(s)
					with open(os.path.join(extractor._dir.log_dir, 'description.json')) as fo:
						js = json.load(fo)
						param_name = js['tuning_parameter_name']
						param_names_list.append(param_name)
						param_val = "{:10.2f}".format(float(js[param_name]))
						#"{:10.4f}".format(x) 
					ax.plot(x, y, label=k.split('_')[-1] + '_' + param_val +'/'+s)

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.4, 0.5))
		ax.title.set_text('Tuning parameter: ' + ', '.join(list(set(param_names_list))))

		return fig


class SummaryExtractor(object):

	def __init__(self, name):
		

		self._dir = Dir(name)
		#self.all_summs_dict = extract_summary(self._dir.log_dir)
		self.all_summs_dict = {}
		
		for i in range(100):
			try:
				#print(self._dir.log_dir + self._dir.delim + str(i))
				self.all_summs_dict.update(extract_summary(
					self._dir.log_dir + self._dir.delim + str(i)))
			except: 
				#print(i, 'simulations')
				self.n_experiments = i
				self._create_experiment_averages()
				break

	def get_summary(self, summ_name, split=True):
		"""Returns numpy arrays (x, y) of summaries.

		Args:
			summary_type: Name of the scalar summary
			

		Returns:
			(x, y) numpy array
		"""
		if split:
			return np.hsplit(self.all_summs_dict[summ_name], 2)
		else:
			return self.all_summs_dict[summ_name]

	def list_available_summaries(self):
		return sorted([k for k in self.all_summs_dict.keys()])
		

	def plot(self, keys=['valid'], match=None, add_swap_marks=False):
		#font_prop = FontProperties()
		#font_prop.set_size('small')
		
		n_col = 0

		fig, ax = plt.subplots()
		for s in self.list_available_summaries():
			summ_name = s.split('/') if match == 'exact' else s
			if all(x in summ_name for x in keys):
				x, y = self.get_summary(s)
				ax.plot(x, y, label=s)
				n_col += 1
				
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.6, 0.5))

		if add_swap_marks:
			with open(os.path.join(self._dir.log_dir, 'description.json')) as fo:
				js = json.load(fo)
			step = js['swap_attempt_step']
			s = self.list_available_summaries()[0]
			x, y = self.get_summary(s)
			len_ = int(x[-1][0])
			for i in range(0, len_, step):
				ax.axvline(x=i)
		return fig

	def get_description(self):
		with open(os.path.join(self._dir.log_dir, 'description.json')) as fo:
			js = json.load(fo)

		return js
		
	def _create_experiment_averages(self):
		all_keys = self.list_available_summaries()
		all_keys.sort(key=lambda x: x.split('/')[1] + x.split('/')[-1])

		completed_keys = []

		for k in all_keys:
			if k in completed_keys:
				continue
			name = '/'.join(k.split('/')[1:])
			arrays = [self.get_summary(str(i) + '/' + name, split=False)
				for i in range(self.n_experiments)]
			self.all_summs_dict['mean/' + name] = np.mean(np.array(arrays), axis=0)

		

"""

def extract_summary2(log_dir):
	res = {} 
	for f in os.listdir(log_dir):
		dirname = os.path.join(log_dir, f)
		if os.path.isdir(dirname):
		
			
			#print(dirname)
			res.update(extract_summary(dirname))

	return res
"""
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

		if os.path.isdir(fullpath):
		
			for _file in os.listdir(fullpath):
				
				filename = os.path.join(fullpath, _file)
				
				ea = event_accumulator.EventAccumulator(filename)
				ea.Reload()
				for k in ea.scalars.Keys():
					lc = np.stack(
						[np.asarray([scalar.step, scalar.value])
						for scalar in ea.Scalars(k)])
					key_name = sim_num + '/' + file + '/' +  k.split('/')[-1]
					key_name = '/'.join(key_name.split('/')[-3:])
					res[key_name] = lc
		
	return res
