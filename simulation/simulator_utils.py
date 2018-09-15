import os
import sys

import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import pandas as pd
import re

from simulation.simulation_builder.summary import Dir

__DEBUG__ = False


class MultiExperimentSummaryExtractor(object):

	def __init__(self, experiments):
		experiments = list(set(experiments))
		self.summary_extractors = {e:SummaryExtractor(e)
			for e in experiments}

	def get_summary_extractor(self, name, simulation_num):
		try:
			return self.summary_extractors[name + '_' + str(simulation_num)]
		except KeyError:
			sim_numbers = self.summary_extractors.keys().sort(key=lambda x: x.split('_')[-1])
			print('The max number of simulation is', sim_numbers[-1], ', but given', simulation_num)

	def get_summary_extractor_by_experiment_num(self, experiment_num):
		name = [n for n in self.summary_extractors if str(experiment_num) == n.split('_')[-1]][0]
		return self.summary_extractors[name]

	
	def plot(self, keys, match=None, param_min=None, param_max=None, mark_lines=None):
		def sort_foo(x):
			return int(x.split('_')[-1])
		param_names_list = []
		fig, ax = plt.subplots()
		keys = keys + ['mean']
		completed_labels = []
		for k in sorted(self.summary_extractors, key=sort_foo):
			extractor = self.summary_extractors[k]
			for s in extractor.list_available_summaries():
				summ_name = s.split('/') if match == 'exact' else s
				if all(x in summ_name for x in keys):
					x, y = extractor.get_summary(s)
					
					js = extractor.get_description()
					param_name = js['tuning_parameter_name']
					param_names_list.append(param_name)
					param_val = "{:10.2f}".format(float(js[param_name]))
					if param_min and param_min > float(param_val):
						continue
					if param_max and param_max < float(param_val):
						continue
					label = k.split('_')[-1] + '_' + param_val +'/'+s
					
					if label in completed_labels:
						continue
					completed_labels.append(label)
					if mark_lines and float(param_val) in mark_lines:

						ax.plot(x, y, label=label, linewidth=3.5)
					else:	
						ax.plot(x, y, label=label)

		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.4, 0.5))
		ax.title.set_text('Tuning parameter: ' + ', '.join(list(set(param_names_list))))

		return fig

	def create_df(self, variable_names, thresh=0.3, df_name=None):
		def sort_foo(x):
			se = self.summary_extractors[x]
			return se.get_description()['temp_factor']

		df_name = 'None' if df_name is None else df_name
		experiment_num = 0
		index = 0
		df = None
		summ_extractors_names = list(self.summary_extractors.keys())

		summ_extractors_names.sort(key=sort_foo)
		for experiment_name in summ_extractors_names:
			se = self.summary_extractors[experiment_name]
			summs_names = [n for n in se.list_available_summaries()
				if all(x in re.split(r"/|_", n) for x in variable_names)]

			for summ_name in summs_names:
				if 'mean' in summ_name:
					continue
				sim_num = int(summ_name.split('/')[0])
				id_ = int(summ_name.split('_')[-1])
				if df is None:
					cols = ['experiment', 'simulation', 'id', 'temp_factor']
					values = se.get_summary(summ_name)
					start_indx = int(values[0].shape[0]*thresh)
					v_cols = [v[0] for v in values[0]]
					cols = cols + v_cols[start_indx:]
					df = pd.DataFrame(columns=cols)
				js = se.get_description()
				temp_factor = js['temp_factor']
				values = se.get_summary(summ_name)
				values = [v[0] for v in values[1]]
				values = [experiment_num, sim_num, id_, temp_factor] + values[start_indx:]
				df.loc[index] = values
				index += 1
			experiment_num += 1
		return df
		"""
		summ_extractors = [k for k in self.summary_extractors.keys() if experiment_name in k]
		df = None
		cnt = 0
		for s in summ_extractors:
			se = self.summary_extractors[s]
			summs_names = [n for n in se.list_available_summaries()
				if all(x in re.split(r"/|_", n) for x in variable_names)]
			if df is None:
				v0 = se.get_summary('0/special_summary/accept_ratio_replica_0')
				v_cols = [v[0] for v in v0[0]]
				start_indx = int(len(v_cols)*thresh)
				v_cols = v_cols[start_indx:]
				v_cols = ['sim_n', 'id'] + v_cols
				
				df = pd.DataFrame(columns=v_cols)

			for name in summs_names:
				summ_vals = se.get_summary(name)
				vals = [int(name.split('/')[0]), int(name.split('_')[-1])]
				vals = vals + [v[0] for v in summ_vals[1][start_indx:]]
				df.loc[cnt] = vals
			
		return df
		"""
	def get_accept_ratio_mean_std_report(self, df, experiment_num):
    
		# ids == replicas or ordered replicas
		n_ids = int(df.id.max()) + 1

		df_exp = df[df.experiment==experiment_num]

		last_column_name = df.columns.tolist()[-1]
		res = {}
		for id_ in range(n_ids):
			d = df_exp[df_exp.id==id_]
			d = d[[last_column_name]]
			stddev = d[[last_column_name]].std(axis=0, ddof=1).tolist()[-1]
			mean = d[[last_column_name]].mean(axis=0).tolist()[-1]
			res[id_] = {'mean':mean,
			'stddev':stddev}
		return res
		
	def get_mean_of_means_from_report(self, report):
		means = [report[k]['mean'] for k in report]
		return sum(means) / len(means)
		
	def get_mean_of_stddevs_from_report(self, report):
		stddevs = [report[k]['stddev'] for k in report]
		return sum(stddevs) / len(stddevs)

	def get_stddev_of_means_from_report(self, report):
		means = [report[k]['mean'] for k in report]
		return np.std(means, ddof=1)

	def get_stddev_of_stddevs_from_report(self, report):
		stddevs = [report[k]['stddev'] for k in report]
		return np.std(stddevs, ddof=1)

	def plot_report(self, report):
		return 1

class SummaryExtractor(object):

	def __init__(self, name):
		

		self._dir = Dir(name)
		self.all_summs_dict = {}
		
		for i in range(100):
			try:
				self.all_summs_dict.update(extract_summary(
					self._dir.log_dir + self._dir.delim + str(i)), delim=self._dir.delim)
			except FileNotFoundError: 
				self.all_summs_dict.pop('delim', None)
				print(i, 'simulations')
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
		return sorted(set([k for k in self.all_summs_dict.keys()]))
		

	def plot(self, keys=['valid'], match=None, add_swap_marks=False):
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
			js = self.get_description()
			step = js['swap_attempt_step']
			s = self.list_available_summaries()[0]
			x, y = self.get_summary(s)
			len_ = int(x[-1][0])
			for i in range(0, len_, step):
				ax.axvline(x=i)
		return fig

	def get_description(self):
		d = self._dir.delim
		file = self._dir.log_dir.replace('summaries'+d, 'summaries'+d+'compressed'+d)
		with open(os.path.join(file, 'description.json')) as fo:
			js = json.load(fo)

		return js

	def _create_experiment_averages(self):
		all_keys = self.list_available_summaries()
		try:
			all_keys.sort(key=lambda x: x.split('/')[1] + x.split('/')[-1])
		except IndexError:
			
			raise
		completed_keys = []

		for k in all_keys:
			if k in completed_keys:
				continue
			name = '/'.join(k.split('/')[1:])
			arrays = [self.get_summary(str(i) + '/' + name, split=False)
				for i in range(self.n_experiments)]
			self.all_summs_dict['mean/' + name] = np.mean(np.array(arrays), axis=0)


def extract_summary(log_dir, delim="/"):
	"""
	Extracts summaries from simulation `name`

	Args:
		log_dir: directory
		tag: summary name (e.g. cross_entropy, zero_one_loss ...)

	Returns:
		A dict where keys are names of the summary scalars and
		vals are numpy arrays of tuples (step, value)

	"""	

	delim ="\\" if 'win' in sys.platform else '/'
	
	compressed_dir = log_dir.replace('summaries'+delim, 'summaries'+delim+'compressed'+delim)
	summary_filename = os.path.join(compressed_dir, 'summary.pickle') 
	
	src_description_file = os.path.join(delim.join(log_dir.split(delim)[:-1]), 'description.json')
	dst_description_file = os.path.join(delim.join(compressed_dir.split(delim)[:-1]), 'description.json')

	if not os.path.exists(compressed_dir):
		
		os.makedirs(compressed_dir)

		with open(src_description_file) as fo:
			js = json.load(fo)
		
		with open(dst_description_file, 'w') as fo:
			json.dump(js, fo, indent=4)

	if os.path.exists(summary_filename):

		with open(summary_filename, 'rb') as fo:
			res = pickle.load(fo)
			return res
	else:


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
		
		with open(summary_filename, 'wb') as fo:
			pickle.dump(res, fo)
	
	
	return res

def extract_and_remove_simulation(path):
	se = SummaryExtractor(path)
	se._dir.clean_dirs()



