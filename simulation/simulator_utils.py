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
from simulation.simulator_exceptions import InvalidExperimentValueError

__DEBUG__ = False


class MultiExperimentSummaryExtractor(object):

	def __init__(self, experiments):
		experiments = list(set(experiments))
		self.summary_extractors = {e:SummaryExtractor(e)
			for e in experiments}
		self.df = None

	def get_summary_extractor(self, name, simulation_num):
		try:
			return self.summary_extractors[name + '_' + str(simulation_num)]
		except KeyError:
			sim_numbers = self.summary_extractors.keys().sort(key=lambda x: x.split('_')[-1])
			print('The max number of simulation is', sim_numbers[-1], ', but given', simulation_num)

	def get_summary_extractor_by_experiment_num(self, experiment_num):
		try:
			name = [n for n in self.summary_extractors if str(experiment_num) == n.split('_')[-1]][0]
		
			return self.summary_extractors[name]
		except:
			return None

	
	def plot(self, keys, match=None, param_min=None, param_max=None, mark_lines=None, axis_color='royalblue'):
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
					if param_name == 'tempfactor':
						param_name = 'temp_factor'
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
		#ax.xaxis.label.set_color('red')
		#ax.yaxis.label.set_color('red')
		ax.tick_params(axis='x', colors='royalblue')
		ax.tick_params(axis='y', colors='royalblue')
		ax.spines['bottom'].set_color('royalblue')
		ax.spines['left'].set_color('royalblue')

		return fig

	def create_df(self, variable_names, thresh=0.0, df_name=None):
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
				if 'accept' in variable_names:
					try:
						id_ = int(summ_name.split('/')[0])
					except ValueError:
						print(summ_name.split('_'))
						raise
				else:

					try:
						id_ = int(re.split(r"/|_", summ_name)[-3])
					except:
						print(summ_name)
						print(re.split(r"/|_", summ_name))
						raise
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
				try:
					df.loc[index] = values
				except ValueError:
					print(len(values))
					print(len(cols))
					raise
				index += 1
			experiment_num += 1
		return df
		

	def get_last_value_mean_std_report(self, df, experiment_num):
		n_ids = int(df.id.max())

		df_exp = df[df.experiment==experiment_num]

		columns = df.columns.tolist()[4:]
		_df = df_exp[columns]
		res = {}

		for id_ in range(n_ids):
			d = _df[_df.id==id_]
			vals = d[[columns[-1]]]
			mean = vals.mean(axis=0)
			stddev = vals.std(axis=0, ddof=1)
			res = {
				'mean':mean,
				'stddev':stddev
			}
		return res

	def get_best_value_mean_std_report(self, df, experiment_num):
		n_ids = int(df.id.max())

		df_exp = df[df.experiment==experiment_num]

		columns = df.columns.tolist()[4:]
		_df = df_exp[columns]
		res = {}

		for id_ in range(n_ids):
			d = _df[_df.id==id_]
			vals = d[d>.00000001].min(axis=1)
			mean = vals.mean(axis=0)
			stddev = vals.std(axis=0, ddof=1)
			res = {
				'mean':mean,
				'stddev':stddev
			}
		return res

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

	def generate_accept_ratio_data_summarized(self):
		
		mean_of_means = []
		stddev_of_means = []
		temp_factors = []

		df = self.create_df(['accept', 'ratio', 'replica'])

		for i in range(100):
			temp_factor = self.get_summary_extractor_by_experiment_num(i)
			
			if temp_factor is not None:
				temp_factor = temp_factor.get_description()['temp_factor']
			else:
				break
			temp_factors.append(temp_factor)
			report = self.get_accept_ratio_mean_std_report(df, i)
			mom = self.get_mean_of_means_from_report(report)
			som = self.get_stddev_of_means_from_report(report)

			mean_of_means.append(mom)
			stddev_of_means.append(som)

		return mean_of_means, stddev_of_means, temp_factors

	def generate_cross_entropy_data_summarized(self):
		mean_of_means = []
		stddev_of_means = []
		temp_factors = []

		df = self.create_df(['cross', 'replica'])
		for i in range(100):
			temp_factor = self.get_summary_extractor_by_experiment_num(i)
			
			if temp_factor is not None:
				temp_factor = temp_factor.get_description()['temp_factor']
			else:
				break
			temp_factors.append(temp_factor)
			report = self.get_accept_ratio_mean_std_report(df, i)
			mom = self.get_mean_of_means_from_report(report)
			som = self.get_stddev_of_means_from_report(report)

			mean_of_means.append(mom)
			stddev_of_means.append(som)

		return mean_of_means, stddev_of_means, temp_factors



		
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


def generate_experiment_name(architecture_name=None, dataset='mnist', 
	temp_ratio=None, optimizer='PTLD', do_swaps=True, 
	swap_proba='boltzmann', n_replicas=None, surface_view='energy', beta_0=None, 
	loss_func_name='crossentropy', swap_attempt_step=None, burn_in_period=None, 
	learning_rate=None, n_epochs=None, version='v5'):
	
	
	"""Experiment name:
	<arhictecture>_<dataset>_<tuning parameter>_<optimizer>_...
	<dynamic=swaps occure/static=swaps don't occur>_...
	<n_replicas>_<surface view>_<starting_beta_>

		version: 'v2' means that summary stores diffusion value
		version: 'v3' means added burn-in period 
		version: 'v4' learning_rate has been added
		version: 'v5' has n_epochs in it
	"""


	if ((architecture_name is None or type(architecture_name) != str) 
		or (dataset is None or  dataset not in ['mnist', 'cifar'])
		or (temp_ratio is None) 
		or (optimizer is None or optimizer not in ['PTLD'])
		or (do_swaps is None or do_swaps not in [True, False, 'True', 'False'])
		or (swap_proba is None or swap_proba not in ['boltzmann'])
		or (n_replicas is None)
		or (surface_view is None or surface_view not in ['energy', 'info'])
		or (beta_0 is None)
		or (loss_func_name is None or loss_func_name not in ['crossentropy', 'zerooneloss', 'stun'])
		or (swap_attempt_step is None )
		or (burn_in_period is None)
		or (learning_rate is None)
		or (n_epochs is None)):
		raise InvalidExperimentValueError()

	name = architecture_name + '_' + dataset + '_'
	name = name + str(temp_ratio) + '_' + optimizer + '_'
	name = name + str(do_swaps) + '_' + str(swap_proba) + '_' + str(n_replicas) + '_'
	name = name + surface_view + '_' + str(beta_0) + '_' 
	name = name + loss_func_name + '_' + str(swap_attempt_step) + '_' + str(burn_in_period) + '_'
	name = name + str(learning_rate) + '_' + str(n_epochs) + '_' + version

	return name 

def clean_dirs(dir_):
    """Recursively removes all train, test and validation summary files \
            and folders from previos training life cycles."""

    try:
        for file in os.listdir(dir_):
            if os.path.isfile(os.path.join(dir_, file)):
                os.remove(os.path.join(dir_, file))
            else:
                clean_dirs(os.path.join(dir_, file))

        os.rmdir(dir_)
    except OSError:
        # if first simulation, nothing to delete
        return
