import os
import sys

import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.ticker as ticker
from scipy.interpolate import spline

import json
import pickle
import pandas as pd
import re

from simulation.simulation_builder.summary import Dir
from simulation.simulator_exceptions import InvalidExperimentValueError


class ExperimentPlot(object):

	def __init__(self, experiment_extractors):
		self._ee = {str(e):e for e in experiment_extractors}

	def __str__(self):
		return ', '.join(list(self._ee))



	def plot_final_crossentropy_per_ordered_replica_vs_tempfactor(self, 
		dataset_type='validation', custom_text = '', markeredgewidth=0.05, 
		elinewidth=0.5, set_ylim=None, set_xlim=None):
		
		fig, ax = plt.subplots()
		text = 'Average final cross entropy for best replica vs beta. \n'
		ax.title.set_text(text + custom_text)
		
		if set_ylim:
			ax.set_ylim(top=set_ylim)
		if set_xlim:
			ax.set_xlim(right=set_xlim)

		for k in self._ee:
			dict_ = self._ee[k].final_crossentropy_per_ordered_replica_vs_tempfactor_data(
				dataset_type)
		
			d = dict_[0]
			bar = ax.errorbar(x=d['x'], y=d['y'], yerr=d['err'], capsize=2,
				elinewidth=elinewidth, markeredgewidth=markeredgewidth)
			bar.set_label(self._ee[k])
		
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.6, 0.5))
		ax.set_xlabel('beta_i+1/beta_i')
		ax.set_ylabel('Final crossentropy loss for best replica')
		return fig

	def plot_best_crossentropy_per_ordered_replica_vs_tempfactor(self, 
		dataset_type='validation', custom_text = '', markeredgewidth=0.05, 
		elinewidth=0.5, set_ylim=None, set_xlim=None):
		
		fig, ax = plt.subplots()
		text = 'Average best cross entropy for best replica vs beta. \n'
		ax.title.set_text(text + custom_text)

		if set_ylim:
			ax.set_ylim(top=set_ylim)
		if set_xlim:
			ax.set_xlim(right=set_xlim)

		for k in self._ee:
			dict_ = self._ee[k].best_crossentropy_per_ordered_replica_vs_tempfactor_data(
				dataset_type)
		
			d = dict_[0]
			bar = ax.errorbar(x=d['x'], y=d['y'], yerr=d['err'], capsize=2,
				elinewidth=elinewidth, markeredgewidth=markeredgewidth)
			bar.set_label(self._ee[k])
		
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.6, 0.5))
		ax.set_xlabel('beta_i+1/beta_i')
		ax.set_ylabel('Best crossentropy loss for best replica')
		return fig

	def plot_accept_ratio_per_replica_vs_tempfactor(self, custom_text='', 
		markeredgewidth=0.05, elinewidth=0.5,
		set_ylim=None, set_xlim=None):
		fig, ax = plt.subplots()
		text = 'Average best cross entropy for best replica vs beta. \n'
		ax.title.set_text(text + custom_text)

		if set_ylim:
			ax.set_ylim(top=set_ylim)
		if set_xlim:
			ax.set_xlim(right=set_xlim)

		for k in self._ee:
			
			dict_ = self._ee[k].accept_ratio_per_replica_vs_tempfactor_data()
		
			d = dict_[0]
			bar = ax.errorbar(x=d['x'], y=d['y'], yerr=d['err'], capsize=2,
				elinewidth=elinewidth, markeredgewidth=markeredgewidth)
			bar.set_label(self._ee[k])
		
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.6, 0.5))
		ax.set_xlabel('beta_i+1/beta_i')
		ax.set_ylabel('Average over all replicas acceptance ratio vs beta_i+1/beta_i')
		return fig


class ExperimentExtractor(object):
	def __init__(self, experiment_names):
		names = list(set(['_'.join(e.split('_')[:-1])
			for e in experiment_names]))
		if len(names) > 1:
			#print(names)
			raise ValueError('Simulations must be from the same experiments, but given:',
				names)
		self._name = names[0]
		self._se = {e:SummaryExtractor(e)
			for e in experiment_names}

		self.df = None


	def __str__(self):
		return self._name

	def _create_dataframe_all_vals(self):
		if self.df is not None:
			return self.df
		
		loss_matching_keys = ['valid', 'ordered', 'cross', 'entropy']




		
		

		for e in self._se:
			
			n_replicas = int(e.split('_')[6])
			surface_view = e.split('_')[7]
			swap_attempt_step = int(e.split('_')[10])
			beta_0 = float(e.split('_')[8])
			experiment_num = int(e.split('_')[-1])
			experiment_name = e

			if self.df is None:
				cols = [
					'experiment_name',
					'experiment_num',
					'simulation_num', 
					'n_replicas',
					'replica_id', 
					'temp_factor', 
					'beta_0',
					'swap_attempt_step',
					'surface_view',
					'final_loss',
					'min_loss',
					'accept_ratio'
					]
				"""
				for i in range(n_replicas):
					cols.append('accept_ratio_replica_' + str(i))
				"""
				self.df = pd.DataFrame(columns=cols)
				index = 0
			for summ_name in self._se[e].list_available_summaries():
				if (all(x in summ_name for x in loss_matching_keys) and
					'mean' not in summ_name) :
					try:
						replica_id = int(summ_name.split('/')[1].split('_')[-1])
						sim_num = int(summ_name.split('/')[0])
						accept_summ_name = str(sim_num) + '/special_summary/accept_ratio_ordered_' + str(replica_id)
					except ValueError:
						print(summ_name)
						print(summ_name.split('/')[0])
						print(summ_name.split('/')[1].split('_')[-1])
						raise

					last_loss_val = self._se[e].get_summary(summ_name)[1][-1][0]
					min_loss_val = self._se[e].get_summary(summ_name)[1].min()
					accept_ratio_val = self._se[e].get_summary(accept_summ_name)[1][-1][0]
					temp_factor = self._se[e].get_description()['temp_factor']

					vals = [
						experiment_name,
						experiment_num,
						sim_num,
						n_replicas,
						replica_id,
						temp_factor,
						beta_0,
						swap_attempt_step,
						surface_view,
						last_loss_val,
						min_loss_val,
						accept_ratio_val
					]
					try:
						self.df.loc[index] = vals
					except ValueError:
						print(len(cols), len(vals))
						raise
					index += 1

		

		return self.df



	def _create_dataframe(self, matching_keywords):
		

		df = None
		index = 0

		for e in self._se:
			
			n_replicas = int(e.split('_')[6])
			surface_view = e.split('_')[7]
			swap_attempt_step = int(e.split('_')[10])
			beta_0 = float(e.split('_')[8])
			experiment_num = int(e.split('_')[-1])

			if df is None:
				cols = [
					'experiment_num',
					'simulation_num', 
					'n_replicas',
					'replica_id', 
					'temp_factor', 
					'beta_0',
					'swap_attempt_step',
					'surface_view',
					'final_matched_value',
					'min_matched_value',
					]
				"""
				for i in range(n_replicas):
					cols.append('accept_ratio_replica_' + str(i))
				"""
				df = pd.DataFrame(columns=cols)

			matched = self._extract_matching_summary_names_from_summary_extractor(
				e, matching_keywords)
			temp_factor = self._se[e].get_description()['temp_factor']

			for summ in matched:
				# summary_name example: '0/valid_ordered_0/cross_entropy'
				# accept_ratio example: '0/special_summary/accept_ratio_ordered_0'
				try:
					simulation_num = int(summ.split('/')[0])
				except ValueError:
					continue
				
				if ('accept' in matching_keywords and
					'ratio' in matching_keywords):
					replica_id = int(summ.split('_')[-1])
				else:
					replica_id = int(summ.split('/')[1].split('_')[-1])
				last_val = self._se[e].get_summary(summ)[1][-1]
				min_val = self._se[e].get_summary(summ)[1].min()
				#accept_ratio_replica = [self._se[e].get_summary('a')]
				


				vals = [
					experiment_num,
					simulation_num, 
					n_replicas,
					replica_id, 
					temp_factor, 
					beta_0,
					swap_attempt_step,
					surface_view,
					last_val[0],
					min_val]

				try:
					df.loc[index] = vals
				except ValueError:
					print(len(cols), len(vals))
					raise
				index += 1
		return df

	def best_crossentropy_per_ordered_replica_vs_tempfactor_data(self,
		dataset_type='validation'):
		dataset_type = ('valid' 
			if dataset_type == 'validation' else dataset_type)

		if dataset_type not in ['test', 'train', 'valid']:
			raise ValueError("dataset_type must be only on of test/train/validation.")

		matching_keywords = ['cross', 'entropy', 'ordered', dataset_type]
		df = self._create_dataframe(matching_keywords)

		n_replicas = list(set(df['n_replicas'].get_values().tolist()))
		if len(n_replicas) > 1:
			raise ValueError('n_replicas must be same for every experiment.')

		n_replicas = n_replicas[0]
		
		experiment_num_list = list(set(df.experiment_num.get_values().tolist()))
		
		res = {}
		for exp in sorted(experiment_num_list):
			
			for i in range(n_replicas):
				if i not in res:
					res[i] = {'x':[], 'y':[], 'err':[]}
				#_df = df[df['experiment_num']==exp]
				_df = df[(df['experiment_num']==exp) & (df['replica_id']==i)]
				
				mean = _df['min_matched_value'].mean(axis=0)
				std = _df['min_matched_value'].std(axis=0)
				tempf = list(set(_df['temp_factor'].get_values().tolist()))
				if len(tempf) > 1:
					raise ValueError('Somewhere above there is a bug...')

				tempfactor = tempf[0]
				res[i]['x'].append(tempfactor)
				res[i]['y'].append(mean)
				res[i]['err'].append(std)



		return res 

	def accept_ratio_per_replica_vs_tempfactor_data(self):
		df = self._create_dataframe(['accept', 'ratio', 'replica'])
		n_replicas = list(set(df['n_replicas'].get_values().tolist()))
		if len(n_replicas) > 1:
			raise ValueError('n_replicas must be same for every experiment.')

		n_replicas = n_replicas[0]
		
		experiment_num_list = list(set(df.experiment_num.get_values().tolist()))
		
		res = {}
		for exp in sorted(experiment_num_list):
			
			for i in range(n_replicas):
				if i not in res:
					res[i] = {'x':[], 'y':[], 'err':[]}
				#_df = df[df['experiment_num']==exp]
				_df = df[(df['experiment_num']==exp) & (df['replica_id']==i)]
				
				mean = _df['final_matched_value'].mean(axis=0)
				std = _df['final_matched_value'].std(axis=0)
				tempf = list(set(_df['temp_factor'].get_values().tolist()))
				if len(tempf) > 1:
					raise ValueError('Somewhere above there is a bug...')

				tempfactor = tempf[0]
				res[i]['x'].append(tempfactor)
				res[i]['y'].append(mean)
				res[i]['err'].append(std)



		return res 

	def accept_ratio_per_replica_vs_best_loss_vs_tempfactor_data(self,
		dataset_type='validation'):

		dataset_type = ('valid' 
			if dataset_type == 'validation' else dataset_type)

		if dataset_type not in ['test', 'train', 'valid']:
			raise ValueError("dataset_type must be only on of test/train/validation.")

		accept_ratio = self.accept_ratio_per_replica_vs_tempfactor_data()
		best_loss = self.best_crossentropy_per_ordered_replica_vs_tempfactor_data(
			dataset_type)

		# x==tempfactor, y==accept_ratio, z==best_loss
		res = accept_ratio[0]
		res['z'] = best_loss[0]['y']

		return res




	def final_crossentropy_per_ordered_replica_vs_tempfactor_data(self, 
		dataset_type='validation'):
		"""Returns avg final val of crossentropy per replica vs tempfactor.

			Args: 
				datasest_type: test/train/validation 

			Returns:
				Dataframe containing all values.

		"""
		
		dataset_type = ('valid' 
			if dataset_type == 'validation' else dataset_type)

		if dataset_type not in ['test', 'train', 'valid']:
			raise ValueError("dataset_type must be only on of test/train/validation.")

		matching_keywords = ['cross', 'entropy', 'ordered', dataset_type]
		df = self._create_dataframe(matching_keywords)

		n_replicas = list(set(df['n_replicas'].get_values().tolist()))
		if len(n_replicas) > 1:
			raise ValueError('n_replicas must be same for every experiment.')

		n_replicas = n_replicas[0]
		
		experiment_num_list = list(set(df.experiment_num.get_values().tolist()))
		
		res = {}
		for exp in sorted(experiment_num_list):
			
			for i in range(n_replicas):
				if i not in res:
					res[i] = {'x':[], 'y':[], 'err':[]}
				#_df = df[df['experiment_num']==exp]
				_df = df[(df['experiment_num']==exp) & (df['replica_id']==i)]
				
				mean = _df['final_matched_value'].mean(axis=0)
				std = _df['final_matched_value'].std(axis=0)
				tempf = list(set(_df['temp_factor'].get_values().tolist()))
				if len(tempf) > 1:
					raise ValueError('Somewhere above there is a bug...')

				tempfactor = tempf[0]
				res[i]['x'].append(tempfactor)
				res[i]['y'].append(mean)
				res[i]['err'].append(std)



		return res 


	

	def _extract_matching_summary_names_from_summary_extractor(self, 
		experiment_name, matching_keywords):
		"""Filters available summaries names based on matching_keywords vals.

		Args:
			summary_name: Name of the experiment (one of the 
				experiment_names provided during initialization).
			matching_keywords: All summaries that has ALL of these
				keywords will be matched and their names returned.
				NOTE: To match, the string splits based on r"/|_".
		Returns:
			Names of the found summaries of SummaryExtractor 
			experiment_name.

		"""

		se = self._se[experiment_name]
		matched = [s for s in se.list_available_summaries() 
			if all(x in re.split(r"/|_", s) for x in matching_keywords)]

		return matched



















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
	def _set_ticks(self, ax, vals=None):
		x, y = self.get_summary('0/train_ordered_0/cross_entropy')
		last_step = int(x[-1][0]) + 100
		vals = vals if vals is not None else range(0, last_step, int(last_step/14))

		ax.set_xticks([round(v, -3) for v in vals])


	def plot_diffusion(self, add_swap_marks=False, N=0, title='diffusion'):
		#N = number of simulation to show
		n_col = 0
		keys = 'diffusion'
		match = None
		fig, ax = plt.subplots()
		for s in self.list_available_summaries():
			summ_name = s.split('/') if match == 'exact' else s
			try:
				n = int(s.split('/')[0])
			except:
				continue
			if n != N:
				continue
			if all(x in summ_name for x in keys):
				x, y = self.get_summary(s)
				ax.plot(x, y, label=s)
				n_col += 1
		js = self.get_description()
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.2, 0.5))
		ax.title.set_text(title)
		if int(self.get_description()['n_epochs']) > 15:
			self._set_ticks(ax)
		
		
		#print(ax.get_xticks())

		if add_swap_marks:
			summ_name = str(N) + '/special_summary/swapped_replica_pair'
			js = self.get_description()
			step = js['swap_attempt_step']
			burn_in_period = int(js['burn_in_period'])
			#s = self.list_available_summaries()[0]
			x, y = self.get_summary(summ_name)

			len_ = int(x[-1][0])
			cnt = 0
			for i in range(0, len_, step):
				#print(x[int(i/len_)], int(i/len_), i, len_)
				try:
					if y[cnt][0] != -1 and x[cnt][0]> burn_in_period:
						ax.axvline(x=i, linewidth=0.9, linestyle=':')
				except:
					print(cnt, y.shape, x.shape, summ_name)
					continue
				cnt += 1
		fig.set_size_inches(12, 4.5) # (width, height)
		return fig



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
		

	def plot(self, keys=['valid'], match=None, add_swap_marks=False, title='', log_y=False):
		n_col = 0
		js = self.get_description()
		fig, ax = plt.subplots()
		for s in self.list_available_summaries():
			summ_name = s.split('/') if match == 'exact' else s
			if all(x in summ_name for x in keys):
				x, y = self.get_summary(s)
				#print(x.shape)
				#x = np.reshape(x, len(x))
				#y = np.reshape(y, len(y))
				#print(x.shape)
				if 'noise' in keys:
					ax.plot(x, y, label=s)
				else:
					x = np.ndarray.flatten(x)
					y = np.ndarray.flatten(y)

					x_new = np.linspace(x.min(), x.max(), x.shape[0]*4)
					y_new = spline(x, y, x_new)
					ax.plot(x_new, y_new, label=s)
				n_col += 1
				
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=(1.2, 0.5))
		ax.title.set_text(title)
		if int(self.get_description()['n_epochs']) > 15:
			self._set_ticks(ax)

		if log_y:
			ax.set_ylabel('log(beta)')
		
			
		ax.set_xlabel('1 training step == ' + str(js['batch_size']) + ' samples')
		#start, end = ax.get_xlim()
		#ax.xaxis.set_ticks(np.arange(start, end, 1000))

		if add_swap_marks:
			
			step = js['swap_attempt_step']
			s = self.list_available_summaries()[0]
			x, y = self.get_summary(s)
			len_ = int(x[-1][0])
			for i in range(0, len_, step):
				ax.axvline(x=i)
		if log_y:
			plt.yscale('log', basey=js['temp_factor'])

		fig.set_size_inches(12, 4.5) # (width, height)
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
	def get_min_val(self, summ_name):
		x, y = self.get_summary(summ_name)
		return(x[y.argmin()][0], y.min()) 

	def print_report(self):
		print(self.get_description()['temp_factor'])
		print('best accuracy on test dataset:',self.get_min_val('0/test_ordered_0/zero_one_loss'))

		print()
		print('cross entropy:')
		print('min_cross_valid_train:', self.get_min_val('0/train_ordered_0/cross_entropy'))
		print('min_cross_valid_test:', self.get_min_val('0/test_ordered_0/cross_entropy'))
		print('min_cross_valid_validation:', self.get_min_val('0/valid_ordered_0/cross_entropy'))
		print('stun:')
		print('min_stun_train:', self.get_min_val('0/train_ordered_0/stun'))
		print('min_stun_test:', self.get_min_val('0/test_ordered_0/stun'))
		print('min_stun_validation:', self.get_min_val('0/valid_ordered_0/stun'))
		print()
		print('accept_ratio:', self.get_summary('0/special_summary/accept_ratio')[1][-1][0])

		fig = self.plot_diffusion(add_swap_marks=True)

		fig = self.plot(['noise', 'replica', 'mean'], 
			title='mixing between replicas', 
			log_y=True)

		fig = self.plot(['accept', 'ratio', 'replica', 'mean'], title='accept_ratio')

		fig = self.plot(['cross', 'entropy', 'ordered', 'mean', 'test'], 
			title='cross entropy for test dataset')

		fig = self.plot(['stun', 'ordered', 'mean', 'test'], 
			title='STUN loss for test dataset')

		fig = self.plot(['zero', 'one', 'loss', 'mean', 'test', 'ordered'], 
			title='0-1 loss for test dataset')




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
