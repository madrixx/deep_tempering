import os
import sys

import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.ticker as ticker
from scipy.interpolate import spline
from scipy.special import erfcinv

import json
import pickle
import pandas as pd
import re
from operator import itemgetter

from simulation.simulation_builder.summary import Dir
from simulation.simulator_exceptions import InvalidExperimentValueError
from math import isinf

class ExperimentExtractor(object):

	def __init__(self, experiment_names):
		names = list(set(['_'.join(e.split('_')[:-1])
			for e in experiment_names]))
		if (len(names) > 1) and ('beta0' not in names[0]):

			raise ValueError('Simulations must be from the same experiments, but given:',
				names)

		self._name = names[0]
		self._se = {e:SummaryExtractor(e)
			for e in experiment_names}

	def __str__(self):
		return self._name

	def get_accept_ratio_vs_separation_ratio_data(self):
		"""Returns tuple of numpy arrays (separation_ratio, accept_ratio, stddev)"""
		
		sep_ratio = []
		accept_ratio = []
		stddev = []
		for se_name in self._se:
			se = self._se[se_name]
			sep, acc, err = se.get_accept_ratio_vs_separation_ratio_data()
			sep_ratio.append(sep)
			accept_ratio.append(acc)
			stddev.append(err)

		x, y, err = zip(*sorted(zip(sep_ratio, accept_ratio, stddev)))


		return list(x), list(y), list(err)


class Plot(object):

	def __init__(self):
		self.__first_use = True
		
	def legend(self, fig, ax, bbox_to_anchor=(1.2, 0.5),
		legend_title='', xlabel=None, ylabel=None, title=None,
		log_x=None, log_y=None,):
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
		ax.legend(loc='center right', fancybox=True, shadow=True, 
			bbox_to_anchor=bbox_to_anchor, title=legend_title)
		if title is not None: ax.title.set_text(title)
		if xlabel is not None: ax.set_xlabel(xlabel)
		if ylabel is not None: ax.set_ylabel(ylabel)
		if log_x is not None: plt.xscale('log', basex=log_x)
		if log_y is not None: 
		  plt.yscale('log', basey=log_y)        
		fig.set_size_inches(12, 4.5)# (width, height)
		self.__first_use = False

	def plot(self, x, y, err=None, fig=None, ax=None, label=None, 
		ylim_top=None, ylim_bottom=None, 
		splined_points_mult=6, elinewidth=0.5,
		markeredgewidth=0.05, ):

		def max_(array):
			#print(type(array))
			l = [a for a in array if isinf(a)==False]
			return max(l)


		if fig is None or ax is None:
			fig, ax = plt.subplots()

		if err is not None:
			plot_func = ax.errorbar
		else:
			plot_func = ax.plot




		# check if there are infinity
		x_ = [e  if isinf(e)==False else max_(x) + max_(x)*2
			for e in x]
		y_ = [e if isinf(e)==False else max_(y) + max_(y)*2
			for e in y] 

		x = np.array(x_)
		y = np.array(y_)

		if splined_points_mult is not None:
			x_new = np.linspace(x.min(), x.max(), x.shape[0]*splined_points_mult)
			y_new = spline(x, y, x_new)
			if err:
				err_new = spline(x, err, x_new)
				plot_func(x_new, y_new, yerr=err_new, 
					errorevery=x_new.shape[0]/splined_points_mult, 
					label=label, elinewidth=elinewidth,
					markeredgewidth=markeredgewidth)
			else:
				plot_func(x_new, y_new, label=label)

		else:
			if err:
				plot_func(x, y, yerr=err, label=label)
			else:
				plot_func(x, y, label)

		

		return fig


class SummaryExtractor(object):

	def __init__(self, name):
		

		self._dir = Dir(name)
		self.all_summs_dict = {}
		self._description = None
		
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

		self._n_simulations = self.get_description()['n_simulations']
		self._n_replicas = len(self.get_description()['noise_list'])

	def get_accept_ratio_vs_separation_ratio_data(self):
		"""Returns tuple (separation_ratio, accept_ratio, stddev)"""
		reps = {i:[] for i in range(self._n_replicas)}
		for s in range(self._n_simulations):
			for r in range(self._n_replicas):
				reps[r].append(self.get_summary('accept_ratio', replica_num=r, simulation_num=s)[1][-1])

		means = [sum(reps[i])/len(reps[i]) for i in range(self._n_replicas)]
		accept_ratio = np.mean(means)
		stddev = np.std(means)
		sep_ratio = self.get_description()['temp_factor']

		return sep_ratio, accept_ratio, stddev


	def _set_ticks(self, ax, vals=None):
		x, y = self._get_summary('0/train_ordered_0/cross_entropy')
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
				x, y = self._get_summary(s)
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
			x, y = self._get_summary(summ_name)

			len_ = int(x[-1][0])
			cnt = 0
			for i in range(0, len_, step):
				#print(x[int(i/len_)], int(i/len_), i, len_)
				try:
					if y[cnt][0] != -1 and x[cnt][0] > burn_in_period:
						ax.axvline(x=i, linewidth=0.9, linestyle=':')
				except:
					print(cnt, y.shape, x.shape, summ_name)
					continue
				cnt += 1
		fig.set_size_inches(12, 4.5) # (width, height)
		return fig



	def get_summary(self, summ_name, dataset_type='test', ordered=True, simulation_num=0, replica_num=0):
		"""Returns summary data by name.
		
		Args:
			summ_name: 'cross_entropy', 'stun', 'diffusion', 
				'zero_one_loss','noise'
			dataset_type: one of 'test'/'train'/'validation'
			ordered: if True, returns ordered values, otherwise, 
				per replica values.
			simulation_num: the number of simulations. Must be
				less than SummaryExtractor._n_simulations.
			replica_num: the number of a ordered/not ordered
				replica for which to return the simulation.

		Returns:
			(x, y) a tuple of numpy arrays if simulation_num < n_simulations
			None, otherwise

		"""
		if simulation_num >= self._n_simulations:
			print("The range of simulations must be less than:", self._n_simulations)
			return None


		req_str = str(simulation_num) + '/'

		if dataset_type == 'validation':
			req_str = req_str + 'valid_'
		elif dataset_type == 'test':
			req_str = req_str + 'test_'
		elif dataset_type == 'train_':
			req_str = req_str + 'train_'

		if ordered:
			req_str = req_str + 'ordered_'
		else:
			req_str = req_str + 'replica_'

		req_str = req_str + str(replica_num) + '/'

		if summ_name == 'cross_entropy':
			req_str = req_str + 'cross_entropy'
		elif summ_name == 'stun':
			req_str = req_str + 'stun'
		elif summ_name == 'zero_one_loss':
			req_str = req_str + 'zero_one_loss'
		elif summ_name == 'noise':
			req_str = req_str + 'noise'
		elif summ_name == 'diffusion':
			req_str = (str(simulation_num) 
				+ '/special_summary/diffusion_'
				+ str(replica_num)) 
		elif summ_name == 'accept_ratio':
			req_str = (str(simulation_num) 
				+ '/special_summary/accept_ratio_replica_'
				+ str(replica_num)) 
		
		x, y = self._get_summary(req_str)
		return np.ndarray.flatten(x), np.ndarray.flatten(y)



	def _get_summary(self, summ_name, split=True):
		"""Returns numpy arrays (x, y) of summaries.

		Args:
			summary_type: Name of the scalar summary
			

		Returns:
			(x, y) not flattened tuple of numpy arrays
		"""
		try:
			if split:
				return np.hsplit(self.all_summs_dict[summ_name], 2)
			else:
				return self.all_summs_dict[summ_name]
		except KeyError:
			print(self._dir.name)
			raise

	def list_available_summaries(self):
		return sorted(set([k for k in self.all_summs_dict.keys()]))
	
	def plot_mixing_between_replicas(self, dataset_type='train', simulation_num=0):
		plot = Plot()
		fig, ax = plt.subplots()


		
		for r in range(self._n_replicas):

			x, y = self.get_summary()
			plot.plot(x, y, fig=fig, ax=ax, 
				simulation_num=simulation_num, replica_id=r)
			plot.legend(fig, ax, legend_title='replica number', xlabel='STEP',
				ylabel='NOISE VALUE', title='mixing between replicas',
				log_y=self.get_description()['temp_factor'])

		return fig


	def plot(self, keys=['valid'], match=None, add_swap_marks=False, title='', log_y=False):
		n_col = 0
		js = self.get_description()
		fig, ax = plt.subplots()
		for s in self.list_available_summaries():
			summ_name = s.split('/') if match == 'exact' else s
			if all(x in summ_name for x in keys):
				x, y = self._get_summary(s)
				
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
			x, y = self._get_summary(s)
			len_ = int(x[-1][0])
			for i in range(0, len_, step):
				ax.axvline(x=i)
		if log_y:
			plt.yscale('log', basey=js['temp_factor'])

		fig.set_size_inches(11, 4.5) # (width, height)
		return fig



	def get_description(self):
		if self._description is not None:
			return self._description
		d = self._dir.delim
		file = self._dir.log_dir.replace('summaries'+d, 'summaries'+d+'compressed'+d)
		with open(os.path.join(file, 'description.json')) as fo:
			js = json.load(fo)
		self._description = js
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
			arrays = [self._get_summary(str(i) + '/' + name, split=False)
				for i in range(self.n_experiments)]
			self.all_summs_dict['mean/' + name] = np.mean(np.array(arrays), axis=0)
	
	def get_min_val(self, summ_name):
		x, y = self._get_summary(summ_name)
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
		print('accept_ratio:', self._get_summary('0/special_summary/accept_ratio')[1][-1][0])

		fig = self.plot_diffusion(add_swap_marks=True)

		fig = self.plot_mixing_between_replicas()
		

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
