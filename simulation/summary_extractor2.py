"""Functions and classes for extraction, manipulation and plotting summary."""
import os
import sys
import json
from math import isinf
import pickle

from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # pylint: disable=useless-import-alias, unused-import
from scipy.interpolate import spline

from simulation.simulation_builder.summary import Dir

class ExperimentExtractor:

  def __init__(self, experiment_names, max_step=None):
    names = list(set(['_'.join(e.split('_')[:-1])
                      for e in experiment_names]))
    if ((len(names) > 1)
        and (len(list(set(['_'.join(e.split('_')[:-1])
                           for e in names]))) > 1)):

      raise ValueError('Simulations must have the same name, but given:',
                       names)

    try:
      self._name = names[0]
    except IndexError:
      print(names)
      print(experiment_names)
      raise
    self._se = {
        e:SummaryExtractor(e, max_step)
        for e in experiment_names}

  def __str__(self):
    return self._name

  def __repr__(self):
    return self.__str__()

  def get_accept_ratio_vs_separation_ratio_data(self): # pylint: disable=invalid-name
    """Returns tuple of numpy arrays (separation_ratio, accept_ratio, stddev)"""

    sep_ratio = []
    accept_ratio = []
    stddev = []
    for se_name in self._se:
      summ_ext = self._se[se_name]
      sep, acc, err = summ_ext.get_accept_ratio_vs_separation_ratio_data()
      sep_ratio.append(sep)
      accept_ratio.append(acc)
      stddev.append(err)

    x, y, err = zip(*sorted(zip(sep_ratio, accept_ratio, stddev))) # pylint: disable=invalid-name


    return list(x), list(y), list(err)

  def get_mixing_ratio_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    sep_ratio = []
    mixing_ratio = []
    for se_name in self._se:
      summ_ext = self._se[se_name]
      mix, sep = summ_ext.get_mixing_ratio_vs_separation_ratio_data()
      sep_ratio.append(sep)
      mixing_ratio.append(mix)

    x, y = zip(*sorted(zip(sep_ratio, mixing_ratio))) # pylint:disable=invalid-name

    return list(x), list(y)

  def get_visiting_ratio_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    sep_ratio = []
    mixing_ratio = []
    for se_name in self._se:
      summ_ext = self._se[se_name]
      mix, sep = summ_ext.get_visiting_ratio_vs_separation_ratio_data()
      sep_ratio.append(sep)
      mixing_ratio.append(mix)

    x, y = zip(*sorted(zip(sep_ratio, mixing_ratio))) # pylint:disable=invalid-name

    return list(x), list(y)

  def get_min_loss_value_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    sep_ratio = []
    min_vals = []
    for se_name in self._se:
      summ_ext = self._se[se_name]
      min_val, sep, err = summ_ext.get_min_loss_value_vs_separation_ratio_data()
      min_vals.append(min_val)
      sep_ratio.append(sep)

    x, y = zip(*sorted(zip(sep_ratio, min_vals))) # pylint:disable=invalid-name

    return list(x), list(y)

  def get_travel_time_vs_separation_ratio_data(self, inf_travel_time=1000): # pylint:disable=invalid-name
    sep_ratio = []
    travel_times = []
    for se_name in self._se:
      summ_ext = self._se[se_name]
      t_time, sep, err = summ_ext.get_travel_time_vs_separation_ratio_data() # pylint:disable=unused-variable
      travel_times.append(t_time)
      sep_ratio.append(sep)


    x, y = zip(*sorted(zip(sep_ratio, travel_times))) # pylint:disable=invalid-name
    x_ = list(x) # pylint:disable=invalid-name
    y_ = list(y) # pylint:disable=invalid-name
    x, y = [], [] # pylint:disable=invalid-name
    for i, j in zip(x_, y_):
      if not np.isinf(j):
        y.append(j)
        x.append(i)

    return x, y

class ExperimentExtractor_v2(ExperimentExtractor): # pylint: disable=invalid-name
  """No verification of input."""
  def __init__(self, experiment_names, max_step=None):
    names = list(set(['_'.join(e.split('_')[:-1])
                      for e in experiment_names]))

    try:
      self._name = names[0]
    except IndexError:
      print(names)
      print(experiment_names)
      raise
    self._se = {e:SummaryExtractor(e, max_step)
                for e in experiment_names}

class Plot:
  """Helper for plotting."""
  def __init__(self):
    self.__first_use = True

  def legend(self,
             fig,
             ax, # pylint: disable=invalid-name
             bbox_to_anchor=(1.2, 0.5),
             legend_title='',
             xlabel=None,
             ylabel=None,
             title=None,
             log_x=None,
             log_y=None):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
    ax.legend(
        loc='center right', fancybox=True, shadow=True,
        bbox_to_anchor=bbox_to_anchor, title=legend_title)

    if title is not None:
      ax.title.set_text(title)
    if xlabel is not None:
      ax.set_xlabel(xlabel)
    if ylabel is not None:
      ax.set_ylabel(ylabel)
    if log_x is not None:
      plt.xscale('log', basex=log_x)
    if log_y is not None:
      plt.yscale('log', basey=log_y)

    fig.set_size_inches(12, 4.5) # (width, height)
    self.__first_use = False

  def plot( # pylint:disable=invalid-name
      self, x, y, err=None, fig=None, ax=None, label=None,
      ylim_top=None, ylim_bottom=None, 
      splined_points_mult=6, elinewidth=0.5,
      markeredgewidth=0.05, fmt=None):

    def max_(array):
      list_ = [a for a in array if isinf(a) is False]
      return max(list_)

    if fig is None or ax is None:
      fig, ax = plt.subplots()

    if err is not None:
      plot_func = ax.errorbar
    else:
      plot_func = ax.plot

    # check if there are infinity vals and replace by finite "big" vals
    x_ = [e  if isinf(e) is False else max_(x) + max_(x)*2 # pylint:disable=invalid-name
          for e in x]
    y_ = [e if isinf(e) is False else max_(y) + max_(y)*2 # pylint:disable=invalid-name
          for e in y]

    x = np.array(x_) # pylint:disable=invalid-name
    y = np.array(y_) # pylint:disable=invalid-name

    if splined_points_mult is not None:
      x_new = np.linspace(x.min(), x.max(), x.shape[0]*splined_points_mult)
      y_new = spline(x, y, x_new)
      if err:
        err_new = spline(x, err, x_new)
        plot_func(
            x_new, y_new, yerr=err_new,
            errorevery=x_new.shape[0]/splined_points_mult,
            label=label, elinewidth=elinewidth,
            markeredgewidth=markeredgewidth)
      else:
        if fmt is None:
          plot_func(x_new, y_new, label=label)
        else:
          plot_func(x_new, y_new, fmt, label=label)

    else:
      if err:
        if fmt is None:
          plot_func(x, y, yerr=err, label=label)
        else:
          plot_func(x, y, fmt, yerr=err, label=label)
      else:
        if fmt is None:
          plot_func(x, y, label=label,)
        else:
          plot_func(x, y, fmt, label=label,)

    return fig

  def plot_loss(self, fig, ax, vals, ees, fmt, splined_points_mult=None):
    """Helper for plotting loss."""
    for k, ee in enumerate(ees):
        x, y = ee.get_min_loss_value_vs_separation_ratio_data()
        label = self.label_name_generator(n=vals['names'][k],
                                          lr=vals['learning_rate'][k],
                                          b_0=vals['beta_0'][k],
                                          batch=vals['batch_size'][k],
                                          proba=vals['proba_coeff'][k],
                                          burn=vals['burn_in_period'][k])
        self.plot(x=x,
                  y=y,
                  fig=fig,
                  ax=ax,
                  splined_points_mult=splined_points_mult,
                  label=label,
                  fmt=fmt[k])
        """
        x, y = ee.get_min_loss_value_vs_separation_ratio_data()
        label = self.label_name_generator(n=vals['names'][k],
                                          lr=vals['learning_rate'][k],
                                          b_0=vals['beta_0'][k],
                                          batch=vals['batch_size'][k],
                                          proba=vals['proba_coeff'][k],
                                          burn=vals['burn_in_period'][k])
        self.plot(x=x,
                  y=y,
                  fig=fig,
                  ax=ax,
                  splined_points_mult=splined_points_mult,
                  label=label,
                  fmt=fmt[k])

        x, y = ee.get_min_loss_value_vs_separation_ratio_data()
        label = self.label_name_generator(n=vals['names'][k],
                                          lr=vals['learning_rate'][k],
                                          b_0=vals['beta_0'][k],
                                          batch=vals['batch_size'][k],
                                          proba=vals['proba_coeff'][k],
                                          burn=vals['burn_in_period'][k])
        self.plot(x=x,
                  y=y,
                  fig=fig,
                  ax=ax,
                  splined_points_mult=splined_points_mult,
                  label=label,
                  fmt=fmt[k])
        """
    self.legend(fig, 
                ax, 
                legend_title = 'beta_0', 
                xlabel='SEPARATION RATIO', 
                ylabel='Min 0-1 Loss', 
                title='Min 0-1 vs Separation ratio')

  def label_name_generator(self, **args): # pylint:disable=unsused-argument, no-self-use
    """Helper for generating legend labels."""

    locals_ = locals()['args'].copy()
    label_name = "|".join([str(k)+':'+str(locals_[k]) for k in locals_])
    return label_name

class SummaryExtractor: # pylint:disable=too-many-instance-attributes
  """Extracts summaries from summary files."""
  def __init__(self, name, max_step=None):

    self._dir = Dir(name)
    self.all_summs_dict = {}
    self._description = None
    self._mixing_ratio = None
    self._visiting_ratio = None
    self._max_step = max_step
    self._travel_time = None

    for i in range(100):
      try:
        self.all_summs_dict.update(
            extract_summary(
                self._dir.log_dir + self._dir.delim + str(i)),
            delim=self._dir.delim)
      except FileNotFoundError:
        self.all_summs_dict.pop('delim', None)
        self.n_experiments = i
        self._create_experiment_averages()

        break

    self._n_simulations = self.get_description()['n_simulations']
    self._n_replicas = len(self.get_description()['noise_list'])

  def get_min_loss_value_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    """Returns a tuple min zero one loss and separation ratio."""
    sep_ratio = self.get_description()['separation_ratio']
    n_simulations = self.get_description()['n_simulations']
    min_vals = []

    for i in range(n_simulations):
      x, y = self.get_summary('zero_one_loss', simulation_num=i) # pylint: disable=invalid-name, unused-variable
      min_vals.append(y.min())

    min_val = np.mean(min_vals)
    err = np.std(min_vals)

    return min_val, sep_ratio, err

  def get_mixing_ratio_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    """Returns mixing ratio vs separation ratio values."""
    separation_ratio = self.get_description()['separation_ratio']
    mixing_ratio = self._get_mixing_ratio_data()
    return mixing_ratio, separation_ratio

  def get_visiting_ratio_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    """Visiting ratio is the fraction of temperatures that a replica has visited."""
    separation_ratio = self.get_description()['separation_ratio']
    mixing_ratio = self._get_visiting_ratio_data()
    return mixing_ratio, separation_ratio

  def get_travel_time_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    """Traveling time is the time takes to visit all temperatures."""
    if self._travel_time is not None:
      return self._travel_time
    sep_ratio = self.get_description()['separation_ratio']
    mixing_ratio = self._get_mixing_ratio_data()

    if mixing_ratio == 0.0:
      return np.inf, sep_ratio, 0.0

    def all_visited(d): # pylint: disable=invalid-name
      return all(d[k] != 0 for k in d)

    keys = [float("{0:.3f}".format(b))
            for b in self.get_description()['noise_list']]
    burn_in_period = self.get_description()['burn_in_period']
    n_replicas = self.get_description()['n_replicas']
    n_simulations = self.get_description()['n_simulations']

    reps = {k:0 for k in keys}
    travel_times = []

    for s in range(n_simulations): # pylint:disable=invalid-name
      for r in range(n_replicas): # pylint:disable=invalid-name
        x, y = self.get_summary( # pylint:disable=invalid-name
            'noise', simulation_num=s,
            replica_num=r, ordered=False, dataset_type='train')
        n_steps = int(x.shape[0])
        travel_time = None
        for step in range(n_steps):
          if (x[step] > burn_in_period
              and travel_time is None):
            travel_time = 0
          elif travel_time is None:
            continue
          else:
            travel_time += 1
          reps[self._get_key(y[step])] += 1
          if all_visited(reps):
            travel_times.append(travel_time)
            reps = {k:0 for k in keys}
            travel_time = 0

    travel_times_ = [t*self.get_description()['batch_size']
                     for t in travel_times]
    travel_time = np.mean(travel_times_)
    err = np.std(travel_times_)
    self._travel_time = (travel_time, sep_ratio, err)
    return self._travel_time


  def _get_mixing_ratio_data(self):
    """Fraction of replicas that visit both lowest and highest temperatures."""
    if self._mixing_ratio is not None:
      return self._mixing_ratio
    keys = [float("{0:.3f}".format(b))
            for b in self.get_description()['noise_list']]

    def get_key(n): # pylint: disable=invalid-name
      return keys[int(np.argmin([abs(k-n) for k in keys]))]


    reps = {k:0 for k in keys}
    mixing = {i:[] for i in range(self._n_replicas)}

    for s in range(self._n_simulations): # pylint:disable=invalid-name
      for r in range(self._n_replicas): # pylint:disable=invalid-name
        x, y = self.get_summary( # pylint:disable=invalid-name
            'noise', simulation_num=s,
            replica_num=r, ordered=False, dataset_type='train')
        n_steps = int(x.shape[0])

        for i in range(n_steps):
          if x[i] > self.get_description()['burn_in_period']:
            reps[get_key(y[i])] += 1

        if all(reps[x]!= 0 for x in reps):
          mixing[r].append(1)
        else:
          mixing[r].append(0)

        reps = {k:0 for k in keys}

    ratios = []

    for s in range(self._n_simulations): # pylint: disable=invalid-name
      ratio = sum([mixing[r][s] for r in range(self._n_replicas)])/self._n_replicas
      ratios.append(ratio)

    self._mixing_ratio = sum(ratios)/len(ratios)
    return self._mixing_ratio

  def _get_visiting_ratio_data(self):
    if self._visiting_ratio is not None:
      return self._visiting_ratio

    keys = [float("{0:.3f}".format(b))
        for b in self.get_description()['noise_list']]

    def get_key(n): # pylint: disable=invalid-name
      return keys[int(np.argmin([abs(k-n) for k in keys]))]


    reps = {k:0 for k in keys}
    mixing = {i:[] for i in range(self._n_replicas)}

    for s in range(self._n_simulations):
      for r in range(self._n_replicas):
        x, y = self.get_summary( # pylint:disable=invalid-name
            'noise', simulation_num=s,
            replica_num=r, ordered=False, dataset_type='train')
        n_steps = int(x.shape[0])

        for i in range(n_steps):
          if x[i] > self.get_description()['burn_in_period']:
            reps[get_key(y[i])] += 1
        visited = [1 if reps[x]!=0 else 0
          for x in reps ]
        mixing[r].append(sum(visited)/len(visited))

        reps = {k:0 for k in keys}

    ratios = []

    for s in range(self._n_simulations): # pylint: disable=invalid-name
      ratio = sum([mixing[r][s] for r in range(self._n_replicas)])/self._n_replicas
      ratios.append(ratio)
    self._visiting_ratio = sum(ratios)/len(ratios)
    return self._visiting_ratio


  def get_accept_ratio_vs_separation_ratio_data(self): # pylint:disable=invalid-name
    """Returns tuple (separation_ratio, accept_ratio, stddev)"""
    reps = {i:[] for i in range(self._n_replicas)}
    for s in range(self._n_simulations): # pylint: disable=invalid-name
      for r in range(self._n_replicas): # pylint: disable=invalid-name
        reps[r].append(self.get_summary(
            'accept_ratio', replica_num=r, simulation_num=s)[1][-1])

    means = [sum(reps[i])/len(reps[i])
             for i in range(self._n_replicas)]
    accept_ratio = np.mean(means)
    stddev = np.std(means)
    sep_ratio = self.get_description()['separation_ratio']

    return sep_ratio, accept_ratio, stddev


  def _set_ticks(self, ax, vals=None): # pylint: disable=invalid-name
    x, y = self._get_summary('0/train_ordered_0/cross_entropy') # pylint: disable=invalid-name, unused-variable
    last_step = int(x[-1][0]) + 100
    vals = vals if vals is not None else range(0, last_step, int(last_step/14))
    ax.set_xticks([round(v, -3) for v in vals])

  def plot_diffusion(self, add_swap_marks=False, N=0, title='diffusion'): # pylint:disable=invalid-name
    """Plots the values of diffusion."""
    # N is number of the simulation to show
    n_col = 0
    keys = 'diffusion'
    match = None
    fig, ax = plt.subplots() # pylint: disable=invalid-name
    for s in self.list_available_summaries():
      summ_name = (s.split('/') if match == 'exact' else s)
      try:
        n = int(s.split('/')[0]) # pylint: disable=invalid-name
      except:
        continue
      if n != N:
        continue
      if all(x in summ_name for x in keys):
        x, y = self._get_summary(s) # pylint: disable=invalid-name
        ax.plot(x, y, label=s)
        n_col += 1
    js = self.get_description() # pylint: disable=invalid-name
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
    ax.legend(
        loc='center right', fancybox=True, shadow=True, 
        bbox_to_anchor=(1.2, 0.5))
    ax.title.set_text(title)
    if int(self.get_description()['n_epochs']) > 15:
      self._set_ticks(ax)

    if add_swap_marks:
      summ_name = str(N) + '/special_summary/swapped_replica_pair'
      js = self.get_description() # pylint:disable=invalid-name
      step = js['swap_step']
      burn_in_period = int(js['burn_in_period'])

      x, y = self._get_summary(summ_name) # pylint: disable=invalid-name

      len_ = int(x[-1][0])
      cnt = 0
      for i in range(0, len_, step):

        try:
          if y[cnt][0] != -1 and x[cnt][0] > burn_in_period:
            ax.axvline(x=i, linewidth=0.9, linestyle=':')
        except: # pylint: disable=bare-except
          print(cnt, y.shape, x.shape, summ_name)
          continue
        cnt += 1
    fig.set_size_inches(12, 4.5) # (width, height)
    return fig



  def _get_key(self, n): # pylint: disable=invalid-name
    keys = [float("{0:.3f}".format(b))
            for b in self.get_description()['noise_list']]
    return keys[int(np.argmin([abs(k-n) for k in keys]))]

  def plot_mixing_between_replicas(self, mixing_log_y=None, dataset_type='train', simulation_num=0):
    plot = Plot()
    fig, ax = plt.subplots() # pylint:disable=invalid-name

    noise_list = self.get_description()['noise_list']
    key_map = {self._get_key(n):i+1 for i, n in enumerate(noise_list)}

    for r in range(self._n_replicas): # pylint:disable=invalid-name

      x, y = self.get_summary( # pylint:disable=invalid-name
          'noise', ordered=False, 
          dataset_type=dataset_type,
          simulation_num=simulation_num,
          replica_num=r)

      y_new = [key_map[self._get_key(i)] for i in y]

      plot.plot(
          x, y_new, fig=fig, ax=ax, label='replica_' + str(r),
          splined_points_mult=None)

    yticks_names = ["{0:.3f}".format(b) for b in noise_list]
    ax.yaxis.set_ticklabels([yticks_names[0]] + yticks_names)

    plot.legend(
        fig, ax, legend_title='replica number',
        xlabel='STEP', ylabel='NOISE VALUE',
        title='mixing between replicas',)

    return fig




  def get_summary(self, # pylint:disable=too-many-branches, too-many-arguments
                  summ_name,
                  dataset_type='test',
                  ordered=True,
                  simulation_num=0,
                  replica_num=0):
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
    elif dataset_type == 'train':
      req_str = req_str + 'train_'
    else:
      raise ValueError('Invalid dataset_type:', dataset_type)


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
    else:
      raise ValueError('Invalid summ_name argument:', summ_name)
    try:
      x, y = self._get_summary(req_str) # pylint: disable=invalid-name
    except KeyError:
      print(req_str)
      print(locals())
      raise

    x = np.ndarray.flatten(x) # pylint: disable=invalid-name
    y = np.ndarray.flatten(y) # pylint: disable=invalid-name

    if self._max_step is not None:
      i = 0
      while x[i] < self._max_step:
        i += 1
    else:
      i = np.inf

    if i < x.shape[0]:
      return x[:i], y[:i]
    return x, y



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

      return self.all_summs_dict[summ_name]

    except KeyError:
      print(summ_name)
      print(self._dir.name)
      raise

  def list_available_summaries(self):
    """Returns a list of all available summaries."""
    return sorted(set([k for k in self.all_summs_dict.keys()]))


  def plot(self, keys=['valid'], match=None, add_swap_marks=False, title='', log_y=False): # pylint: disable=dangerous-default-value, too-many-arguments
    n_col = 0
    js = self.get_description() # pylint:disable=invalid-name
    fig, ax = plt.subplots() # pylint:disable=invalid-name
    for s in self.list_available_summaries():
      summ_name = s.split('/') if match == 'exact' else s
      if all(x in summ_name for x in keys):
        x, y = self._get_summary(s) # pylint: disable=invalid-name

        if 'noise' in keys:
          ax.plot(x, y, label=s)
        else:
          x = np.ndarray.flatten(x) # pylint:disable=invalid-name
          y = np.ndarray.flatten(y) # pylint:disable=invalid-name

          x_new = np.linspace(x.min(), x.max(), x.shape[0]*4)
          y_new = spline(x, y, x_new)
          ax.plot(x_new, y_new, label=s)
        n_col += 1

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*2, box.height*2])
    ax.legend(
        loc='center right', fancybox=True, shadow=True, 
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

      step = js['swap_step']
      s = self.list_available_summaries()[0] # pylint:disable=invalid-name
      x, y = self._get_summary(s) # pylint:disable=invalid-name
      len_ = int(x[-1][0])
      for i in range(0, len_, step):
        ax.axvline(x=i)
    if log_y:
      plt.yscale('log', basey=js['separation_ratio'])

    fig.set_size_inches(11, 4.5) # (width, height)
    return fig



  def get_description(self):
    if self._description is not None:
      return self._description
    d = self._dir.delim # pylint:disable=invalid-name
    file = self._dir.log_dir.replace(
        'summaries'+d, 'summaries'+d+'compressed'+d)
    with open(os.path.join(file, 'description.json')) as fo:
      dict_ = json.load(fo)
    self._description = dict_
    if 'swap_step' not in self._description:
      self._description['swap_step'] = self._description['swap_attempt_step']
    elif 'swap_attempt_step' not in self._description:
      self._description['swap_attempt_step'] = self._description['swap_step']

    if 'separation_ratio' not in self._description:
      self._description['separation_ratio'] = self._description['temp_factor']
    return self._description

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
    """Returns minimum value for simulation `summ_name`."""
    x, y = self._get_summary(summ_name) # pylint:disable=invalid-name
    return(x[y.argmin()][0], y.min())

  def get_min_val_zero_one_loss(self):
    """Returns min zero-one loss value."""
    x, y = self.get_summary( # pylint:disable=invalid-name
        'zero_one_loss', dataset_type='test',
        ordered=True, simulation_num=0, replica_num=0)
    return (x[int(y.argmin())], y.min())

  def get_min_cross_valid(self):
    """Returns min cross validation value."""
    x, y = self.get_summary( # pylint:disable=invalid-name
        'cross_entropy', dataset_type='test',
        ordered=True, simulation_num=0, replica_num=0)

    return (x[int(y.argmin())], y.min())

  def print_report(self, mixing_log_y=None):
    """Prints simulation report."""
    print('Separation Ratio:', self.get_description()['separation_ratio'])
    print('Best Accuracy on test dataset:',self.get_min_val('0/test_ordered_0/zero_one_loss'))
    #print('Best Accuracy on test dataset:',self.get_min_val_zero_one_loss())
    print()
    print('cross entropy:')
    print('min_cross_valid_train:', self.get_min_val('0/train_ordered_0/cross_entropy'))
    print('min_cross_valid_test:', self.get_min_val('0/test_ordered_0/cross_entropy'))
    print('min_cross_valid_test:', self.get_min_cross_valid())
    print('min_cross_valid_validation:', self.get_min_val('0/valid_ordered_0/cross_entropy'))
    #print('stun:')
    #print('min_stun_train:', self.get_min_val('0/train_ordered_0/stun'))
    #print('min_stun_test:', self.get_min_val('0/test_ordered_0/stun'))
    #print('min_stun_validation:', self.get_min_val('0/valid_ordered_0/stun'))
    print()
    print('Acceptance Ratio:',
          self._get_summary('0/special_summary/accept_ratio')[1][-1][0])
    print()
    print('Mixing ratio (fraction of replicas that travelled all temperatures):',
          self._get_mixing_ratio_data())
    print('Average fraction of visited temperatures:',
          self._get_visiting_ratio_data())
    print('Burn in period:', self.get_description()['burn_in_period'])
    t_time, sep_ratio, err = self.get_travel_time_vs_separation_ratio_data()


    print('Travel Time:', t_time, '+/-', err)
    try:
      print('Proba Coeff:', self.get_description()['proba_coeff'])
    except KeyError:
      pass
    print('Noise:', self.get_description()['noise_list'])
    fig = self.plot_diffusion(add_swap_marks=True) # pylint:disable=unused-variable

    fig = self.plot_mixing_between_replicas(mixing_log_y)

    fig = self.plot(['accept', 'ratio', 'replica', 'mean'],
                    title='accept_ratio')

    fig = self.plot(['cross', 'entropy', 'ordered', 'mean', 'test'],
                    title='cross entropy for test dataset')


    fig = self.plot(['zero', 'one', 'loss', 'mean', 'test', 'ordered'],
                    title='0-1 loss for test dataset')

def extract_summary(log_dir, delim="/"): # pylint:disable=too-many-locals
  """
  Extracts summaries from simulation `name`

  Args:
    log_dir: directory
    tag: summary name (e.g. cross_entropy, zero_one_loss ...)

  Returns:
    A dict where keys are names of the summary scalars and
    vals are numpy arrays of tuples (step, value)

  """


  delim = ("\\" if 'win' in sys.platform else '/')

  compressed_dir = log_dir.replace(
      'summaries'+delim, 'summaries'+delim+'compressed'+delim)
  summary_filename = os.path.join(compressed_dir, 'summary.pickle')

  src_description_file = os.path.join(
      delim.join(log_dir.split(delim)[:-1]), 'description.json')
  dst_description_file = os.path.join(
      delim.join(compressed_dir.split(delim)[:-1]),
      'description.json')

  if not os.path.exists(compressed_dir):

    os.makedirs(compressed_dir)

    with open(src_description_file) as fo: # pylint: disable=invalid-name
      js = json.load(fo) # pylint: disable=invalid-name
    with open(dst_description_file, 'w') as _:
      json.dump(js, _, indent=4)

  if os.path.exists(summary_filename):

    with open(summary_filename, 'rb') as _:
      res = pickle.load(_)
      return res
  else:


    sim_num = log_dir.split(delim)[-1]
    res = {}
    for file in os.listdir(log_dir):
      fullpath = os.path.join(log_dir, file)

      if os.path.isdir(fullpath):

        for _file in os.listdir(fullpath):

          filename = os.path.join(fullpath, _file)

          ea = event_accumulator.EventAccumulator(filename) # pylint:disable=invalid-name
          ea.Reload()
          for k in ea.scalars.Keys(): 
            lc = np.stack( # pylint: disable=invalid-name
                [np.asarray([scalar.step, scalar.value])
                 for scalar in ea.Scalars(k)])
            key_name = sim_num + '/' + file + '/' +  k.split('/')[-1]
            key_name = '/'.join(key_name.split('/')[-3:])
            res[key_name] = lc

    with open(summary_filename, 'wb') as _:
      pickle.dump(res, _)

  return res
