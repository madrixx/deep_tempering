"""Various helper functions.
'export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/'
"""
import os
import sys
import json
import pickle
import csv

from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd


from simulation.simulator_exceptions import InvalidExperimentValueError
from simulation.summary_extractor2 import SummaryExtractor

__DEBUG__ = False

def extract_summary(log_dir, delim="/"): # pylint: disable=too-many-locals
  """Extracts summaries from simulation `name`

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
  dst_description_file = (os.path.join(
      delim.join(compressed_dir.split(delim)[:-1]), 'description.json'))

  # pylint: disable=invalid-name
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
  """Convertes tf summary files to pickle objects and deletes tf files."""
  se = SummaryExtractor(path) #pylint:disable=invalid-name
  se._dir.clean_dirs() # pylint: disable=protected-access

# pylint:disable=too-many-arguments, too-many-locals
def generate_experiment_name(model_name=None,
                             dataset='mnist',
                             separation_ratio=None,
                             do_swaps=True,
                             n_replicas=None,
                             beta_0=None,
                             loss_func_name='crossentropy',
                             swap_step=None,
                             burn_in_period=None,
                             learning_rate=None,
                             n_epochs=None,
                             noise_type=None,
                             batch_size=None,
                             proba_coeff=1.0,
                             version='v7'):
  """Experiment name:
  <arhictecture>_<dataset>_<tuning parameter>_<optimizer>_...
  <dynamic=swaps occure/static=swaps don't occur>_...
  <n_replicas>_<surface view>_<starting_beta_>...
    version: 'v2' means that summary stores diffusion value
    version: 'v3' means added burn-in period
    version: 'v4' learning_rate has been added
    version: 'v5' has n_epochs in it
    version: 'v6' has batch_size and noise_type
    version: 'v7' has proba coefficient + optimizer has been removed
      + surface_view + swap_proba has been removed
  """
  nones = [(x, y)
           for x, y in zip(locals().keys(), locals().values()) if y is None]
  loss_func_name = loss_func_name.replace('_', '')
  # pylint:disable=too-many-boolean-expressions
  if ((model_name is None or not isinstance(model_name, str))
      or (dataset is None or  dataset not in ['mnist', 'cifar'])
      or (separation_ratio is None)
      or (do_swaps is None or do_swaps not in [True, False, 'True', 'False'])
      or (n_replicas is None)
      or (beta_0 is None)
      or (loss_func_name is None
          or loss_func_name not in ['crossentropy', 'zerooneloss', 'stun'])
      or (swap_step is None)
      or (burn_in_period is None)
      or (learning_rate is None)
      or (n_epochs is None)
      or (batch_size is None)
      or (noise_type is None)
      or (proba_coeff is None)):
    raise InvalidExperimentValueError(nones)

  name = model_name + '_' + dataset + '_'
  name = name + str(separation_ratio) + '_'
  name = name + str(do_swaps) + '_' + str(n_replicas) + '_'
  name = name + str(beta_0) + '_'
  name = name + loss_func_name + '_' + str(swap_step) + '_'
  name = name + str(burn_in_period) + '_'
  name = name + str(learning_rate) + '_' + str(n_epochs) + '_'
  name = name + str(batch_size) + '_'
  name = name + str(noise_type.replace('_', '')) + '_'
  name = name + str(proba_coeff) + '_' + version

  return name

# pylint:disable=inconsistent-return-statements, too-many-return-statements, too-many-branches
def get_value_from_name(full_name, value):
  """Works for names v7 and higher."""
  # pylint:disable=no-else-return
  if value == 'model_name':
    return full_name.split('_')[0]

  elif value == 'dataset':
    return full_name.split('_')[1]

  elif value == 'temp_ratio':
    return float(full_name.split('_')[2])

  elif value == 'do_swaps':
    return full_name.split('_')[3]

  elif value == 'n_replicas':
    return int(full_name.split('_')[4])

  elif value == 'beta_0':
    return float(full_name.split('_')[5])

  elif value == 'loss_func_name':
    return full_name.split('_')[6]

  elif value == 'swap_step':
    return float(full_name.split('_')[7])

  elif value == 'burn_in_period':
    return float(full_name.split('_')[8])

  elif value == 'learning_rate':
    return float(full_name.split('_')[9])

  elif value == 'n_epochs':
    return int(full_name.split('_')[10])

  elif value == 'batch_size':
    return int(full_name.split('_')[11])

  elif value == 'noise_type':
    return full_name.split('_')[12]

  elif value == 'proba_coeff':
    return float(full_name.split('_')[13])

  else:
    raise ValueError('Invalid value:', value)

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


class GlobalDescriptor(object): # pylint:disable=useless-object-inheritance
  """Helper for working with summary files."""
  def __init__(self):
    self.delim = "\\" if 'win' in sys.platform else '/'
    current_dir = self.delim.join(
        os.path.abspath(__file__).split(self.delim)[:-1])
    self.summaries_dir = os.path.join(current_dir, 'summaries')
    self.logfile_path = os.path.join(self.summaries_dir, 'test_logs.csv')

    self.columns = [
        'model_name',
        'noise_type',
        'n_epochs',
        'learning_rate',
        'n_replicas',
        'swap_step',
        'sep_ratio',
        'beta_0',
        'burn_in_period',
        'proba_coeff',
        'batch_size',
        'cross_entropy',
        'zero_one',
        'accept_ratio',
        'travel_time',
        'part_mix_ratio',
        'mix_ratio',
        ]

  def add_row(self, filename):
    """Adds row to the csv files."""
    # pylint:disable=invalid-name
    se = SummaryExtractor(filename)
    d = se.get_description()
    vals = [
        filename.split('_')[0],
        d['noise_type'],
        d['n_epochs'],
        d['learning_rate'],
        d['n_replicas'],
        d['swap_step'],
        d['separation_ratio'],
        d['noise_list'][0],
        d['burn_in_period'],
        d['proba_coeff'],
        d['batch_size'],
        ]
    # pylint:disable=protected-access, invalid-name
    xentropy = se.get_min_val('0/test_ordered_0/cross_entropy')
    v = float("{0:.4f}".format(xentropy[1]))
    vals = vals + [(v, int(xentropy[0]))]

    zero_one = se.get_min_val('0/test_ordered_0/zero_one_loss')
    v = float("{0:.4f}".format(zero_one[1]))
    vals = vals + [(v, int(zero_one[0]))]

    _, acc, err = se.get_accept_ratio_vs_separation_ratio_data()
    acc = float("{0:.4f}".format(acc))
    err = float("{0:.4f}".format(err))
    vals = vals + [str(acc) + '+/-' + str(err)]

    t_time, _, err = se.get_travel_time_vs_separation_ratio_data()
    v = float("{0:.4f}".format(t_time))
    err = float("{0:.4f}".format(err))
    vals = vals + [str(v) + '+/-' + str(err)]

    part_mix_ratio = float("{0:.4f}".format(se._get_visiting_ratio_data()))
    vals = vals + [part_mix_ratio]

    mix_ratio = float("{0:.4f}".format(se._get_mixing_ratio_data()))
    vals = vals + [mix_ratio]

    if not os.path.exists(self.logfile_path):
      with open(self.logfile_path, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(self.columns)

    with open(self.logfile_path, 'a') as fo:
      writer = csv.writer(fo)
      writer.writerow(vals)

  def get_dataframe(self):
    """Returns dataframe."""
    df = pd.read_csv(self.logfile_path)# pylint:disable=invalid-name
    return df

  def print_param_ranges(self, token='v7'): # pylint:disable=no-self-use
    """Prints all available hyperparams in files containing `token`."""
    path = os.path.join('simulation', 'summaries')
    path = os.path.join(path, 'compressed')
    files = [f for f in os.listdir(path)
             if token in f]

    model_names = [get_value_from_name(f, 'model_name')
                   for f in files]
    model_names = list(set(model_names))

    sep_ratios = [get_value_from_name(f, 'temp_ratio')
                  for f in files]
    sep_ratios = list(set(sep_ratios))
    try:
      sep_ratios = sorted([float(x) for x in sep_ratios])
    except: # pylint:disable=bare-except
      pass

    swap_steps = [get_value_from_name(f, 'swap_step')
                  for f in files]
    swap_steps = list(set(swap_steps))
    swap_steps = sorted([float(x) for x in swap_steps])

    burn_in_period = [get_value_from_name(f, 'burn_in_period')
                      for f in files]
    burn_in_period = sorted(list(set(burn_in_period)))

    learning_rate = [get_value_from_name(f, 'learning_rate')
                     for f in files]
    learning_rate = sorted(list(set(learning_rate)))
    learning_rate = [float(x) for x in learning_rate]

    batch_size = [get_value_from_name(f, 'batch_size')
                  for f in files]
    batch_size = sorted(list(set(batch_size)))
    batch_size = [float(x) for x in batch_size]

    proba_coeff = [get_value_from_name(f, 'proba_coeff')
                   for f in files]
    proba_coeff = sorted(list(set(proba_coeff)))
    proba_coeff = [float(x) for x in proba_coeff]

    print('Files found:', len(files))
    print('model names:', model_names)
    print('separation ratios:', sep_ratios)
    print('swap steps:', swap_steps)
    print('burn in periods:', burn_in_period)
    print('batch sizes:', batch_size)
    print('learning rates:', learning_rate)
    print('proba coeffs:', proba_coeff)

  # pylint:disable=no-self-use, unused-argument
  def filter_filenames(
      self, token='v7', model_name=None,
      learning_rate=None, burn_in_period=None,
      separation_ratio=None, batch_size=None,
      swap_step=None, proba_coeff=None,
      beta_0=None):
    """Filters files that has `token` based on args.

    Returns: A list of files satisfying arguments.
    """
    locals_ = locals().copy()
    locals_.pop('token')
    locals_.pop('self')
    keys = list(locals_.keys())
    _ = [locals_.pop(k) for k in keys if locals_[k] is None]
    if not all(isinstance(locals_[k], list) for k in locals_):
      raise TypeError('arguments must be lists')

    path = os.path.join('simulation', 'summaries')
    path = os.path.join(path, 'compressed')
    files = [f for f in os.listdir(path)
             if token in f]
    result = []

    for file in files:
      bool_test = []
      for key in locals_:
        val = get_value_from_name(file, key)
        bool_test.append(val in locals_[key])

      if all(bool_test):
        result.append(file)

    return result
