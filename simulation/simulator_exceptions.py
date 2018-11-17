"""Defines custom exceptions."""

class InvalidDatasetTypeError(Exception): # pylint: disable=missing-docstring
  pass # pylint: disable=unnecessary-pass

  def __str__(self, ):
    return "The dataset_type must be one of: 'train', 'test' or 'validation'"


class InvalidArchitectureFuncError(Exception): # pylint: disable=missing-docstring
  pass # pylint: disable=unnecessary-pass

  def __init__(self, len_res, noise_type): # pylint: disable=super-init-not-called
    self.len_res = str(len_res)
    self.noise_type = str(noise_type)
    msg = "`model` function must return 3 variables if noise_type is "
    msg = msg + ("'random_normal'/'betas'  and 4 variables if "
                 + "`noise_type` is 'dropout'. ")
    msg = msg + "The given `model` function returns " + self.len_res
    msg = msg + (" variables and given `noise_type` is " + "'"
                 + self.noise_type + "'")
    self.msg = msg

  def __str__(self):
    return self.msg

class NoGpusFoundError(Exception): # pylint: disable=missing-docstring
  pass # pylint: disable=unnecessary-pass
  def __init__(self): # pylint: disable=super-init-not-called

    msg = 'No gpus found. (To remove this exception and execute on CPU, '
    msg = msg + 'set RAISE_IF_NO_GPU flag to false in device_placer.py file)'
    self.msg = msg

  def __str__(self):

    return self.msg

class InvalidLossFuncError(Exception): # pylint: disable=missing-docstring
  pass # pylint: disable=unnecessary-pass
  def __init__(self): # pylint: disable=super-init-not-called
    msg = ('Invalid loss function. Possible functions are: '
           +'`cross_entropy` and `zero_one_loss`')
    self.msg = msg

  def __str__(self):
    return self.msg

class InvalidNoiseTypeError(Exception): # pylint: disable=missing-docstring
  pass # pylint: disable=unnecessary-pass

  def __init__(self, noise_type, noise_types): # pylint: disable=super-init-not-called
    msg = "Invalid Noise Type. Avalable types are: "
    msg = msg + ', '.join(noise_types)
    msg = msg + ". But given: " + noise_type + ".\n"

    self.msg = msg

  def __str__(self):
    return self.msg

class InvalidExperimentValueError(Exception): # pylint: disable=missing-docstring
  pass # pylint: disable=unnecessary-pass

  def __init__(self, nones): # pylint: disable=super-init-not-called
    msg = ''
    if nones:
      msg = msg + "The following args have None values:\n"
      msg = msg + ", ".join([str(x[0])+':'+str(x[1]) for x in nones])
      msg = msg + "\n"
    msg = msg + 'Valid args are: \n'
    msg = msg + "model_name: 'nn/cnn' + \\{ 075, 125...\\} \n"
    msg = msg + "dataset: 'mnist' or 'cifar' \n"
    msg = msg + "do_swaps: True==do swap, False==do not swap\n"
    msg = msg + "swap_proba: boltzamann or MAYBE add more (TODO)\n"
    msg = msg + "n_replicas: int or str \n"
    msg = msg + "surface_view: 'energy' or 'info' \n"
    msg = msg + "beta_0: int or str \n"
    msg = msg + "loss_func_name: crossentropy or zerooneloss or stun' \n"
    msg = msg + "swap_step: int or str \n"
    msg = msg + "burn_in_period: int or float\n"
    msg = msg + "learning_rate: float\n"
    msg = msg + "n_epochs: int pr str\n"
    msg = msg + "batch_size: int or str\n"
    msg = msg + ("noise_type: see GraphBuilder.__noise_types"
                 + " for available noise vals\n")
    msg = msg + "proba_coeff: a float."
    self.msg = msg

  def __str__(self):
    return self.msg
