class InvalidDatasetTypeError(Exception):
	pass
	
	def __str__(self, ):
		return """The dataset_type must be one of: 'train', 'test' or 'validation'"""


class InvalidArchitectureFuncError(Exception):
	pass
	
	def __init__(self, len_res, noise_type):
		self.len_res = str(len_res)
		self.noise_type = str(noise_type)
		msg = "`architecture` function must return 3 variables if noise_type is "
		msg = msg + "'random_normal'/'betas'  and 4 variables if `noise_type` is 'dropout'. "
		msg = msg + "The given `architecture` function returns " + self.len_res
		msg = msg + " variables and given `noise_type` is " + "'" + self.noise_type + "'"
		self.msg = msg
	
	def __str__(self):
		
		return self.msg

class NoGpusFoundError(Exception):
	pass
	def __init__(self):

		msg = 'No gpus found. (To remove this exception and execute on CPU, ' 
		msg = msg + 'set RAISE_IF_NO_GPU flag to false in device_placer.py file)'
		self.msg = msg

	def __str__(self):
		
		return self.msg

class InvalidLossFuncError(Exception):
	pass
	def __init__(self):
		msg = 'Invalid loss function. Possible functions are: `cross_entropy` and `zero_one_loss`'
		self.msg = msg

	def __str__(self):
		return self.msg

class InvalidExperimentValueError(Exception):
	pass

	def __init__(self, ):
		msg = 'Invalid value of a parameter. Valid parameters are: \n'
		msg = msg + "architecture_name: 'nn' or 'cnn' \n"
		msg = msg + "dataset: 'mnist' or 'cifar' \n"
		msg = msg + "tuning_param_name: 'swapstep' or 'tempfactor' \n"
		msg = msg + "optimizer: 'PTLD' \n"
		msg = msg + "do_swaps: True==do swap, False==do not swap\n"
		msg = msg + "swap_proba: boltzamann or MAYBE add more (TODO)\n"
		msg = msg + "n_replicas: int or str \n"
		msg = msg + "surface_view: 'energy' or 'info' \n"
		msg = msg + "beta_0: int or str \n"
		msg = msg + "loss_func_name: 'crossentropy' or 'zerooneloss' \n"
		msg = msg + "swap_attempt_step: int or str \n"
		msg = msg + "burn_in_period: int or float\n"
		msg = msg + "learning_rate: float"
		self.msg = msg

	def __str__(self):
		return self.msg

