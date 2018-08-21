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