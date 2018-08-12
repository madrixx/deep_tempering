import os

import tensorflow as tf

from simulation.simulation_builder.summary import Dir

__DEBUG__ = False

def extract_summary(name, tag):
	"""Extracts summaries from simulation `name`

	Args:
		name: name of the simulation
		tag: summary name (e.g. loss, accuracy ...)

	Returns:
		A dict where keys are names of the tf.summary.FileWriter folders
		and vals are lists with summary values. 
	"""
	
	current_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
	dirpath = os.path.join(current_dir, 'summaries', name)
	res = {}
	for file in os.listdir(dirpath):
		fullpath = os.path.join(dirpath, file)
		if os.path.isdir(fullpath):
			res[file] = []
			for _file in os.listdir(fullpath):
				print(_file)
				for e in tf.train.summary_iterator(os.path.join(fullpath, _file)):
					for v in e.summary.value:
						if tag in v.tag:
							res[file].append(v.simple_value)

	return res



