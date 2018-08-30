from __future__ import absolute_import

from tensorflow.python.client import device_lib

from simulation.simulator_exceptions import NoGpusFoundError

RAISE_IF_NO_GPU = True

def _get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _gpu_device_name(worker_id):
	gpus = _get_available_gpus()
	if RAISE_IF_NO_GPU and len(gpus) == 0:
		raise NoGpusFoundError()
	if len(gpus) == 0:
		return '/cpu:0'
	return '/gpu:' + str(worker_id % len(gpus))
