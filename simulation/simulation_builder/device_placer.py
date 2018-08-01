from __future__ import absolute_import

from tensorflow.python.client import device_lib

def _get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	return [x.name for x in local_device_protos if x.device_type == 'GPU']


def _gpu_device_name(worker_id):
	gpus = _get_available_gpus()

	if len(gpus) == 0:
		return '/cpu:0'

	return '/gpu:' + str(worker_id % len(gpus))
