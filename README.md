# Deep Tempering

##
Examples executing a single simulation

1. Simulate mnist dataset with multilayer perceptron using Langevin dynamics optimizer
```python
from tensorflow.examples.tutorials.mnist import input_data

from simulation.simulator import Simulator
from simulation.architectures.mnist_architectures import nn_mnist_architecture
from simulation.summary_extractor2 import SummaryExtractor
import simulation.simulator_utils as s_utils

MNIST_DATAPATH = 'simulation/data/mnist/'

mnist = input_data.read_data_sets(MNIST_DATAPATH)
train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
valid_data = mnist.validation.images
valid_labels = mnist.validation.labels


# set simulation parameters
n_simulations = 1
batch_size = 50
n_epochs = 30
burn_in_period = 2000
learning_rate = 0.01
beta_0 = 200
temp_factor = 1.1
n_replicas = 8
noise_list = [beta_0*temp_factor**i for i in range(n_replicas)]
simulation_name = 'test_simulation'
swap_attempt_step = 500 # 1 step==batch_size
description='Test simulation!'

# make sure that there are no directories that were previously created with same name
# otherwise, there will be problems extracting simulated results
s_utils.clean_dirs('simulation/summaries/' + simulation_name)
s_utils.clean_dirs('simulation/summaries/compressed/' + simulation_name)

# create and run simulation
sim = Simulator(
	architecture=nn_mnist_architecture,
	learning_rate=learning_rate,
	noise_list=noise_list,
	noise_type='betas',
	batch_size=batch_size,
	n_epochs=n_epochs,
	name=simulation_name,
	swap_attempt_step=swap_attempt_step,
	temp_factor=temp_factor,
	tuning_parameter_name='temp_factortemp',
	description=description,
	burn_in_period=burn_in_period,
	loss_func_name='cross_entropy'
	)

sim.train(train_data=train_data, train_labels=train_labels,
	test_data=test_data, test_labels=test_labels, 
	validation_data=valid_data, validation_labels=valid_labels)


# plot results
se = SummaryExtractor(experiment_name)
se.print_report()
```

2. Simulate mnist dataset with multilayer perceptron using dropout and GD optimizer

```python
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from simulation.simulator import Simulator
from simulation.summary_extractor2 import SummaryExtractor
import simulation.simulator_utils as s_utils
from simulation.architectures.mnist_architectures import nn_mnist_architecture_dropout
MNIST_DATAPATH = 'simulation/data/mnist/'

mnist = input_data.read_data_sets(MNIST_DATAPATH)
train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
valid_data = mnist.validation.images
valid_labels = mnist.validation.labels


# set simulation parameters
architecture_func = nn_mnist_architecture_dropout
n_simulations = 1
batch_size = 50
n_epochs = 12
burn_in_period = 2000
learning_rate = 0.01
noise_list = list(np.linspace(start=0.1, stop=0.99, num=8))
temp_factor = noise_list[1] - noise_list[0]
n_replicas = len(noise_list)
simulation_name = 'test2_simulation'
swap_attempt_step = 500 # 1 step==batch_size
description='Test2 simulation!'

# make sure that there are no directories that were previously created with same name
# otherwise, there will be problems extracting simulated results
s_utils.clean_dirs('simulation/summaries/' + simulation_name)
s_utils.clean_dirs('simulation/summaries/compressed/' + simulation_name)

# create and run simulation
sim = Simulator(
	architecture=architecture_func,
	learning_rate=learning_rate,
	noise_list=noise_list,
	noise_type='dropout',
	batch_size=batch_size,
	n_epochs=n_epochs,
	name=simulation_name,
	swap_attempt_step=swap_attempt_step,
	temp_factor=temp_factor,
	tuning_parameter_name='temp_factor',
	description=description,
	burn_in_period=burn_in_period,
	loss_func_name='cross_entropy'
	)

sim.train(train_data=train_data, train_labels=train_labels,
	test_data=test_data, test_labels=test_labels, 
	validation_data=valid_data, validation_labels=valid_labels)


# plot results
se = SummaryExtractor(simulation_name)
se.print_report()
```