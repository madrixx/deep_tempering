# Deep Tempering

## Examples executing a single simulation

### 1. Simulate mnist dataset with multilayer perceptron using Langevin dynamics optimizer
```python
from tensorflow.examples.tutorials.mnist import input_data

from simulation.simulator import Simulator
from simulation.models.mnist_models import nn_mnist_model
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
batch_size = 50
n_epochs = 5
learning_rate = 0.01
beta_0 = 200
separation_ratio = 1.1
n_replicas = 8
noise_list = [beta_0*separation_ratio**i for i in range(n_replicas)]
name = 'ptld_simulation'
swap_step = 500 # 1 step==batch_size
burn_in_period = 2000
test_step = 500
description='Parallel Tempering with Langevin Dynamics'

# make sure that there are no directories that were previously created with same name
# otherwise, there will be problems extracting simulated results
s_utils.clean_dirs('simulation/summaries/' + name)
s_utils.clean_dirs('simulation/summaries/compressed/' + name)

# create and run simulation
sim = Simulator(
  model=nn_mnist_model,
  learning_rate=learning_rate,
  noise_list=noise_list,
  noise_type='betas',
  batch_size=batch_size,
  n_epochs=n_epochs,
  name=name,
  swap_step=swap_step,
  test_step=test_step,
  separation_ratio=separation_ratio,
  description=description,
  burn_in_period=burn_in_period,
  loss_func_name='cross_entropy'
  )

sim.train(train_data=train_data, train_labels=train_labels,
  test_data=test_data, test_labels=test_labels, 
  validation_data=valid_data, validation_labels=valid_labels)


# plot results
se = SummaryExtractor(name)
se.print_report()
```

### 2. Simulate mnist dataset with multilayer perceptron using dropout and RMSPropOptimizer

```python
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from simulation.simulator import Simulator
from simulation.summary_extractor2 import SummaryExtractor
import simulation.simulator_utils as s_utils
from simulation.models.mnist_models import nn_mnist_model_dropout

MNIST_DATAPATH = 'simulation/data/mnist/'

mnist = input_data.read_data_sets(MNIST_DATAPATH)
train_data = mnist.train.images
train_labels = mnist.train.labels
test_data = mnist.test.images
test_labels = mnist.test.labels
valid_data = mnist.validation.images
valid_labels = mnist.validation.labels

n_replicas = 8

# set simulation parameters
model_func = nn_mnist_model_dropout
learning_rate = 0.01
noise_list = [1-x for x in np.geomspace(start=0.01, stop=0.99, num=n_replicas)]
noise_type = 'dropout_rmsprop'
batch_size = 200
n_epochs = 20
name = 'rmsprop_simulation' # simulation name
test_step = 400 # 1 step==batch_size
swap_step = 300
burn_in_period = 400
loss_func_name = 'cross_entropy'
description = 'RMSProp with dropout.'
proba_coeff = 250
rmsprop_decay = 0.9
rmsprop_momentum = 0.001
rmsprop_epsilon=1e-6

# make sure that there are no directories that were previously created with same name
# otherwise, there will be problems extracting simulated results
s_utils.clean_dirs('simulation/summaries/' + name)
s_utils.clean_dirs('simulation/summaries/compressed/' + name)

# create and run simulation

sim = Simulator(
  model=model_func,
  learning_rate=learning_rate,
  noise_list=noise_list,
  noise_type='dropout_rmsprop',
  batch_size=batch_size,
  n_epochs=n_epochs,
  test_step=test_step,
  name=name,
  swap_step=swap_step,
  burn_in_period=burn_in_period,
  loss_func_name='cross_entropy',
  description=description,
  proba_coeff=proba_coeff,
  rmsprop_decay=rmsprop_decay,
  rmsprop_epsilon=rmsprop_epsilon,
  rmsprop_momentum=rmsprop_momentum
  )

sim.train(train_data=train_data, train_labels=train_labels,
  test_data=test_data, test_labels=test_labels, 
  validation_data=valid_data, validation_labels=valid_labels)


# plot results
se = SummaryExtractor(name)
se.print_report()

```
