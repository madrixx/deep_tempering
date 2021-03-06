"""A class that performs simulations."""
import sys
import os
import gc
import json

import tensorflow as tf

from simulation.simulation_builder.graph_builder import GraphBuilder
from simulation.simulation_builder.summary import Dir
from simulation import simulator_utils as s_utils

class Simulator: # pylint: disable=too-many-instance-attributes
  """Performs single/multiple simulation for calculating averages.

  This class defines the API for performing simulations. This class
  trains models (possibly multiple times), while class GraphBuilder
  creates dataflow graphs with duplicated replicas. More functions
  can be added to train models in different setups.

  ### Usage

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
  separation_ratio = 1.21

  # set simulation parameters
  model_func = nn_mnist_model_dropout
  learning_rate = 0.01
  noise_list = [1/separation_ratio**i for i in range(n_replicas)]
  noise_type = 'dropout_rmsprop'
  batch_size = 200
  n_epochs = 50
  name = 'test_simulation' # simulation name
  test_step = 300 # 1 step==batch_size
  swap_step = 300
  burn_in_period = 400
  loss_func_name = 'cross_entropy'
  description = 'RMSProp with dropout.'
  proba_coeff = 250
  rmsprop_decay = 0.9
  rmsprop_momentum = 0.001
  rmsprop_epsilon=1e-6

  # make sure that there are no directories that were previously created
  # with same name, otherwise, there will be problems extracting 
  # simulated results
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
  se.print_report(mixing_log_y=separation_ratio)
  ```
  """

  def __init__(self, # pylint: disable=too-many-arguments, too-many-locals
               model,
               learning_rate,
               noise_list,
               noise_type,
               batch_size,
               n_epochs,
               name,
               n_simulations=1,
               summary_type=None,
               test_step=500,
               swap_step=500,
               separation_ratio=None,
               tuning_parameter_name=None,
               burn_in_period=None,
               loss_func_name='cross_entropy',
               proba_coeff=1.0,
               surface_view='energy',
               description=None,
               rmsprop_decay=0.9,
               rmsprop_momentum=0.001,
               rmsprop_epsilon=1e-6):
    """Creates a new simulator object.

    Args:
      model: A function that creates inference model (e.g.
        see simulation.models.nn_mnist_model)
      learning_rate: Learning rate for optimizer
      noise_list: A list (not np.array!) for noise/temperatures/dropout
        values. In case of dropout (dropout_rmsprop, dropout_gd), noise_list
        represents the values of KEEPING the neurons, and NOT the probability
        of excluding the neurons.
      noise_type: A string specifying the noise type and optimizer to apply.
        Possible values could be seen at
        simulation.simulation_builder.graph_builder.GraphBuilder.__noise_types
      batch_size: Batch Size
      n_epochs: Number of epochs for each simulation
      name: The name of the simulation. Specifies the a folder name
        through which a summary files can be later accessed.
      n_simulatins: Number of simulation to run.
      summary_type: Specifies what summary types to store. Detailed
      possibilities could be seen in
        simulation.simulation_builder.graph_builder.Summary.
        Default is None (if None stores all summaries)
      test_step: An integer specifing an interval of steps to perform until
        running a test dataset (1 step equals batch_size)
      swap_step: An integer specifying an interval to perform until
        attempting to swap between replicas based on validation dataset.
      separation_ratio: A separation ratio between two adjacent temperatures.
        This value is not important for simulation because the
        noise_list already contains the separated values. This value is
        (as well as some others) are stored in the simulation
        description file (this file is created by _log_params()
        function).
      tuning_parameter_name: As the separation_ratio value, this argument is
        also not important for simulation. It is stored in the description
        file as well.
      burn_in_period: A number of steps until the swaps start to be
        proposed.
      loss_func_name: A function which we want to optimize. Currently,
        only cross_entropy and stun (stochastic tunneling) are
        supported.
      proba_coeff: The coeffecient is used in calculation of probability
        of swaps. Specifically, we have
        P(accept_swap) = exp(proba_coeff*(beta_1-beta_2)(E_1-E_2))
      surface_view: 'information' or 'energy'. See
        GraphBuilder.swap_replicas() for detailed explanation.
      description: A custom string that is stored in the description file.
      rmsprop_decay: Used in
        simulation.simulation_builder.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other noise_types.
      rmsprop_momentum: Used in
        simulation.simulation_builder.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other noise_types.
      rmsprop_epsilon: Used in
        simulation.simulation_builder.optimizers.RMSPropOptimizer
        for noise type 'dropout_rmsprop'. This value is ignored for
        other noise_types.

    """
    self.model = model
    self.learning_rate = learning_rate
    self.noise_type = noise_type
    self.noise_list = noise_list
    self.summary_type = summary_type
    self.learning_rate = learning_rate
    self.name = name
    self.n_simulations = n_simulations
    self.burn_in_period = burn_in_period
    self.loss_func_name = loss_func_name
    self.proba_coeff = proba_coeff
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.test_step = test_step
    self.swap_step = swap_step
    self.separation_ratio = separation_ratio
    self.tuning_parameter_name = tuning_parameter_name
    self.surface_view = surface_view
    tf.logging.set_verbosity(tf.logging.ERROR)
    self.delim = "\\" if 'win' in sys.platform else "/"
    self._dir = Dir(name)
    self.description = description
    self._log_params()

    self.rmsprop_decay = rmsprop_decay
    self.rmsprop_momentum = rmsprop_momentum
    self.rmsprop_epsilon = rmsprop_epsilon

    if n_simulations == 1:
      self.graph = GraphBuilder(self.model,
                                self.learning_rate,
                                self.noise_list,
                                self.name,
                                self.noise_type,
                                self.summary_type,
                                simulation_num=0,
                                surface_view=self.surface_view,
                                loss_func_name=self.loss_func_name,
                                proba_coeff=self.proba_coeff,
                                rmsprop_decay=self.rmsprop_decay,
                                rmsprop_momentum=self.rmsprop_momentum,
                                rmsprop_epsilon=self.rmsprop_epsilon)

  def train_n_times(self, train_func, **kwargs):
    """Trains `n_simulations` times using the same setup.

    During training, the summary values are stored using native
    tensorflow Summary class, but later copied (see
    simulation.simulator_utils.extract_and_remove_simulation())
    to a pickle object, since tensorflow summary objects
    take too many space on disk. These pickle summaries are
    accessed through SummaryExtractor class defined in
    simulation.summary_extractor2 file. After they copied, the
    original tensorflow summary files are deleted and connot be
    accessed by tensorboard.

    Args:
      train_func: A function that performs training (e.g.
        train_PTLD)
      kwargs: train_dataset, train_labels, test_data, test_labels,
        validation_data, validation_labels
    """

    sim_names = []

    for i in range(self.n_simulations):
      self.graph = GraphBuilder(self.model,
                                self.learning_rate,
                                self.noise_list,
                                self.name,
                                self.noise_type,
                                self.summary_type,
                                simulation_num=i,
                                surface_view=self.surface_view,
                                loss_func_name=self.loss_func_name,
                                proba_coeff=self.proba_coeff,
                                rmsprop_decay=self.rmsprop_decay,
                                rmsprop_momentum=self.rmsprop_momentum,
                                rmsprop_epsilon=self.rmsprop_epsilon)

      sim_names.append(self.graph._summary.dir.name) # pylint: disable=protected-access

      train_func(kwargs)

      gc.collect()

    _ = [s_utils.extract_and_remove_simulation(n) for n in sim_names]

  def explore_heat_capacity(self, train_func, betas, **kwargs):
    """Used for exploring heat capacity function."""

    sim_names = []
    i = 0
    separation_ratio = betas[1]/betas[0]
    for beta_0 in betas:
      for i in range(self.n_simulations):
        noise_list = [beta_0, separation_ratio*beta_0]
        self.graph = GraphBuilder(self.model,
                                  self.learning_rate,
                                  noise_list,
                                  self.name,
                                  self.noise_type,
                                  self.summary_type,
                                  simulation_num=i,
                                  surface_view=self.surface_view,
                                  loss_func_name=self.loss_func_name,
                                  proba_coeff=self.proba_coeff,
                                  rmsprop_decay=self.rmsprop_decay,
                                  rmsprop_momentum=self.rmsprop_momentum,
                                  rmsprop_epsilon=self.rmsprop_epsilon)

        sim_names.append(self.graph._summary.dir.name) # pylint: disable=protected-access

        train_func(kwargs)

        gc.collect()

        i += 1

  def train_PTLD(self, kwargs): # pylint: disable=too-many-locals, invalid-name
    """Trains and swaps between replicas"""

    try:
      g = self.graph # pylint: disable=invalid-name
    except AttributeError as err:
      if not err.args:
        err.args = ('',)

      err.args = (err.args
                  + ("The GraphBuilder object is not initialized.",))
      raise

    try:
      train_data = kwargs.get('train_data', None)
      train_labels = kwargs.get('train_labels', None)
      test_data = kwargs.get('test_data', None)
      test_labels = kwargs.get('test_labels', None)
      valid_data = kwargs.get('validation_data', None)
      valid_labels = kwargs.get('validation_labels', None)
      test_feed_dict = g.create_feed_dict(test_data, test_labels,
                                          dataset_type='test')

      # create iterator for train dataset
      with g.get_tf_graph().as_default(): # pylint: disable=not-context-manager
        data = tf.data.Dataset.from_tensor_slices({
            'X':train_data,
            'y':train_labels
            }).batch(self.batch_size)
        iterator = data.make_initializable_iterator()

    except: # pylint: disable=try-except-raise
      raise

    with g.get_tf_graph().as_default(): # pylint: disable=not-context-manager

      step = 0

      with tf.Session() as sess:

        sess.run(iterator.initializer)
        sess.run(g.variable_initializer)
        next_batch = iterator.get_next()

        for epoch in range(self.n_epochs):

          while True:
            try:
              step += 1

              ### train ###
              batch = sess.run(next_batch)
              feed_dict = g.create_feed_dict(batch['X'], batch['y'])
              evaluated = sess.run(g.get_train_ops(),
                                   feed_dict=feed_dict)

              g.add_summary(evaluated, step=step)

              ### test ###
              if step % self.test_step == 0 or step == 1:
                evaluated = sess.run(g.get_train_ops('test'),
                                     feed_dict=test_feed_dict)
                g.add_summary(evaluated, step, dataset_type='test')
                loss = g.extract_evaluated_tensors(evaluated,
                                                   self.loss_func_name)

                self.print_log(loss, epoch, g.swap_accept_ratio,
                               g.latest_accept_proba, step)

              ### validation + swaps ###
              if step % self.swap_step == 0:

                valid_feed_dict = g.create_feed_dict(valid_data,
                                                     valid_labels,
                                                     'validation')
                evaluated = sess.run(g.get_train_ops('validation'),
                                     feed_dict=valid_feed_dict)
                g.add_summary(evaluated, step,
                              dataset_type='validation')
                if step > self.burn_in_period:

                  g.swap_replicas(evaluated)

                g._summary.flush_summary_writer() # pylint: disable=protected-access

            except tf.errors.OutOfRangeError:
              sess.run(iterator.initializer)
              break

        g._summary.close_summary_writer() # pylint: disable=protected-access

  def train(self, **kwargs):
    """Trains model single time.

    This function is basically the same as train_PTLD"""
    self.train_PTLD(kwargs)


  def _log_params(self):
    """Creates a description file."""
    dirpath = self._dir.log_dir
    filepath = os.path.join(dirpath, 'description.json')
    if not os.path.exists(dirpath):
      os.makedirs(dirpath)
    _log = {
        'name':self.name,
        'noise_type': self.noise_type,
        'noise_list': self.noise_list,
        'n_replicas': len(self.noise_list),
        'learning_rate':self.learning_rate,
        'n_epochs':self.n_epochs,
        'batch_size':self.batch_size,
        'swap_step': self.swap_step,
        'separation_ratio': self.separation_ratio,
        'n_simulations': self.n_simulations,
        'tuning_parameter_name':self.tuning_parameter_name,
        'surface_view':self.surface_view,
        'description':self.description,
        'burn_in_period':self.burn_in_period,
        'proba_coeff':self.proba_coeff
    }
    with open(filepath, 'w') as file:
      json.dump(_log, file, indent=4)

  def print_log(self, # pylint: disable=too-many-arguments
                loss,
                epoch,
                swap_accept_ratio,
                latest_accept_proba,
                step):
    """Helper for logs during training."""
    buff = 'epoch:' + str(epoch) + ', step:' + str(step) + ', '
    buff = buff + ','.join([str(l) for l in loss]) + ', '
    buff = buff + 'accept_ratio:' + str(swap_accept_ratio)
    buff = buff + ', proba:' + str(latest_accept_proba) + '         '
    self.stdout_write(buff)

  def stdout_write(self, buff): # pylint: disable=no-self-use
    """Writes to stdout buffer with beginning of the line character."""
    sys.stdout.write('\r' + buff)
    sys.stdout.flush()
