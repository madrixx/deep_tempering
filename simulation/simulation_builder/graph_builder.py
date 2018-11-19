"""Classes and functions that manipulate tensorflow graphs."""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import random

import tensorflow as tf
import numpy as np

from simulation.simulation_builder.graph_duplicator import copy_and_duplicate
from simulation.simulation_builder.optimizers import GDOptimizer
from simulation.simulation_builder.optimizers import NormalNoiseGDOptimizer
from simulation.simulation_builder.optimizers import GDLDOptimizer
from simulation.simulation_builder.optimizers import RMSPropOptimizer
from simulation.simulation_builder.summary import Summary
from simulation.simulator_exceptions import InvalidDatasetTypeError
from simulation.simulator_exceptions import InvalidArchitectureFuncError
from simulation.simulator_exceptions import InvalidLossFuncError
from simulation.simulator_exceptions import InvalidNoiseTypeError

class GraphBuilder: # pylint: disable=too-many-instance-attributes
  """Defines a dataflow graph with duplicated ensembles.

  This object stores all copies of the systems at different
  temperatures and provides an API for performing the
  exchanges between two ensembles and storing the summary
  values. It is used to train models in the
  Parallel Tempering framework.
  """

  def __init__(self, model, learning_rate, noise_list, name, # pylint: disable=too-many-locals, too-many-branches, too-many-arguments, too-many-statements
               noise_type='random_normal', summary_type=None,
               simulation_num=None, surface_view='energy',
               loss_func_name='cross_entropy', proba_coeff=1.0,
               rmsprop_decay=0.9, rmsprop_momentum=0.0,
               rmsprop_epsilon=1e-10):
    """Creates a GraphBuilder object.

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
      name: A name of the simulation. Specifies the a folder name through which
          a summary files can be later accessed.
      summary_type: Specifies what summary types to store. Detailed
        possibilities could be seen in
        simulation.simulation_builder.graph_builder.Summary.
        Default is None (if None stores all summaries)
      simulation_num: Specifies the simulation number that is currently in
        progress. It is relevant when we simulating the same simulation
        multiple times. In this case, each simulation is stored in
        the location: 'summaries/name/simulation_num'.
      surface_view: 'information' or 'energy'. See
        GraphBuilder.swap_replicas() for detailed explanation.
      loss_func_name: A function which we want to optimize. Currently,
        only cross_entropy and stun (stochastic tunneling) are
        supported.
      proba_coeff: The coeffecient is used in calculation of probability
        of swaps. Specifically, we have
        P(accept_swap) = exp(proba_coeff*(beta_1-beta_2)(E_1-E_2))
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

    self.__noise_types = ['random_normal',
                          'betas',
                          'dropout',
                          'dropout_rmsprop',
                          'dropout_gd'] # possible noise types

    if not isinstance(noise_list, list) or not noise_list:
      raise ValueError("Invalid `noise_list`. Should be non-empty python list.")

    self._model = model
    self._learning_rate = learning_rate
    self._n_replicas = len(noise_list)
    self._noise_type = noise_type
    self._name = name
    self._surface_view = surface_view
    self._graph = tf.Graph()
    self._noise_list = sorted(noise_list)
    self._summary_type = summary_type
    self._simulation_num = '' if simulation_num is None else str(simulation_num)
    self._loss_func_name = loss_func_name
    self._proba_coeff = proba_coeff

    # create graph with duplicated ensembles based on the provided
    # model function and noise type
    res = []
    try:
      res = self._model(tf.Graph())
      if (len(res) == 3 and
          noise_type in ['random_normal', 'betas']):
        X, y, logits = res # pylint: disable=invalid-name
        self.X, self.y, logits_list = copy_and_duplicate(X, y, logits, # pylint: disable=invalid-name
                                                         self._n_replicas,
                                                         self._graph)

        # _noise_plcholders will be used to store noise vals for summaries
        with self._graph.as_default(): # pylint: disable=not-context-manager
          self._noise_plcholders = {i:tf.placeholder(tf.float32, shape=[])
                                    for i in range(self._n_replicas)}

        # curr_noise_dict stores {replica_id:current noise stddev VALUE}
        self._curr_noise_dict = {i:n for i, n in enumerate(self._noise_list)}

      elif (len(res) == 4
            and noise_type in ['dropout', 'dropout_rmsprop', 'dropout_gd']):
        X, y, prob_placeholder, logits = res # pylint: disable=invalid-name
        self.X, self.y, probs, logits_list = copy_and_duplicate(
            X, y, logits, self._n_replicas, self._graph,
            prob_placeholder)

        # _noise_plcholders stores dropout plcholders: {replica_id:plcholder}
        # it is used also to store summaries
        self._noise_plcholders = {i:p for i, p in enumerate(probs)}

        # in case of noise_type == dropout, _curr_noise_dict stores
        # probabilities for keeping optimization parameters
        # (W's and b's): {replica_id:keep_proba}
        self._curr_noise_dict = {
            i:n
            for i, n in enumerate(sorted(self._noise_list, reverse=True))}

      elif noise_type not in self.__noise_types:
        raise InvalidNoiseTypeError(noise_type, self.__noise_types)

      else:
        raise InvalidArchitectureFuncError(len(res), self._noise_type)

    except:
      print("Problem with model function.")
      raise

    # from here, whole net that goes after logits is created
    self._cross_entropy_loss_dict = {}
    self._zero_one_loss_dict = {}
    self._stun_loss_dict = {}
    self._optimizer_dict = {}

    # special vals for summary:
    self.swap_accept_ratio = 0.0
    self.n_swap_attempts = 0
    self.latest_accept_proba = 1.0
    self.latest_swapped_pair = -1
    self.replica_swap_ratio = {i:0.0 for i in range(self._n_replicas)}
    self.ordered_swap_ratio = {i:0.0 for i in range(self._n_replicas)}
    self.replica_n_swap_attempts = {i:0 for i in range(self._n_replicas)}
    self.ordered_n_swap_attempts = {i:0 for i in range(self._n_replicas)}

    # set loss function and optimizer
    with self._graph.as_default(): # pylint: disable=not-context-manager
      for i in range(self._n_replicas):

        with tf.name_scope('Metrics' + str(i)):

          self._cross_entropy_loss_dict[i] = self._cross_entropy_loss(
              self.y,
              logits_list[i])

          self._zero_one_loss_dict[i] = self._zero_one_loss(
              self.y, logits_list[i])

          self._stun_loss_dict[i] = self._stun_loss(
              self._cross_entropy_loss_dict[i])

        with tf.name_scope('Optimizer_' + str(i)):

          if noise_type.lower() == 'random_normal':
            optimizer = NormalNoiseGDOptimizer(
                self._learning_rate, i, self._noise_list)

          elif noise_type.lower() == 'betas':
            optimizer = GDLDOptimizer(
                self._learning_rate, i, self._noise_list)

          elif noise_type.lower() == 'dropout_rmsprop':
            optimizer = RMSPropOptimizer(
                self._learning_rate, i, self._noise_list,
                decay=rmsprop_decay, momentum=rmsprop_momentum,
                epsilon=rmsprop_epsilon)

          elif noise_type.lower() in ['dropout_gd', 'dropout']:
            optimizer = GDOptimizer(
                self._learning_rate, i, self._noise_list)

          else:
            raise InvalidNoiseTypeError(noise_type, self.__noise_types)

          self._optimizer_dict[i] = optimizer

          if self._loss_func_name.replace('_', '') == 'crossentropy':
            optimizer.minimize(self._cross_entropy_loss_dict[i])

          elif self._loss_func_name == 'zero_one_loss':
            optimizer.minimize(self._zero_one_loss_dict[i])

          elif self._loss_func_name == 'stun':
            optimizer.minimize(self._stun_loss_dict[i])

          else:
            raise ValueError('Invalid loss function name.',
                             'Available functions are: \
                             cross_entropy/zero_one_loss/stun,',
                             'But given:', self._loss_func_name)

      # The Summary object helps to store summaries
      self._summary = Summary(
          self._graph, self._n_replicas, self._name,
          self._cross_entropy_loss_dict, self._zero_one_loss_dict,
          self._stun_loss_dict, self._noise_list,
          self._noise_plcholders, simulation_num,
          self._optimizer_dict, summary_type=self._summary_type)

      self.variable_initializer = tf.global_variables_initializer()


  def create_feed_dict(self, X_batch, y_batch, dataset_type='train'): # pylint: disable=invalid-name
    """Creates feed_dict for session run.

    Args:
      X_batch: input X training batch
      y_batch: input y training batch
      dataset_type: 'train', 'test' or 'validation'

    Returns:
      A dictionary to feed into session's run.
      If dataset_type=='train', adds to feed_dict placeholders
      to store noise (for summary).
      If dataset_type=='validation'/'test', then doesn't add
      this placeholder (since we don't add noise for test or
      validation).
      If noise_type is 'dropout' and dataset_type is 'train',
      adds values for keeping parameters during optimization
      (placeholders keep_prob for each replica).

    Raises:
      InvalidDatasetTypeError: if incorrect `dataset_type`.
    """

    feed_dict = {self.X:X_batch, self.y:y_batch}

    if dataset_type == 'test':
      dict_ = {self._noise_plcholders[i]:1.0
               for i in range(self._n_replicas)}

    elif dataset_type == 'validation':
      dict_ = {self._noise_plcholders[i]:1.0
               for i in range(self._n_replicas)}
      dict_.update({
          self._summary.swap_accept_ratio_plcholder:self.swap_accept_ratio,
          self._summary.accept_proba_plcholder:self.latest_accept_proba,
          self._summary.swap_replica_pair_plcholder:self.latest_swapped_pair})
      temp_dict1 = {
          self._summary.replica_accept_ratio_plcholders[i]:self.replica_swap_ratio[i] # pylint: disable=line-too-long
          for i in range(self._n_replicas)}
      temp_dict2 = {
          self._summary.ordered_accept_ratio_plcholders[i]:self.ordered_swap_ratio[i] # pylint: disable=line-too-long
          for i in range(self._n_replicas)}

      dict_.update(temp_dict1)
      dict_.update(temp_dict2)

    elif dataset_type == 'train':
      dict_ = {self._noise_plcholders[i]:self._curr_noise_dict[i]
               for i in range(self._n_replicas)}

    else:
      raise InvalidDatasetTypeError()

    feed_dict.update(dict_)

    return feed_dict

  def get_train_ops(self, dataset_type='train'):
    """Returns train ops for session's run.

    The returned list should be used as:
    # evaluated = sess.run(get_train_ops(), feed_dict=...)

    Args:
      dataset_type: One of 'train'/'test'/'validation'

    Returns:
      train_ops for session run.

    Raises:
      InvalidDatasetTypeError if incorrect dataset_type.
    """

    loss = [self._cross_entropy_loss_dict[i]
            for i in range(self._n_replicas)]
    zero_one_loss = [self._zero_one_loss_dict[i]
                     for i in range(self._n_replicas)]
    stun_loss = [self._stun_loss_dict[i]
                 for i in range(self._n_replicas)]
    summary = self._summary.get_summary_ops(dataset_type)

    if dataset_type in ['test', 'validation']: # pylint: disable=no-else-return
      return loss + zero_one_loss + stun_loss + summary

    elif dataset_type == 'train':
      train_op = [self._optimizer_dict[i].get_train_op()
                  for i in range(self._n_replicas)]

      return loss + zero_one_loss + stun_loss + summary + train_op
    else:
      raise InvalidDatasetTypeError()

  def add_summary(self, evaluated, step, dataset_type='train'):
    """Adds summary using Summary class object.

    ### Usage:

    ```python
    # Suppose g is a GraphBuilder object and step is the value
    # that is incremented after each mini-batch.
    # Evaluate train data and store computed values:
    evaluated = sess.run(g.get_train_ops(dataset_type='train'))
    g.add_summary(evaluated, step, dataset_type='train')
    ```

    Args:
      evaluated: A list returned by `sess.run(get_train_ops())`
      step: A step for tf.summary.FileWriter.add_summary(). A
        value that is incremented after each mini-batch.
      dataset_type: One of 'train'/'test'/'validation'

    """
    summs = self.extract_evaluated_tensors(evaluated, 'summary')
    self._summary.add_summary(summs, step, dataset_type)

  def extract_evaluated_tensors(self, evaluated, tensor_type):
    """Extracts tensors from a list of tensors evaluated by tf.Session.

    ### Usage:

    ```python
    # Suppose g is a GraphBuilder object.
    # Run and print cross entropy loss vals for each replica for test data:
    evaluated = sess.run(g.get_train_ops(dataset_type='test'))
    loss_vals = g.extract_evaluated_tensors(evaluated,
      tensor_type='cross_entropy')
    print(loss_vals)
    ```

    Args:
      evaluated: A list returned by sess.run(get_train_ops())
      tensor_type: One of 'cross_entropy'/'zero_one_loss'/'stun'/'summary'

    Returns:
      A list of specified (by `tensor_type`) tensors.

    Raises:
      InvlaidLossFuncError: Incorrect `tensor_type` value.
      """

    if tensor_type == 'cross_entropy': # pylint: disable=no-else-return
      return evaluated[:self._n_replicas]

    elif tensor_type == 'zero_one_loss':
      return evaluated[self._n_replicas:2*self._n_replicas]

    elif tensor_type == 'stun':
      return evaluated[2*self._n_replicas:3*self._n_replicas]

    elif tensor_type == 'summary':
      end_mult = (5 if self._summary_type is None else 4)
      if len(evaluated) % self._n_replicas == 0: # pylint: disable=no-else-return
        return evaluated[3*self._n_replicas:end_mult*self._n_replicas]
      else:
        # special summary case
        return evaluated[3*self._n_replicas:end_mult*self._n_replicas + 1]
    else:
      raise InvalidLossFuncError()

  def swap_replicas(self, evaluated): # pylint: disable=too-many-locals
    """Swaps between replicas.

    Swaps according to:
      1. Uniformly randomly select a pair of adjacent temperatures
        1/beta_i and 1/beta_i+1, for which swap move is proposed.
      2. Swap according to:
          If surface_view is 'information', accept with probability:
            min{1, exp((beta_i-beta_i+1)*(loss_i/beta_i-loss_i+1/beta_i+1)}
          if surface_view is 'energy', accept with probability:
            min{1, exp((beta_i-beta_i+1)*(loss_i-loss_i+1)}
      3. Update the acceptance ratio for the proposed swap.

    Args:
      evaluated: a list returned by sess.run(get_train_ops())

    Raises:
      ValueError: if invalid `surface_view`.
    """
    random_pair = random.choice(range(self._n_replicas - 1)) # pair number

    beta = [self._curr_noise_dict[x] for x in range(self._n_replicas)]
    beta_id = [(b, i) for i, b in enumerate(beta)]
    beta_id.sort(key=lambda x: x[0], reverse=True)

    i = beta_id[random_pair][1]
    j = beta_id[random_pair+1][1]

    loss_list = self.extract_evaluated_tensors(evaluated, self._loss_func_name)

    sorted_losses = sorted(loss_list)

    sorted_i = sorted_losses.index(loss_list[i])
    sorted_j = sorted_losses.index(loss_list[j])

    self.n_swap_attempts += 1
    self.replica_n_swap_attempts[i] += 1
    self.replica_n_swap_attempts[j] += 1
    self.ordered_n_swap_attempts[sorted_i] += 1
    self.ordered_n_swap_attempts[sorted_j] += 1

    if self._surface_view == 'information' or self._surface_view == 'info':
      l1, l2 = loss_list[i]/beta[i], loss_list[j]/beta[j] # pylint: disable=invalid-name
    elif self._surface_view == 'energy':
      l1, l2 = loss_list[i], loss_list[j] # pylint: disable=invalid-name
    else:
      raise ValueError('Invalid surface view.')

    accept_proba = np.exp(self._proba_coeff*(l1-l2)*(beta[i] - beta[j]))
    self.latest_accept_proba = accept_proba


    if np.random.uniform() < accept_proba:
      self._curr_noise_dict[i] = beta[j]
      self._curr_noise_dict[j] = beta[i]

      self._optimizer_dict[i].set_train_route(j)
      self._optimizer_dict[j].set_train_route(i)

      self.swap_accept_ratio = (((self.n_swap_attempts-1)/self.n_swap_attempts)
                                * self.swap_accept_ratio
                                + (1/self.n_swap_attempts))
      self.latest_swapped_pair = i

      for x in [i, j]: # pylint: disable=invalid-name
        n = self.replica_n_swap_attempts[x] # pylint: disable=invalid-name
        ratio = self.replica_swap_ratio[x]
        self.replica_swap_ratio[x] = ((n - 1)/n)*ratio + (1/n)
      for x in [sorted_j, sorted_i]: # pylint: disable=invalid-name
        n = self.ordered_n_swap_attempts[x] # pylint: disable=invalid-name
        ratio = self.ordered_swap_ratio[x]
        self.ordered_swap_ratio[x] = ((n - 1)/n)*ratio + (1/n)
    else:

      self.latest_swapped_pair = -1
      self.swap_accept_ratio = (((self.n_swap_attempts-1)/self.n_swap_attempts)
                                * self.swap_accept_ratio)

      for x in [i, j]: # pylint: disable=invalid-name
        n = self.replica_n_swap_attempts[x] # pylint: disable=invalid-name
        ratio = self.replica_swap_ratio[x]
        self.replica_swap_ratio[x] = ((n - 1)/n)*ratio
      for x in [sorted_j, sorted_i]: # pylint: disable=invalid-name
        n = self.ordered_n_swap_attempts[x] # pylint: disable=invalid-name
        ratio = self.ordered_swap_ratio[x]
        self.ordered_swap_ratio[x] = ((n - 1)/n)*ratio

  def get_tf_graph(self):
    """Returns tensorflow graph."""
    return self._graph

  def _store_tf_graph(self, path):
    """Stores tensorflow graph."""
    tf.summary.FileWriter(path, self._graph).close()

  def _cross_entropy_loss(self, y, logits, clip_value_max=1000.0): # pylint: disable=invalid-name, no-self-use
    """Cross entropy on cpu"""
    with tf.name_scope('cross_entropy'):
      with tf.device('/cpu:0'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        if clip_value_max is not None:
          loss = tf.clip_by_value(loss, 0.0, clip_value_max)
    return loss

  def _zero_one_loss(self, y, logits): # pylint: disable=invalid-name, no-self-use
    """0-1 loss"""
    with tf.name_scope('zero_one_loss'):
      with tf.device('/cpu:0'):
        y_pred = tf.nn.in_top_k(predictions=logits, targets=y, k=1)

        zero_one_loss = 1.0 - tf.reduce_mean(
            tf.cast(x=y_pred, dtype=tf.float32),
            name='zero_one_loss')
    return zero_one_loss

  def _stun_loss(self, cross_entropy, gamma=1): # pylint: disable=no-self-use
    """Stochastic tunnelling loss."""
    with tf.name_scope('stun'):
      with tf.device('/cpu:0'):
        stun = 1 - tf.exp(-gamma*cross_entropy) # pylint: disable=no-member

    return stun
