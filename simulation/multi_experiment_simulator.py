from time import time

from simulation.simulator_utils import generate_experiment_name
from simulation.simulator import Simulator

class MultiExperimentSimulator(object):
	def __init__(self, architecture, tuning_parameter_vals, n_replicas, beta_0,
		learning_rate, architecture_name, surface_view, data, noise_type,
		n_simulations=10, batch_size=50, n_epochs=25, swap_attempt_step=400, 
		dataset_name='mnist', tuning_param_name='tempfactor', swap_proba='boltzmann',
		optimizer_name='PTLD', loss_func_name='crossentropy', do_swaps=True, 
		start_experiments_at=0, burn_in_period=None,
		temp_factor=None, description='v3', verbose=True):

		self.n_experiments = len(tuning_parameter_vals)

		self.experiment_name = generate_experiment_name(
			architecture_name=architecture_name, dataset=dataset_name, 
			temp_ratio=tuning_param_name.replace('_', ''), optimizer=optimizer_name,
			do_swaps=do_swaps, swap_proba='boltzmann', n_replicas=n_replicas,
			surface_view=surface_view, beta_0=beta_0, 
			loss_func_name=loss_func_name.replace('_',''), 
			swap_attempt_step=swap_attempt_step, burn_in_period=burn_in_period, 
			learning_rate=learning_rate, n_epochs=n_epochs)

		self.architecture = architecture
		self.tuning_parameter_vals = tuning_parameter_vals
		self.n_replicas = n_replicas
		self.beta_0 = beta_0
		self.learning_rate = learning_rate
		self.architecture_name = architecture_name
		self.surface_view = surface_view
		self.n_simulations = n_simulations
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.swap_attempt_step = swap_attempt_step
		self.dataset_name = dataset_name
		self.tuning_param_name = tuning_param_name
		self.swap_proba = swap_proba
		self.optimizer_name = optimizer_name
		self.loss_func_name = loss_func_name
		self.do_swaps = do_swaps,
		self.description = description
		self.verbose = verbose
		self.data = data
		self.noise_type = noise_type
		self.start_experiments_at = start_experiments_at
		self.burn_in_period = burn_in_period
		self.loss_func_name = loss_func_name
		self.temp_factor = temp_factor



	def run_all(self, beta_0_list=None):
		if self.tuning_param_name.replace('_', '') == 'tempfactor':
			self._run_tempfactor_experiments()
		elif self.tuning_param_name == 'beta0':
			self._run_accept_experiments(beta_0_list)

	def _run_accept_experiments(self, beta0_list):
		separation_ratio = self.temp_factor
		if beta0_list is None:
			raise ValueError("beta0_list is None, but must contain beta_0 vals.")

		timer = Timer()
		name = self.experiment_name
		
		for exp in range(self.start_experiments_at, len(beta0_list)):
			beta0 = beta0_list[exp]
			noise_list = [beta0, beta0*separation_ratio]
			name = self.experiment_name + '_' + str(exp)
			if self.verbose: print('experiment:', exp+1, '/', len(beta0_list), 
				', separation_ratio:', separation_ratio)
			if self.verbose: print(noise_list)

			sim = Simulator(self.architecture, self.learning_rate,
				noise_list, self.noise_type, batch_size=self.batch_size,
				n_epochs=self.n_epochs, name=name, n_simulations=self.n_simulations,
				swap_attempt_step=self.swap_attempt_step,
				temp_factor=separation_ratio, 
				tuning_parameter_name=self.tuning_param_name,
				surface_view=self.surface_view, burn_in_period=self.burn_in_period,
				loss_func_name=self.loss_func_name, description=self.description)

			sim.train_n_times(sim.train_PTLD, train_data=self.data['train_data'],
				train_labels=self.data['train_labels'], 
				test_data=self.data['test_data'],
				test_labels=self.data['test_labels'], 
				validation_data=self.data['valid_data'], 
				validation_labels=self.data['valid_labels']
				)
			if self.verbose: print()
			if self.verbose: print('time took:', timer.elapsed_time(), 'min')

			#exp += 1


		

	def _run_tempfactor_experiments(self):
		timer = Timer()
		for exp in range(self.start_experiments_at, self.n_experiments):
			temp_factor = self.tuning_parameter_vals[exp]
			name = self.experiment_name + '_' + str(exp)
			if self.verbose: print('experiment:', exp+1, '/', self.n_experiments, 
				', param_val:', temp_factor)
			if self.verbose: print(name)

			noise_list = [self.beta_0*temp_factor**i for i in range(self.n_replicas)]
			if self.verbose: print(noise_list)


			sim = Simulator(self.architecture, self.learning_rate, 
				noise_list, self.noise_type,
				batch_size=self.batch_size, n_epochs=self.n_epochs,
				name=name, n_simulations=self.n_simulations, 
				swap_attempt_step=self.swap_attempt_step, 
				temp_factor=temp_factor,
				tuning_parameter_name=self.tuning_param_name,
				surface_view=self.surface_view, burn_in_period=self.burn_in_period,
				loss_func_name=self.loss_func_name, description=self.description)

			sim.train_n_times(sim.train_PTLD, train_data=self.data['train_data'],
				train_labels=self.data['train_labels'], 
				test_data=self.data['test_data'],
				test_labels=self.data['test_labels'], 
				validation_data=self.data['valid_data'], 
				validation_labels=self.data['valid_labels']
				)
			if self.verbose: print()
			if self.verbose: print('time took:', timer.elapsed_time(), 'min')









class Timer(object):
    def __init__(self):
        self.start_time = time()
    def start_timer(self):
        self.start_time = time()
    def elapsed_time(self):
        res = int((time() - self.start_time) / 60.0)
        self.start_time = time()
        return res