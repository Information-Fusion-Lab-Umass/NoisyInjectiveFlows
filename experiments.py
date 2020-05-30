import shutil
import os
import numpy as np
import yaml
from model import MODEL_LIST
from optimizer import Optimizer
from jax import random
import pandas as pd
import pathlib
import pickle

class Experiment():
    """
    This class will represent everything inside a checkpoint
    """
    def __init__(self,
                 experiment_name,
                 x_shape,
                 data_loader,
                 quantize_level_bits,
                 checkpoint_iters,
                 start_it=None,
                 experiment_root='Experiments'):
        self.experiment_name     = experiment_name
        self.experiment_root     = os.path.join(experiment_root, experiment_name)
        self.data_loader         = data_loader
        self.x_shape             = x_shape
        self.quantize_level_bits = quantize_level_bits
        self.checkpoint_iters    = checkpoint_iters

        if(start_it < 0):
            start_it = None

        # Find the most recent iteration
        if(start_it is None):
            completed_iterations = []
            for root, dirs, _ in os.walk(self.experiment_root):
                for d in dirs:
                    try:
                        completed_iterations.append(int(d))
                    except:
                        pass
            completed_iterations = sorted(completed_iterations)
            start_it = None if len(completed_iterations) == 0 else completed_iterations[-1]

        self.current_iteration = start_it

        # These will be initialized later
        self.model       = None
        self.optimizer   = None
        self.plotter     = None

    # The directory structure should be:
    # - Experiments
    #   - <model_name>
    #     - Iteration number
    #         - meta_data
    #             - optimizer_meta_data.yaml
    #             - model_meta_data.yaml
    #         - checkpoint
    #             - opt_state.npz
    #             - model_state.npz
    #             - key.p
    #             - losses.txt
    #         - plots

    @property
    def experiment_iteration_path(self):
        path = os.path.join(self.experiment_root, str(self.current_iteration))
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return path

    @property
    def meta_data_path(self):
        path = os.path.join(self.experiment_iteration_path, 'meta_data')
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_meta_data_path(self):
        return os.path.join(self.meta_data_path, 'model_meta_data.yaml')

    @property
    def optimizer_meta_data_path(self):
        return os.path.join(self.meta_data_path, 'optimizer_meta_data.yaml')

    @property
    def checkpoint_path(self):
        path = os.path.join(self.experiment_iteration_path, 'checkpoint')
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        return path

    @property
    def optimizer_state_path(self):
        return os.path.join(self.checkpoint_path, 'opt_state.npz')

    @property
    def model_state_path(self):
        return os.path.join(self.checkpoint_path, 'model_state.npz')

    @property
    def keys_path(self):
        return os.path.join(self.checkpoint_path, 'keys.p')

    @property
    def losses_path(self):
        return os.path.join(self.checkpoint_path, 'losses.csv')

    #####################################################################

    def save_model_meta_data(self):
        meta_dict = self.model.gen_meta_data()

        with open(self.model_meta_data_path, 'w') as f:
            yaml.dump(meta_dict, f)

    def save_optimizer_meta_data(self):
        meta_dict = self.optimizer.gen_meta_data()

        with open(self.optimizer_meta_data_path, 'w') as f:
            yaml.dump(meta_dict, f)

    #####################################################################

    def load_model_instance_from_meta_data(self):
        # Load the meta data dictionary
        with open(self.model_meta_data_path) as f:
            meta = yaml.safe_load(f)

        # Return an instance of the model
        model_name = meta['model']
        ModelClass = MODEL_LIST[model_name]
        model = ModelClass.initialize_from_meta_data(meta)
        return model

    def load_optimizer_instance_from_meta_data(self):
        # Load the meta data dictionary
        with open(self.optimizer_meta_data_path) as f:
            meta = yaml.safe_load(f)

        # Return an instance of the optimizer
        optimizer = Optimizer.initialize_from_meta_data(meta)
        return optimizer

    #####################################################################

    def save_keys(self):
        keys = {'model_init_key': self.model_init_key,
                'opt_key':        self.opt_key}

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)

    def load_keys(self):
        with open(self.keys_path,'rb') as f:
            keys = pickle.load(f)

        self.model_init_key = keys['model_init_key']
        self.opt_key        = keys['opt_key']

    #####################################################################

    def save_losses(self, losses):

        series = pd.Series(losses)

        # Save the losses as a csv file
        series.to_csv(self.losses_path, mode='a', header=False)

    #####################################################################

    def create_experiment_from_meta_data(self, key, model_meta_data, optimizer_meta_data):
        """ Create an experiment from meta data """

        self.current_iteration = 0

        # Create the keys
        self.model_init_key, self.opt_key, data_dependent_init_key = random.split(key, 3)

        # Create the model
        model_name = model_meta_data['model']
        ModelClass = MODEL_LIST[model_name]
        model = ModelClass.initialize_from_meta_data(model_meta_data)

        # Initalize the model
        model.build_model(self.quantize_level_bits)
        model.initialize_model(self.model_init_key)

        # Do data dependent initialization
        model.data_dependent_init(data_dependent_init_key, self.data_loader, n_seed_examples=1000, batch_size=64)

        # Initialize the optimizer
        optimizer = Optimizer.initialize_from_meta_data(optimizer_meta_data)
        optimizer.initialize(model)

        self.model     = model
        self.optimizer = optimizer

        # Save a dummy first checkpoint
        self.checkpoint_experiment(0, self.opt_key, np.array([]))

    def load_experiment(self):

        # Load the keys
        self.load_keys()

        # Load the model and optimizer
        model = self.load_model_instance_from_meta_data()
        optimizer = self.load_optimizer_instance_from_meta_data()

        # Initalize the model and optimizer
        model.build_model(self.quantize_level_bits)
        model.initialize_model(self.model_init_key)
        optimizer.initialize(model)

        # Load the most recent optimizer parameters
        optimizer.load_opt_state_from_file(self.optimizer_state_path)

        # Fill these parameters in the model
        model.params = optimizer.get_params(optimizer.opt_state)

        # Load the model state
        model.load_state_from_file(self.model_state_path)

        # Ensure that the start iteration is correct
        try:
            losses = pd.read_csv(self.losses_path, header=None)
        except pd.errors.EmptyDataError:
            losses = np.array([])
        assert self.current_iteration == losses.shape[0], 'self.current_iteration: %d, losses.shape[0]: %d'%(self.current_iteration, losses.shape[0])

        self.model     = model
        self.optimizer = optimizer

    def initialize(self):
        if(self.current_iteration is None):
            # We have no saved checkpoint
            self.create_experiment_from_meta_data
        else:
            # We have a saved checkpoint
            self.load_experiment()

    #####################################################################

    def checkpoint_experiment(self, i, opt_key, losses):

        old_loss_path = self.losses_path

        # Update the current iteration
        self.current_iteration = i

        # Save the meta data for the model and optimizer
        self.save_model_meta_data()
        self.save_optimizer_meta_data()

        # Save the model state and optimizer state
        self.model.save_state(self.model_state_path)
        self.optimizer.save_opt_state(self.optimizer_state_path)

        # Save off keys
        self.opt_key = opt_key
        self.save_keys()

        # Copy old losses so that we can append to it
        if(os.path.exists(old_loss_path) and old_loss_path != self.losses_path):
            shutil.copy2(old_loss_path, self.losses_path)

        # Save the losses
        self.save_losses(losses)

        return []

    #####################################################################

    def train(self):
        self.optimizer.train(self.opt_key,
                             self.x_shape,
                             self.data_loader,
                             self.model,
                             self.current_iteration,
                             self.checkpoint_iters,
                             self.checkpoint_experiment)