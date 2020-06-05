import shutil
import os
import numpy as np
import yaml
from model import MODEL_LIST
from optimizer import Optimizer
from jax import random, jit
import jax.numpy as jnp
import pandas as pd
import pathlib
import pickle
from datasets import celeb_dataset_loader, cifar10_data_loader, mnist_data_loader
from functools import partial
import util

class Experiment():
    """
    This class will represent everything inside a checkpoint
    """
    def __init__(self,
                 experiment_name,
                 quantize_level_bits,
                 checkpoint_iters,
                 start_it=None,
                 experiment_root='Experiments'):
        self.experiment_name     = experiment_name
        self.experiment_root     = os.path.join(experiment_root, experiment_name)
        self.data_loader         = None
        self.split_shapes        = None
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

    # @property
    # def plots_path(self):
    #     path = os.path.join(self.experiment_iteration_path, 'plots')
    #     pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    #     return path

    # def get_comparison_folder(self, exp):
    #     plots_path = self.plots_path
    #     folder_name = '%s_%d'%(exp.experiment_name, exp.current_iteration)
    #     path = os.path.join(plots_path, folder_name)
    #     pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    #     return path

    #####################################################################

    @property
    def is_nf(self):
        return type(self.model).__name__ == 'GLOW'

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

    def get_data_loader(self, dataset_name):

        data_key = random.PRNGKey(0)
        split = (0.6, 0.2, 0.2)

        if(dataset_name == 'CelebA'):
            data_fun = celeb_dataset_loader
        elif(dataset_name == 'CIFAR10'):
            data_fun = cifar10_data_loader
        elif(dataset_name == 'FashionMNIST'):
            data_fun = partial(mnist_data_loader, kind='fashion')
        else:
            assert 0, 'Invalid dataset'

        data_loader, x_shape, split_shapes = data_fun(data_key, quantize_level_bits=self.quantize_level_bits, split=split)
        self.data_loader  = data_loader
        self.split_shapes = split_shapes
        return x_shape

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

        # Get the data loader
        x_shape = self.get_data_loader(model.dataset_name)
        assert x_shape == model.x_shape, 'The dataset has the wrong dimensions!  Has %s, expected %s'%(str(x_shape), str(model.x_shape))

        # Initalize the model.  Use a key to ensure things are initialized correctly
        init_key = random.PRNGKey(0)
        model.build_model(self.quantize_level_bits, init_key=init_key)
        model.initialize_model(self.model_init_key)

        # Do data dependent initialization
        model.data_dependent_init(data_dependent_init_key, self.data_loader, batch_size=64)

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

        # Get the data loader
        x_shape = self.get_data_loader(model.dataset_name)
        assert x_shape == model.x_shape, 'The dataset has the wrong dimensions!  Has %s, expected %s'%(str(x_shape), str(model.x_shape))

        # Initalize the model and optimizer.  Use a key to ensure things are initialized correctly
        init_key = random.PRNGKey(0)
        model.build_model(self.quantize_level_bits, init_key=init_key)
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
                             self.data_loader,
                             self.model,
                             self.current_iteration,
                             self.checkpoint_iters,
                             self.checkpoint_experiment)

    #####################################################################

    def get_jitted_sampler(self, for_plotting=True):
        """ Create a sampler that we can use quickly """
        # jitted_inverse = (partial(self.model.inverse, self.model.params, self.model.state, test=util.TEST))
        jitted_inverse = jit(partial(self.model.inverse, self.model.params, self.model.state, test=util.TEST))

        def sampler(n_samples, key, temperature, sigma, **kwargs):
            # Use vmap to pull multiple samples
            k1, k2 = random.split(key, 2)

            # Sample from the latent state with some temperature
            z = random.normal(k1, (n_samples,) + self.model.z_shape)*temperature

            # Invert the samples
            log_pfz, fz, _ = jitted_inverse(jnp.zeros(n_samples), z, (), key=k2, sigma=sigma, **kwargs)

            # Undo the dequantization and logit scaling to make sure we end up between 0 and 1
            if(for_plotting):
                fz /= (2.0**self.quantize_level_bits)
                fz *= (1.0 - 2*0.05) # This is the default scaling using in nf.Logit
                fz += 0.05

            return log_pfz, fz

        return sampler

    #####################################################################

    def get_jitted_forward(self):
        """ Create an encoder that we can use quickly """
        jitted_forward = partial(self.model.forward, self.model.params, self.model.state, test=util.TEST)
        # jitted_forward = jit(partial(self.model.forward, self.model.params, self.model.state, test=util.TEST))

        def encoder(x, key, **kwargs):
            if(x.ndim == len(self.model.x_shape)):
                batch_size = 1
            else:
                batch_size = x.shape[0]

            log_px, z, _ = jitted_forward(jnp.zeros(batch_size), x, (), key=key, **kwargs)
            return log_px, z

        return encoder

    def get_jitted_inverse(self, for_plotting=True):
        """ Create an decoder that we can use quickly """
        jitted_inverse = partial(self.model.inverse, self.model.params, self.model.state, test=util.TEST, s=0.0)
        # jitted_inverse = jit(partial(self.model.inverse, self.model.params, self.model.state, test=util.TEST, s=0.0))

        def decoder(z, key, **kwargs):
            if(z.ndim == len(self.model.z_shape)):
                batch_size = 1
            else:
                batch_size = z.shape[0]

            log_px, x, _ = jitted_inverse(jnp.zeros(batch_size), z, (), key=key, **kwargs)

            # Undo the dequantization and logit scaling to make sure we end up between 0 and 1
            if(for_plotting):
                x /= (2.0**self.quantize_level_bits)
                x *= (1.0 - 2*0.05) # This is the default scaling using in nf.Logit
                x += 0.05

            return log_px, x

        return decoder