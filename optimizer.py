from functools import partial
import os
from tqdm import tqdm
import jax
from jax import random, vmap, jit, pmap
from jax.experimental import optimizers
from jax.lib import xla_bridge
from jax.tree_util import tree_map
import jax.numpy as jnp
import util
import numpy as np
clip_grads = jit(optimizers.clip_grads)

################################################################################################################

MAX_ACCEPTABLE_LOSS = 1e10

@partial(jit, static_argnums=(0,))
def nll(forward, params, state, x, **kwargs):
    """ Return the mean negative log likelihood of the model """
    log_px, z, updated_state = forward(params, state, jnp.zeros(x.shape[0]), x, (), **kwargs)
    return -jnp.mean(log_px), updated_state

@partial(pmap, static_broadcasted_argnums=(0, 1, 2), axis_name='batch')
def spmd_update(forward, opt_update, get_params, i, opt_state, state, x_batch, key):
    """ Can distribute large batch sizes across a gpu """
    params = get_params(opt_state)

    # Create the autodiff function
    valgrad = jit(jax.value_and_grad(partial(nll, forward), has_aux=True))

    # Evaluate a gradient
    (val, updated_state), grads = valgrad(params, state, x_batch, key=key, test=util.TRAIN)

    # Early during training we can get spikes in training that make everything fail.
    @jit
    def perform_grad_update(dummy):

        # Sum the gradient over all of the gpu shards
        g = jax.lax.psum(grads, 'batch')

        # Gradient clipping
        g = clip_grads(g, 5.0)

        # Update the optimizer state
        return state, opt_update(i, g, opt_state)

    @jit
    def do_nothing(dummy):
        return state, opt_state

    # Only update the gradient if we know its not garbage
    predicate = val > MAX_ACCEPTABLE_LOSS
    updated_state, opt_state = jax.lax.cond(predicate, (), do_nothing, (), perform_grad_update)

    return val, updated_state, opt_state

################################################################################################################

class Optimizer():

    def __init__(self, batch_size=32, lr=1e-4, warmup=1e5, lr_decay=0.99999, n_gpus=None):
        self.batch_size     = batch_size
        self.lr             = lr
        self.warmup         = warmup
        self.lr_decay       = lr_decay

        assert self.lr < 1e-2

        self.n_gpus         = n_gpus if n_gpus is not None else xla_bridge.device_count()
        self.update_fun     = None
        self.get_params     = None
        self.opt_update     = None
        self.opt_state      = None
        self.max_iterations = 10000000 # Just keep training

    #####################################################################

    def initialize(self, model):
        """ Initialize the trainer with a model """

        def lr_schedule(i):
            return jnp.where(i < self.warmup,
                             self.lr*i/self.warmup,
                             self.lr*(self.lr_decay**(i - self.warmup)))

        opt_init, self.opt_update, self.get_params = optimizers.adam(lr_schedule)
        self.opt_update = jit(self.opt_update)
        self.get_params = jit(self.get_params)
        self.opt_state = opt_init(model.params)

        self.update_fun = partial(spmd_update, model.forward, self.opt_update, self.get_params)

    #####################################################################

    def gen_meta_data(self):
        """ Create a dictionary that will tell us exactly how to create this model """
        meta = {'batch_size': self.batch_size,
                'n_gpus'    : self.n_gpus,
                'lr'        : self.lr,
                'warmup'    : self.warmup,
                'lr_decay'  : self.lr_decay}
        return meta

    @classmethod
    def initialize_from_meta_data(cls, meta):
        batch_size = meta['batch_size']
        n_gpus     = meta['n_gpus']
        lr         = meta['lr']
        warmup     = meta['warmup']
        lr_decay   = meta['lr_decay']
        return Optimizer(batch_size, lr, warmup, lr_decay, n_gpus)

    #####################################################################

    def save_opt_state(self, path):
        util.save_pytree_to_file(self.opt_state, path)

    def load_opt_state_from_file(self, path):
        self.opt_state = util.load_pytree_from_file(self.opt_state, path)

    #####################################################################

    def train_step(self, key, i, replicated_model_state, replicated_opt_state, data_loader):
        # data_key, gpu_key = random.split(key, 2)
        data_key, gpu_key = random.split(key, 2)

        # Take the next batch of data.  This has a huge performance impact!!!
        x_batch = data_loader((self.n_gpus, self.batch_size), key=key, split='train')

        # We need to replicate things for each gpu
        train_keys = jnp.array(random.split(gpu_key, self.n_gpus))
        replicated_i = jnp.ones(self.n_gpus)*i

        replicated_val, replicated_model_state, replicated_opt_state = self.update_fun(replicated_i,
                                                                                       replicated_opt_state,
                                                                                       replicated_model_state,
                                                                                       x_batch,
                                                                                       train_keys)

        return util.unreplicate(replicated_val), replicated_model_state, replicated_opt_state

    #####################################################################

    def train(self, key, data_loader, model, start_it, checkpoint_iters=5000, save_hook=None):

        losses = []

        # Copy the model state
        replicated_model_state = util.replicate((self.n_gpus,), model.state)
        replicated_opt_state   = util.replicate((self.n_gpus,), self.opt_state)

        bits_per_dim_scale = jnp.prod(model.x_shape)*jnp.log(2)

        pbar = tqdm(np.arange(start_it, self.max_iterations), initial=start_it)
        for i in pbar:

            # Save a checkpoint
            if(i%checkpoint_iters == 0):

                # Update the model and the optimizer state
                model_state    = util.unreplicate(replicated_model_state)
                opt_state      = util.unreplicate(replicated_opt_state)
                model.state    = model_state
                self.opt_state = opt_state

                # Checkpoint the current model.  Also reset the losses array so that it doesn't get too big
                losses = save_hook(i, key, losses)

            # Make sure we do this after the checkpoint
            key, _ = random.split(key, 2)

            # Take a gradient step
            val, replicated_model_state, replicated_opt_state = self.train_step(key, i, replicated_model_state, replicated_opt_state, data_loader)

            # Convert to bits per dimension and save
            val /= bits_per_dim_scale
            losses.append(val)

            progress_str = f'Bits/Dim: {val:.3f}'
            pbar.set_description(progress_str)
