import os
import glob
import jax
import jax.numpy as jnp
import staxplusplus as spp
import normalizing_flows as nf
import jax.nn.initializers as jaxinit
from jax.tree_util import tree_flatten
import util
import non_dim_preserving as ndp
from functools import partial

######################################################################################################################################################

class Model():

    def __init__(self, dataset_name, x_shape):
        self.x_shape      = x_shape
        self.dataset_name = dataset_name

        self.init_fun, self.forward, self.inverse = None, None, None
        self.names, self.z_shape, self.params, self.state = None, None, None, None
        self.n_params = None

    def get_architecture(self):
        assert 0, 'unimplemented'

    def get_prior(self):
        assert 0, 'unimplemented'

    #####################################################################

    def build_model(self, quantize_level_bits):
        architecture = self.get_architecture()
        prior = self.get_prior()

        # Use uniform dequantization to build our model
        flow = nf.sequential_flow(nf.Dequantization(scale=2**quantize_level_bits),
                                  nf.Logit(),
                                  architecture,
                                  nf.Flatten(),
                                  prior)

        self.init_fun, self.forward, self.inverse = flow

    #####################################################################

    def initialize_model(self, key):
        assert self.init_fun is not None, 'Need to call build_model'
        self.names, self.z_shape, self.params, self.state = self.init_fun(key, self.x_shape, ())
        self.n_params = jax.flatten_util.ravel_pytree(self.params)[0].shape[0]
        print('Total number of parameters:', self.n_params)

    #####################################################################

    def gen_meta_data(self):
        """ Create a dictionary that will tell us exactly how to create this model """
        meta = {'x_shape'     : list(self.x_shape),
                'dataset_name': self.dataset_name,
                'model'       : None}
        return meta

    @classmethod
    def initialize_from_meta_data(cls, meta):
        assert 0, 'unimplemented'

    #####################################################################

    def save_state(self, path):
        util.save_pytree_to_file(self.state, path)

    def load_state_from_file(self, path):
        self.state = util.load_pytree_from_file(self.state, path)

######################################################################################################################################################

class GLOW(Model):

    def __init__(self, dataset_name, x_shape, n_filters=256, n_blocks=16, n_multiscale=5):
        super().__init__(dataset_name, x_shape)
        self.n_filters    = n_filters
        self.n_blocks     = n_blocks
        self.n_multiscale = n_multiscale

    #####################################################################

    def get_architecture(self):
        """ Build the architecture from GLOW https://arxiv.org/pdf/1807.03039.pdf """

        def GLOWNet(out_shape, n_filters):
            """ Transformation used inside affine coupling """
            _, _, channels = out_shape
            return spp.sequential(spp.Conv(n_filters, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False),
                                  spp.Relu(),
                                  spp.Conv(n_filters, filter_shape=(1, 1), padding=((0, 0), (0, 0)), bias=True, weightnorm=False),
                                  spp.Relu(),
                                  spp.Conv(2*channels, filter_shape=(3, 3), padding=((1, 1), (1, 1)), bias=True, weightnorm=False, W_init=jaxinit.zeros, b_init=jaxinit.zeros),
                                  spp.Split(2, axis=-1),
                                  spp.parallel(spp.Tanh(), spp.Identity()))  # log_s, t

        def GLOWComponent(name_iter, n_filters, n_blocks):
            """ Compose glow blocks """
            layers = [nf.GLOWBlock(partial(GLOWNet, n_filters=n_filters),
                                   masked=False,
                                   name=next(name_iter),
                                   additive_coupling=False)]*n_blocks
            return nf.sequential_flow(nf.Debug(''), *layers)

        # To initialize our model, we want a debugger to print out the size of the network at each multiscale
        debug_kwargs = dict(print_init_shape=True, print_forward_shape=False, print_inverse_shape=False, compare_vals=False)

        # We want to name the glow blocks so that we can do data dependent initialization
        name_iter = iter(['glow_%d'%i for i in range(400)])

        # The multiscale architecture factors out pixels
        def multi_scale(flow):
            return nf.sequential_flow(nf.Squeeze(),
                                      GLOWComponent(name_iter, self.n_filters, self.n_blocks),
                                      nf.FactorOut(2),
                                      nf.factored_flow(flow, nf.Identity()),
                                      nf.FanInConcat(2),
                                      nf.UnSqueeze())
        flow = nf.Identity()
        for i in range(self.n_multiscale):
            flow = multi_scale(flow)

        return flow

    def get_prior(self):
        return nf.UnitGaussianPrior()

    #####################################################################

    def gen_meta_data(self):
        """ Create a dictionary that will tell us exactly how to create this model """
        meta = {'n_filters'   : self.n_filters,
                'n_blocks'    : self.n_blocks,
                'n_multiscale': self.n_multiscale,
                'model'       : 'GLOW'}

        parent_meta = super().gen_meta_data()
        parent_meta.update(meta)
        return parent_meta

    @classmethod
    def initialize_from_meta_data(cls, meta):
        """ Using a meta data, construct an instance of this model """
        dataset_name = meta['dataset_name']
        x_shape      = tuple(meta['x_shape'])
        n_filters    = meta['n_filters']
        n_blocks     = meta['n_blocks']
        n_multiscale = meta['n_multiscale']

        return GLOW(dataset_name, x_shape, n_filters, n_blocks, n_multiscale)

    #####################################################################

    def data_dependent_init(self, key, data_loader, n_seed_examples=1000, batch_size=64):
        actnorm_names = [name for name in tree_flatten(self.names)[0] if 'act_norm' in name]
        flow_model = (self.names, self.z_shape, self.params, self.state), self.forward, self.inverse
        params = nf.multistep_flow_data_dependent_init(None,
                                                       actnorm_names,
                                                       flow_model,
                                                       (),
                                                       'actnorm_seed',
                                                       key,
                                                       data_loader=data_loader,
                                                       n_seed_examples=n_seed_examples,
                                                       batch_size=batch_size,
                                                       notebook=False)
        self.params = params

######################################################################################################################################################

class SimpleNIF(GLOW):

    def __init__(self, dataset_name, x_shape, z_dim, n_filters=256, n_blocks=16, n_multiscale=5):
        super().__init__(dataset_name, x_shape, n_filters, n_blocks, n_multiscale)
        self.z_dim = z_dim

    def get_prior(self):
        return ndp.AffineGaussianPriorDiagCov(self.z_dim)

    #####################################################################

    def gen_meta_data(self):
        """ Create a dictionary that will tell us exactly how to create this model """
        parent_meta = super().gen_meta_data()
        meta = {'z_dim': self.z_dim,
                'model': 'SimpleNIF'}
        parent_meta.update(meta)
        return parent_meta

    @classmethod
    def initialize_from_meta_data(cls, meta):
        """ Using a meta data, construct an instance of this model """
        dataset_name = meta['dataset_name']
        x_shape      = tuple(meta['x_shape'])
        n_filters    = meta['n_filters']
        n_blocks     = meta['n_blocks']
        n_multiscale = meta['n_multiscale']
        z_dim        = meta['z_dim']

        return SimpleNIF(dataset_name, x_shape, z_dim, n_filters, n_blocks, n_multiscale)

######################################################################################################################################################

# Use a global to make loading easy
MODEL_LIST = {'GLOW'     : GLOW,
              'SimpleNIF': SimpleNIF}
