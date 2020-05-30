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

    def get_architecture(self, init_key=None):
        assert 0, 'unimplemented'

    def get_prior(self):
        assert 0, 'unimplemented'

    #####################################################################

    def build_model(self, quantize_level_bits, init_key=None):
        architecture = self.get_architecture(init_key=init_key)
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

    def __init__(self, dataset_name, x_shape, n_filters=256, n_blocks=16, n_multiscale=5, data_init_iterations=1000):
        super().__init__(dataset_name, x_shape)
        self.n_filters            = n_filters
        self.n_blocks             = n_blocks
        self.n_multiscale         = n_multiscale
        self.data_init_iterations = data_init_iterations

    #####################################################################

    def get_architecture(self, init_key=None):
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
        def multi_scale(i, flow):
            if(isinstance(self.n_filters, int)):
                n_filters = self.n_filters
            else:
                n_filters = self.n_filters[i]

            if(isinstance(self.n_blocks, int)):
                n_blocks  = self.n_blocks
            else:
                n_blocks  = self.n_blocks[i]

            return nf.sequential_flow(nf.Squeeze(),
                                      GLOWComponent(name_iter, n_filters, n_blocks),
                                      nf.FactorOut(2),
                                      nf.factored_flow(flow, nf.Identity()),
                                      nf.FanInConcat(2),
                                      nf.UnSqueeze())
        flow = nf.Identity()
        for i in range(self.n_multiscale):
            flow = multi_scale(i, flow)

        if(init_key is not None):
            # Add the ability to ensure that things arae initialized together
            flow = nf.key_wrap(flow, init_key)

        return flow

    def get_prior(self):
        return nf.UnitGaussianPrior()

    #####################################################################

    def gen_meta_data(self):
        """ Create a dictionary that will tell us exactly how to create this model """
        meta = {'n_filters'           : self.n_filters,
                'n_blocks'            : self.n_blocks,
                'n_multiscale'        : self.n_multiscale,
                'data_init_iterations': self.data_init_iterations,
                'model'               : 'GLOW'}

        parent_meta = super().gen_meta_data()
        parent_meta.update(meta)
        return parent_meta

    @classmethod
    def initialize_from_meta_data(cls, meta):
        """ Using a meta data, construct an instance of this model """
        dataset_name         = meta['dataset_name']
        x_shape              = tuple(meta['x_shape'])
        n_filters            = meta['n_filters']
        n_blocks             = meta['n_blocks']
        n_multiscale         = meta['n_multiscale']
        data_init_iterations = meta['data_init_iterations']

        return GLOW(dataset_name, x_shape, n_filters, n_blocks, n_multiscale, data_init_iterations)

    #####################################################################

    def data_dependent_init(self, key, data_loader, batch_size=64):
        actnorm_names = [name for name in tree_flatten(self.names)[0] if 'act_norm' in name]
        flow_model = (self.names, self.z_shape, self.params, self.state), self.forward, self.inverse
        params = nf.multistep_flow_data_dependent_init(None,
                                                       actnorm_names,
                                                       flow_model,
                                                       (),
                                                       'actnorm_seed',
                                                       key,
                                                       data_loader=data_loader,
                                                       n_seed_examples=self.data_init_iterations,
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

class NIF(GLOW):

    def __init__(self, dataset_name, x_shape, z_dim, n_filters=256, n_blocks=16, n_multiscale=5, n_hidden_layers=3, layer_size=1024, n_flat_layers=5, n_importance_samples=16):
        super().__init__(dataset_name, x_shape, n_filters, n_blocks, n_multiscale)
        self.z_dim                = z_dim
        self.n_hidden_layers      = n_hidden_layers
        self.layer_size           = layer_size
        self.n_flat_layers        = n_flat_layers
        self.n_importance_samples = n_importance_samples

    def get_prior(self):
        an_names = iter(['flat_act_norm_%d'%i for i in range(100)])

        def FlatTransform(out_shape):
            dense_layers = [spp.Dense(self.layer_size), spp.Relu()]*self.n_hidden_layers
            return spp.sequential(*dense_layers,
                                  spp.Dense(out_shape[-1]*2),
                                  spp.Split(2, axis=-1),
                                  spp.parallel(spp.Tanh(), spp.Identity())) # log_s, t

        layers = [nf.AffineCoupling(FlatTransform), nf.ActNorm(name=next(an_names)), nf.Reverse()]*self.n_flat_layers
        prior_flow = nf.sequential_flow(*layers, nf.UnitGaussianPrior())
        return ndp.TallAffineDiagCov(prior_flow, self.z_dim, n_training_importance_samples=self.n_importance_samples)

    #####################################################################

    def gen_meta_data(self):
        """ Create a dictionary that will tell us exactly how to create this model """
        parent_meta = super().gen_meta_data()
        meta = {'z_dim'               : self.z_dim,
                'n_hidden_layers'     : self.n_hidden_layers,
                'layer_size'          : self.layer_size,
                'n_flat_layers'       : self.n_flat_layers,
                'n_importance_samples': self.n_importance_samples,
                'model'          : 'NIF'}
        parent_meta.update(meta)
        return parent_meta

    @classmethod
    def initialize_from_meta_data(cls, meta):
        """ Using a meta data, construct an instance of this model """
        dataset_name         = meta['dataset_name']
        x_shape              = tuple(meta['x_shape'])
        n_filters            = meta['n_filters']
        n_blocks             = meta['n_blocks']
        n_multiscale         = meta['n_multiscale']
        z_dim                = meta['z_dim']
        n_hidden_layers      = meta['n_hidden_layers']
        layer_size           = meta['layer_size']
        n_flat_layers        = meta['n_flat_layers']
        n_importance_samples = meta['n_importance_samples']

        return NIF(dataset_name, x_shape, z_dim, n_filters, n_blocks, n_multiscale, n_hidden_layers, layer_size, n_flat_layers, n_importance_samples)

######################################################################################################################################################

# Use a global to make loading easy
MODEL_LIST = {'GLOW'     : GLOW,
              'SimpleNIF': SimpleNIF,
              'NIF'      : NIF}
