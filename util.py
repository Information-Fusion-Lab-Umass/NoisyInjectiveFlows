import os
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
import jax
import pickle
import pathlib

TEST = jnp.ones((0, 0))
TRAIN = jnp.ones((0,))

################################################################################################################


def scaled_logsumexp(x, log_b, axis=0):
    """ logsumexp with scaling
    """
    x_max = jnp.amax(log_b + x, axis=axis, keepdims=True)
    y = jnp.sum(jnp.exp(log_b + x - x_max), axis=axis)
    sign_y = jnp.sign(y)
    abs_y = jnp.log(jnp.abs(y))
    return abs_y + jnp.squeeze(x_max, axis=axis)


# def scaled_logsumexp(x, b, axis=0):
#     """ logsumexp with scaling
#     """
#     x_max = jnp.amax(x, axis=axis, keepdims=True)
#     y = jnp.sum(b*jnp.exp(x - x_max), axis=axis)
#     sign_y = jnp.sign(y)
#     abs_y = jnp.log(jnp.abs(y))
#     return abs_y + jnp.squeeze(x_max, axis=axis)

################################################################################################################

@partial(jit, static_argnums=(0,))
def replicate(shape, pytree):
    replicate_fun = lambda x: jnp.broadcast_to(x, shape + x.shape)
    return tree_map(replicate_fun, pytree)

@jit
def unreplicate(pytree):
    return tree_map(lambda x:x[0], pytree)

################################################################################################################

def save_np_array_to_file(np_array, path):
    np.savetxt(path, np_array, delimiter=",")

def save_pytree_to_file(pytree, path):
    """ Save a pytree to file in pickle format"""
    dir_structure, file_name = os.path.split(path)
    assert file_name.endswith('.npz')

    # Create the path if it doesn't exist
    pathlib.Path(dir_structure).mkdir(parents=True, exist_ok=True)

    # Save the raw numpy parameters
    flat_pytree, _ = ravel_pytree(pytree)
    numpy_tree = np.array(flat_pytree)

    # Save the array to an npz file
    np.savez_compressed(path, flat_tree=numpy_tree)

def load_pytree_from_file(pytree, path):
    assert os.path.exists(path), '%s does not exist!'%path

    # Load the pytree structure
    _, unflatten = ravel_pytree(pytree)

    with np.load(path) as data:
        numpy_tree = data['flat_tree']

    return unflatten(numpy_tree)

################################################################################################################

@jit
def is_testing(x):
    return x.ndim == 2

################################################################################################################

def gaussian_logpdf(x, mean, cov):
    dx = x - mean
    cov_inv = jnp.linalg.inv(cov)
    log_px = -0.5*jnp.sum(jnp.dot(dx, cov_inv.T)*dx, axis=-1)
    return log_px - 0.5*jnp.linalg.slogdet(cov)[1] - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

def gaussian_diag_cov_logpdf(x, mean, log_diag_cov):
    dx = x - mean
    log_px = -0.5*jnp.sum(dx*jnp.exp(-log_diag_cov)*dx, axis=-1)
    return log_px - 0.5*jnp.sum(log_diag_cov) - 0.5*x.shape[-1]*jnp.log(2*jnp.pi)

################################################################################################################

@jit
def upper_cho_solve(chol, x):
    return jax.scipy.linalg.cho_solve((chol, True), x)

def upper_triangular_indices(N):
    values = jnp.arange(N)
    padded_values = jnp.hstack([values, 0])

    idx = np.ogrid[:N,N:0:-1]
    idx = sum(idx) - 1

    mask = jnp.arange(N) >= jnp.arange(N)[:,None]
    return (idx + jnp.cumsum(values + 1)[:,None][::-1] - N + 1)*mask

def n_elts_upper_triangular(N):
    return N*(N + 1) // 2 - 1

def upper_triangular_from_values(vals, N):
    assert n_elts_upper_triangular(N) == vals.shape[-1]
    zero_padded_vals = jnp.pad(vals, (1, 0))
    return zero_padded_vals[upper_triangular_indices(N)]

tri_solve = jax.scipy.linalg.solve_triangular
L_solve = jit(partial(tri_solve, lower=True, unit_diagonal=True))
U_solve = jit(partial(tri_solve, lower=False, unit_diagonal=True))

################################################################################################################

@jit
def householder(x, v):
    return x - 2*jnp.einsum('i,j,j', v, v, x)/jnp.sum(v**2)

@jit
def householder_prod_body(carry, inputs):
    x = carry
    v = inputs
    return householder(x, v), 0

@jit
def householder_prod(x, vs):
    return jax.lax.scan(householder_prod_body, x, vs)[0]

@jit
def householder_prod_transpose(x, vs):
    return jax.lax.scan(householder_prod_body, x, vs[::-1])[0]
