import os
from tqdm import tqdm
import jax
from jax import random, vmap, jit
import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
import pandas as pd
from jax.experimental import optimizers
import yaml
import jax.numpy as np
import numpy as onp
import jax.ops

######################################################################################################################################################

def generate_images_for_fid(key,
                            sampler,
                            temperature,
                            sigma,
                            n_samples,
                            save_folder,
                            n_samples_per_batch=8):
    filled_sampler = partial(sampler, temperature=temperature, sigma=sigma)

    # Generate the keys we will use
    n_samples_per_batch = min(n_samples, n_samples_per_batch)
    n_batches = int(jnp.ceil(n_samples/n_samples_per_batch))
    keys = random.split(key, n_batches)

    # Generate the list of batch sizes we will be using
    batch_sizes = n_samples_per_batch*jnp.ones((n_samples//n_samples_per_batch),)
    if(n_samples%n_samples_per_batch != 0):
        batch_sizes = jnp.hstack([batch_sizes, n_samples%n_samples_per_batch])
    batch_sizes = batch_sizes.astype(jnp.int32)

    assert batch_sizes.shape[0] == keys.shape[0]

    # Loop over all of the samples
    index = 0
    for i, (key, batch_size) in enumerate(zip(keys, batch_sizes)):
        _, x = filled_sampler(batch_size, key)

        # Save the images
        for j, im in enumerate(x):
            path = os.path.join(save_folder, '%s.jpg'%index)
            im = im[:,:,0] if im.shape[-1] == 1 else im
            matplotlib.image.imsave(path, im)
            index += 1

################################################################################################################################################

def batched_samples(key, filled_sampler, n_samples, n_samples_per_batch):
    """ Generate a bunch of samples in batches """

    # Generate the keys we will use
    n_samples_per_batch = min(n_samples, n_samples_per_batch)
    n_batches = int(jnp.ceil(n_samples/n_samples_per_batch))
    keys = random.split(key, n_batches)

    # Generate the list of batch sizes we will be using
    batch_sizes = n_samples_per_batch*jnp.ones((n_samples//n_samples_per_batch),)
    if(n_samples%n_samples_per_batch != 0):
        batch_sizes = jnp.hstack([batch_sizes, n_samples%n_samples_per_batch])
    batch_sizes = batch_sizes.astype(jnp.int32)

    assert batch_sizes.shape[0] == keys.shape[0]

    # Pull all of the samples
    likelihoods = []
    samples = []
    for key, batch_size in zip(keys, batch_sizes):
        log_px, x = filled_sampler(batch_size, key)
        likelihoods.append(log_px)
        samples.append(x)

    return jnp.concatenate(samples, axis=0), jnp.concatenate(likelihoods, axis=0)

def batched_evaluate(key, fun, x, n_samples_per_batch):
    """ Generate a bunch of samples in batches """
    n_samples = x.shape[0]

    # Generate the keys we will use
    n_samples_per_batch = min(n_samples, n_samples_per_batch)
    n_batches = int(jnp.ceil(n_samples/n_samples_per_batch))
    keys = random.split(key, n_batches)

    # Generate the list of batch sizes we will be using
    batch_sizes = n_samples_per_batch*jnp.ones((n_samples//n_samples_per_batch),)
    if(n_samples%n_samples_per_batch != 0):
        batch_sizes = jnp.hstack([batch_sizes, n_samples%n_samples_per_batch])
    batch_sizes = batch_sizes.astype(jnp.int32)

    assert batch_sizes.shape[0] == keys.shape[0]

    # Pull all of the samples
    likelihoods = []
    embeddings = []
    i = 0
    pbar = tqdm(list(zip(keys, batch_sizes)))
    for key, batch_size in pbar:
        log_px, z = fun(x[i:i + batch_size], key)
        i += batch_size
        likelihoods.append(log_px)
        embeddings.append(z)

    return jnp.concatenate(embeddings, axis=0), jnp.concatenate(likelihoods, axis=0)

######################################################################################################################################################

def compare_vertical(key, experiments, n_samples, save_path, n_samples_per_batch=8, sigma=0.0):
    """ Compare samples from the different models """

    samples = []
    plot_names = []
    for exp, sampler, encoder, decoder in experiments:
        filled_sampler = partial(sampler, temperature=1.0, sigma=sigma)
        x, _ = batched_samples(key, filled_sampler, n_samples, n_samples_per_batch)
        samples.append(x)
        plot_names.append(exp.experiment_name)

    # Create the axes
    n_rows = n_samples
    n_cols = len(experiments)

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    if(len(experiments) == 2):
        plot_names = ['NF', 'NIF (Manifold)']
    for i, (x, plot_name) in enumerate(zip(samples, plot_names)):
        for j, im in enumerate(x):
            im = im[:,:,0] if im.shape[-1] == 1 else im
            ax = axes[j, i]
            ax.imshow(im)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both',length=0)
            if(j == 0):
                ax.set_title(plot_name, fontsize=20)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

######################################################################################################################################################

def compare_samples(key, experiments, n_samples, save_path, n_samples_per_batch=8, sigma=0.3):
    """ Compare samples from the different models """

    samples = []
    plot_names = []
    for exp, sampler, encoder, decoder in experiments:
        filled_sampler = partial(sampler, temperature=1.0, sigma=sigma)
        x, _ = batched_samples(key, filled_sampler, n_samples, n_samples_per_batch)
        samples.append(x)
        plot_names.append(exp.experiment_name)

    # Create the axes
    n_rows = len(experiments)
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    if(len(experiments) == 2):
        plot_names = ['NF', 'NIF']
    for i, (x, plot_name) in enumerate(zip(samples, plot_names)):
        for j, im in enumerate(x):
            im = im[:,:,0] if im.shape[-1] == 1 else im
            ax = axes[i,j]
            ax.imshow(im)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both',length=0)
            if(j == 0):
                ax.set_ylabel(plot_name, fontsize=20)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

######################################################################################################################################################

def compare_manifold_vs_full_samples(key, sampler, baseline_sampler, n_samples, save_path, n_samples_per_batch=8):
    """ Compare samples from a baseline and our  """

    samples = []

    filled_sampler = partial(baseline_sampler, temperature=1.0, sigma=0.0)
    x, _ = batched_samples(key, filled_sampler, n_samples, n_samples_per_batch)
    samples.append(x)

    filled_sampler = partial(sampler, temperature=1.0, sigma=1.0)
    x, _ = batched_samples(key, filled_sampler, n_samples, n_samples_per_batch)
    samples.append(x)

    filled_sampler = partial(sampler, temperature=1.0, sigma=0.0)
    x, _ = batched_samples(key, filled_sampler, n_samples, n_samples_per_batch)
    samples.append(x)

    # Create the axes
    n_rows = 3
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    plot_names = ['NF', 'NIF', 'NIF (Manifold)']
    for i, (x, plot_name) in enumerate(zip(samples, plot_names)):
        for j, im in enumerate(x):
            im = im[:,:,0] if im.shape[-1] == 1 else im
            ax = axes[i,j]
            ax.imshow(im)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', which='both',length=0)
            if(j == 0):
                ax.set_ylabel(plot_name, fontsize=20)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def reconstructions(data_key, key, data_loader, encoder, decoder, save_path, n_samples, quantize_level_bits, n_samples_per_batch=8):
    """ Generate reconstructions of data """

    # Pull samples
    x = data_loader((n_samples,), key=data_key)

    # Generate the reconstructions
    k1, k2 = random.split(key, 2)
    z,  _ = batched_evaluate(k1, encoder, x, n_samples_per_batch)
    fz, _ = batched_evaluate(k2, decoder, z, n_samples_per_batch)

    # Plot the reconstructions
    n_rows = 2
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    for i, im in enumerate(fz):
        im = im[:,:,0] if im.shape[-1] == 1 else im
        axes[0,i].imshow(im)
        axes[0,i].set_axis_off()

    for i, im in enumerate(x):
        im = im[:,:,0] if im.shape[-1] == 1 else im
        axes[1,i].imshow(im/(2.0**quantize_level_bits))
        axes[1,i].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

######################################################################################################################################################

def compare_t(key, experiments, n_samples, save_path, n_samples_per_batch=8):
    """ Compare samples at different values of t """

    # Define the samples we'll be using
    temperatures = jnp.linspace(0.0, 3.0, n_samples)

    # We will vmap over temperature. Also will be sharing the same random key everywhere
    def temp_sampler(sampler, key, temp):
        _, fz = sampler(1, key, temp, 0.0)
        return fz

    samples = []
    for exp, sampler, encoder, decoder in experiments:
        keys = jnp.array(random.split(key, n_samples))
        x = vmap(partial(temp_sampler, sampler))(keys, temperatures)[:,0,...]
        samples.append(x)

    # Create the axes
    n_rows = len(experiments)
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    for i, x in enumerate(samples):
        for j, im in enumerate(x):
            im = im[:,:,0] if im.shape[-1] == 1 else im
            axes[i,j].imshow(im)
            axes[i,j].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

######################################################################################################################################################

def samples_vary_t(data_key, key, experiments, n_samples, save_path, n_samples_per_batch=8, reuse_key=True):
    """ Compare the same sample for different values of t """

    # Use a sample from the NF model for these plots
    x = None
    for exp, sampler, encoder, decoder in experiments:
        if(exp.is_nf):
            _, x = sampler(1, data_key, 1.0, 1.0)
            break
    assert x is not None

    # Define the samples we'll be using
    temperatures = jnp.linspace(0.0, 3.0, n_samples)

    # We will vmap over temperature. Also will be sharing the same random key everywhere
    def temp_decode(decoder, z, temp, key):
        _, fz = decoder(z*temp, key, sigma=0.0)
        return fz

    # If we want to show the effect of a parameter, reuse a key
    if(reuse_key):
        keys = [key]*n_samples
    else:
        keys = random.split(key, n_samples)
    keys = jnp.array(keys)

    # Loop over every experiment
    all_temperature_samples = []
    for exp, sampler, encoder, decoder in experiments:

        # Encode the image
        print(x.shape)
        _, z = encoder(x, key=key, sigma=1.0) # Need to verify we want sigma of 1.0

        # Decode at different temperatures
        temperature_samples = vmap(partial(temp_decode, decoder, z))(temperatures, keys) # (n_temp, 1, x_shape)
        temperature_samples = temperature_samples[:,0,...]
        all_temperature_samples.append(temperature_samples)

    # Create the axes
    n_rows = len(experiments)
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    for i, x in enumerate(all_temperature_samples):
        for j, im in enumerate(x):
            im = im[:,:,0] if im.shape[-1] == 1 else im
            axes[i,j].imshow(im)
            axes[i,j].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

######################################################################################################################################################

def samples_vary_s(data_key, key, experiments, n_samples, save_path, n_samples_per_batch=8, reuse_key=True):
    """ Compare the same sample for different values of t """

    # Use a sample from the NF model for these plots
    x = None
    for exp, sampler, encoder, decoder in experiments:
        if(exp.is_nf):
            _, x = sampler(1, data_key, 1.0, 1.0)
            break
    assert x is not None

    # Define the samples we'll be using
    sigmas = jnp.linspace(0.0, 3.0, n_samples)

    # We will vmap over temperature. Also will be sharing the same random key everywhere
    def sigma_decode(decoder, z, s, key):
        _, fz = decoder(z, key, sigma=s)
        return fz

    # If we want to show the effect of a parameter, reuse a key
    if(reuse_key):
        keys = [key]*n_samples
    else:
        keys = random.split(key, n_samples)
    keys = jnp.array(keys)

    # Loop over every experiment
    all_sigma_samples = []
    for exp, sampler, encoder, decoder in experiments:

        # Encode the image
        _, z = encoder(x, key=key, sigma=1.0)

        # Decode at different sigmas
        sigma_samples = vmap(partial(sigma_decode, decoder, z))(sigmas, keys) # (n_temp, 1, x_shape)
        sigma_samples = sigma_samples[:,0,...]
        all_sigma_samples.append(sigma_samples)

    # Create the axes
    n_rows = len(experiments)
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    for i, x in enumerate(all_sigma_samples):
        for j, im in enumerate(x):
            im = im[:,:,0] if im.shape[-1] == 1 else im
            axes[i,j].imshow(im)
            axes[i,j].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

@jit
def cartesian_to_spherical(x):
    r = jnp.sqrt(jnp.sum(x**2))
    denominators = jnp.sqrt(jnp.cumsum(x[::-1]**2)[::-1])[:-1]
    phi = jnp.arccos(x[:-1]/denominators)

    last_value = jnp.where(x[-1] >= 0, phi[-1], 2*jnp.pi - phi[-1])
    phi = jax.ops.index_update(phi, -1, last_value)

    return jnp.hstack([r, phi])

@jit
def spherical_to_cartesian(phi_x):
    r = phi_x[0]
    phi = phi_x[1:]
    return r*jnp.hstack([1.0, jnp.cumprod(jnp.sin(phi))])*jnp.hstack([jnp.cos(phi), 1.0])

def interpolate_pairs(data_key, key, experiment, n_pairs, n_interp, save_path):
    """
    Interpolate images
    """
    exp, sampler, encoder, decoder = experiment

    # Load the data that we'll use for interpolation
    x_for_interpolation = exp.data_loader((2*n_pairs,), key=data_key)

    # Split the data into pairs
    random_pairs = random.randint(key, (2*n_pairs,), minval=0, maxval=x_for_interpolation.shape[0])
    pairs_iter = iter(random_pairs)
    index_pairs = [(next(pairs_iter), next(pairs_iter)) for _ in range(n_pairs)]

    n_cols = n_interp
    n_rows = len(index_pairs)

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    for i, (idx1, idx2) in enumerate(index_pairs):
        x = x_for_interpolation[[idx1, idx2]]

        # Find the embeddings of the data
        _, finvx = encoder(x, key, sigma=1.0)

        # Interpolate
        phi = jit(vmap(cartesian_to_spherical))(finvx)
        phi1, phi2 = phi
        interpolation_phi = jnp.linspace(phi1, phi2, n_interp)
        interpolation_z = jit(vmap(spherical_to_cartesian))(interpolation_phi)

        # Decode the interpolations
        _, fz = decoder(finvx, key, sigma=0.0)

        # Plot
        for j in range(n_interp):
            im = fz[j][:,:,0] if fz[j].shape[-1] == 1 else fz[j]
            axes[i,j].imshow(im)
            axes[i,j].set_axis_off()

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

# Compare log likelihoods on the validation set
# Tune the value of s for each model on the test set to find the highest log likelihood
def save_best_s_for_nll(key, experiments, save_path, batch_size=16):

    @partial(jit, static_argnums=(0,))
    def loss(encoder, s, key, x):
        log_px, _ = encoder(x, key, sigma=s)
        return -jnp.mean(log_px)

    # Optimize for each experiment
    best_values_of_s = {}
    for exp, sampler, encoder, decoder in experiments:
        data_loader = exp.data_loader

        # Going to optimize to find the best s
        s = 1.0
        if(True or exp.is_nf):
            # Don't need to do anything for normalizing flows
            best_values_of_s[exp.experiment_name] = s
            continue

        opt_init, opt_update, get_params = optimizers.adam(1e-3)
        opt_update = jit(opt_update)
        get_params = jit(get_params)
        opt_state = opt_init(s)

        valgrad = jax.value_and_grad(partial(loss, encoder))
        valgrad = jit(valgrad)

        pbar = tqdm(jnp.arange(1000))
        for i in pbar:
            # Take the next batch of test data
            key, *keys = random.split(key, 3)
            x = data_loader((batch_size,), key=keys[0], split='test')

            # Take a gradient step
            s = get_params(opt_state)
            nll, g = valgrad(s, keys[1], x)
            opt_state = opt_update(i, g, opt_state)

            pbar.set_description('s: %5.3f, nll: %5.3f'%(s, nll))

        # Store the result
        s = get_params(opt_state)
        best_values_of_s[exp.experiment_name] = float(s)

    # Save the best values of s
    with open(save_path, 'w') as f:
        yaml.dump(best_values_of_s, f)

def validation_nll_from_best_s(key, experiments, best_s_path, save_path, n_samples_per_batch=8):
    # Load the best values of s for each model
    with open(best_s_path) as f:
        best_values_of_s = yaml.safe_load(f)

    # Compute the validation nll for each experiment
    validation_bits_per_dims = {}
    for exp, _, encoder, _ in experiments:

        # Retrieve the best value of s and fill the encoder function
        s = best_values_of_s[exp.experiment_name]
        filled_encoder = jit(partial(encoder, sigma=s))

        # This should be small enough to fit in memory
        n_validation = exp.split_shapes[2]
        validation_data = exp.data_loader((n_validation,), start=0, split='validation')

        # Compute the log likelihoods
        _, log_likelihoods = batched_evaluate(key, filled_encoder, validation_data, n_samples_per_batch)

        # Compute the bits per dimensions to save
        bits_per_dim = -jnp.mean(log_likelihoods)/(jnp.prod(exp.model.x_shape)*jnp.log(2))
        validation_bits_per_dims[exp.experiment_name] = float(bits_per_dim)

    # Save the best values of s
    with open(save_path, 'w') as f:
        yaml.dump(validation_bits_per_dims, f)

################################################################################################################################################

@jit
def log_hx(x, A, log_diag_cov):
    diag_cov = jnp.exp(log_diag_cov)

    # Find the pseudo inverse and the projection
    ATSA = A.T/diag_cov@A
    ATSA_inv = jnp.linalg.inv(ATSA)
    z = jnp.dot(x, (ATSA_inv@A.T/diag_cov).T)
    x_proj = jnp.dot(z, A.T)/diag_cov

    # Get the terms that don't depend on z
    dim_x, dim_z = A.shape
    log_hx = -0.5*jnp.sum(x*(x/diag_cov - x_proj), axis=-1)
    log_hx -= 0.5*jnp.linalg.slogdet(ATSA)[1]
    log_hx -= 0.5*log_diag_cov.sum()
    log_hx -= 0.5*(dim_x - dim_z)*jnp.log(2*jnp.pi)
    return log_hx

def manifold_penalty(key, experiment, save_path):
    """ Compute the manifold penalty for each data point in the validation set """

    # Loop through the dataset and get the manifold penalties
    n_images = 0
    max_images = 500
    batch_size = 8
    penalties = []
    while(True):
        # Get the next batch of data
        x, is_done = experiment.data_loader((batch_size,), start=n_images, split='validation', return_if_at_end=True)
        n_images += x.shape[0]

        # Retrieve the manifold penalties
        _, _, state = experiment.model.forward(experiment.model.params, experiment.model.state, jnp.zeros(x.shape[0]), x, (), get_manifold_penalty=True)
        mp = state[-1][0]
        mp = -mp # Use the negative

        # Subtract the constant
        dim_z = jnp.prod(experiment.model.z_shape)
        dim_x = jnp.prod(experiment.model.x_shape)
        mp -= 0.5*(dim_x - dim_z)*jnp.log(2*jnp.pi)

        penalties.append(mp)

        if(is_done or n_images > max_images):
            break

    # Sort the penalties
    penalties = jnp.concatenate(penalties, axis=0)
    sorted_indices = jnp.argsort(penalties)
    assert 0

################################################################################################################################################

def get_embeddings_test(key, data_loader, model, n_samples_per_batch=4):
    """
    Save reconstructions
    """
    inital_key = key
    embeddings = []
    labels = []
    for j in range(10000//n_samples_per_batch):
        key, *keys = random.split(key, 3)
        _x, _y = data_loader((n_samples_per_batch,), None, j*n_samples_per_batch, 'tpv', True, True)
        keys = np.array(random.split(key, 64))
        log_px, z  = model(_x, keys[0])
        embeddings.append(z)
        labels.extend(_y)
        if(j % 100 == 1):
            print(j)
    final_labels = np.array(labels)
    final_embeddings = np.concatenate(embeddings, axis = 0)
    return final_embeddings, final_labels

def get_embeddings_training(key, data_loader, model, n_samples_per_batch=4):
    """
    Save reconstructions
    """
    inital_key = key
    embeddings = []
    labels = []
    for j in range(50000//n_samples_per_batch):
        key, *keys = random.split(key, 3)
        _x, _y = data_loader((n_samples_per_batch,), None, j*n_samples_per_batch, 'train', True, True)
        keys = np.array(random.split(key, 64))
        log_px, z  = model(_x, keys[0])
        embeddings.append(z)
        labels.extend(_y)
        if(j % 100 == 1):
            print(j)
    final_labels = np.array(labels)
    final_embeddings = np.concatenate(embeddings, axis = 0)
    return final_embeddings, final_labels


def save_embeddings(key, data_loader, model, save_path, test = True, n_samples_per_batch=4):
    if(test):
        test_embeddings, y = get_embeddings_test(key, data_loader, model, n_samples_per_batch=4)
        test_embeddings,  y = onp.array(test_embeddings), onp.array(y)
        onp.save(os.path.join(save_path, 'test_embeddings'), test_embeddings)
        onp.save(os.path.join(save_path, 'test_y'), y)
    else:
        training_embeddings, y = get_embeddings_training(key, data_loader, model, n_samples_per_batch=4)
        training_embeddings, training_y = onp.array(training_embeddings), onp.array(y)
        onp.save(os.path.join(save_path, 'training__embeddings'), training_embeddings)
        onp.save(os.path.join(save_path, 'training_y'), training_y)

def print_reduced_embeddings(key, data_loader, nf_model, nif_model, path, test=True, n_samples_per_batch=4):
    if(test):
        test_nif_embeddings = onp.array(onp.load(os.path.join(path, 'test_nif_embeddings.npy')))
        test_nf_embeddings = onp.array(onp.load(os.path.join(path, 'test_nf_embeddings.npy')))
        y = onp.array(onp.load(os.path.join(path, 'test_y.npy')))
    else:
        test_nif_embeddings = onp.array(onp.load(os.path.join(path, 'training_nif_embeddings.npy')))
        test_nf_embeddings = onp.array(onp.load(os.path.join(path, 'training_nf_embeddings.npy')))
        y = onp.array(onp.load(os.path.join(path, 'training_y.npy')))
    print(test_nif_embeddings == test_nf_embeddings)
    print(y.shape)
    nf_2d_embeddings = umap.UMAP(random_state=0).fit_transform(test_nf_embeddings, y=y)
    nif_2d_embeddings = umap.UMAP(random_state=0).fit_transform(test_nif_embeddings, y=y)
    colors = y

    def outlier_mask(data, m=2):
        return np.all(np.abs(data - np.mean(data)) < m * np.std(data), axis=1)

    #colorsnf = colors[outlier_mask(nf_2d_embeddings)]
    #colorsnif = colors[outlier_mask(nif_2d_embeddings)]
    #nf_2d_embeddings = nf_2d_embeddings[outlier_mask(nf_2d_embeddings)]
    #nif_2d_embeddings = nif_2d_embeddings[outlier_mask(nif_2d_embeddings)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(nif_2d_embeddings[:,0], nif_2d_embeddings[:,1], s=3.0, c=y, cmap='Spectral', alpha=0.6)
    scatter = axes[1].scatter(nf_2d_embeddings[:,0], nf_2d_embeddings[:,1], s=3.0, c=y, cmap='Spectral', alpha=0.6)

    #axes[0].set_title('Our Method', fontdict={'fontsize': 18})
    #axes[1].set_title('GLOW', fontdict={'fontsize': 18})

    #axes[0].xaxis.set_visible(False)
    #axes[0].yaxis.set_visible(False)
    #axes[1].xaxis.set_visible(False)
    #axes[1].yaxis.set_visible(False)
    #axes[0].set_xlim(1, 11)
    #axes[0].set_ylim(-4, 5)

    #axes[1].set_xlim(-5, 1.5)
    #axes[1].set_ylim(-5, 2)

    cbar = fig.colorbar(scatter, boundaries=np.arange(11) - 0.5)
    cbar.set_ticks(np.arange(10))
    cbar.ax.set_yticklabels(['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])
    cbar.ax.tick_params(labelsize=12)
    plt.savefig('subplot.pdf', format='pdf')
    plt.close()