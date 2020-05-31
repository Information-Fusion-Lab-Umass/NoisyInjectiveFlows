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
import jax.ops

######################################################################################################################################################

def generate_images_for_fid(key,
                            sampler,
                            temperature,
                            sigma,
                            n_samples,
                            save_folder,
                            n_samples_per_batch=32):
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

def compare_samples(key, experiments, n_samples, save_path, n_samples_per_batch=32):
    """ Compare samples from the different models """

    samples = []
    for exp, sampler, encoder, decoder in experiments:
        filled_sampler = partial(sampler, temperature=1.0, sigma=0.0)
        x, _ = batched_samples(key, filled_sampler, n_samples, n_samples_per_batch)
        samples.append(x)

    # Create the axes
    n_rows = len(experiments)
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
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

################################################################################################################################################

def reconstructions(data_key, key, data_loader, encoder, decoder, save_path, n_samples, quantize_level_bits, n_samples_per_batch=32):
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

def compare_t(key, experiments, n_samples, save_path, n_samples_per_batch=32):
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

def samples_vary_t(data_key, key, experiments, n_samples, save_path, n_samples_per_batch=32, reuse_key=True):
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
        _, z = encoder(x, key=key, sigma=1.0) # Need to verify we want sigma of 1.0

        # Decode at different temperatures
        temperature_samples = vmap(partial(temp_decode, decoder, z))(temperatures, keys) # (n_temp, 1, x_shape)
        temperature_samples = temperature_samples[:,0,...]
        all_temperature_samples.append(temperature_samples)

    # Create the axes
    n_rows = len(experiments)
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
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

def samples_vary_s(data_key, key, experiments, n_samples, save_path, n_samples_per_batch=32, reuse_key=True):
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
        if(exp.is_nf):
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

def validation_nll_from_best_s(key, experiments, best_s_path, save_path, n_samples_per_batch=32):
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
