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
import numpy as np
import jax.ops
import umap
import util

######################################################################################################################################################

def generate_images_for_fid(key,
                            sampler,
                            temperature,
                            sigma,
                            n_samples,
                            save_folder,
                            n_samples_per_batch=128):
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
    for i, (key, batch_size) in tqdm(list(enumerate(zip(keys, batch_sizes)))):
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

def figure_2_plots(key, nf_exp, nif_exp, n_samples, save_path, n_samples_per_batch=8, sigma=0.0):

    samples = []
    plot_names = []
    for exp, sampler, encoder, decoder in [nf_exp, nif_exp]:
        filled_sampler = partial(sampler, temperature=1.0, sigma=sigma)
        x, _ = batched_samples(key, filled_sampler, n_samples, n_samples_per_batch)
        samples.append(x)
        plot_names.append(exp.experiment_name)

    # Create the axes
    n_rows = 2
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    if(axes.ndim == 1):
        axes = axes[None]
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
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

def compare_samples(key, experiments, n_samples, save_path, n_samples_per_batch=8, sigma=0.0):
    """ Compare samples from the different models """

    samples = []
    plot_names = []
    for exp, sampler, encoder, decoder in experiments:
        if(exp.is_nf):
            temp = 1.0
        else:
            temp = 2.0
        filled_sampler = partial(sampler, temperature=temp, sigma=sigma)
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

################################################################################################################################################

# def reconstructions(data_key, key, data_loader, encoder, decoder, save_path, n_samples, quantize_level_bits, n_samples_per_batch=8):
#     """ Generate reconstructions of data """

#     # Pull samples
#     x = data_loader((n_samples,), key=data_key)

#     # Generate the reconstructions
#     k1, k2 = random.split(key, 2)
#     z,  _ = batched_evaluate(k1, encoder, x, n_samples_per_batch)
#     fz, _ = batched_evaluate(k2, decoder, z, n_samples_per_batch)

#     # Plot the reconstructions
#     n_rows = 2
#     n_cols = n_samples

#     fig, axes = plt.subplots(n_rows, n_cols)
#     if(axes.ndim == 1):
#         axes = axes[None]
#     fig.set_size_inches(2*n_cols, 2*n_rows)

#     # Plot the samples
#     for i, im in enumerate(fz):
#         im = im[:,:,0] if im.shape[-1] == 1 else im
#         axes[0,i].imshow(im)
#         axes[0,i].set_axis_off()

#     for i, im in enumerate(x):
#         im = im[:,:,0] if im.shape[-1] == 1 else im
#         axes[1,i].imshow(im/(2.0**quantize_level_bits))
#         axes[1,i].set_axis_off()

#     plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
#     plt.savefig(save_path, bbox_inches='tight', format='pdf')
#     plt.close()

######################################################################################################################################################

def compare_t(key, experiments, n_samples, save_path, n_samples_per_batch=8):
    """ Compare samples at different values of t """

    # Define the samples we'll be using
    temperatures = jnp.linspace(0.0, 5.0, n_samples)

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
        for j, (t, im) in enumerate(zip(temperatures, x)):
            im = im[:,:,0] if im.shape[-1] == 1 else im
            axes[i,j].imshow(im)
            if(i == 0):
                axes[i,j].set_title('t=%5.3f'%t, fontsize=18)
            if(j == 0):
                if(i == 0):
                    axes[i,j].set_ylabel('NF', fontsize=18)
                else:
                    axes[i,j].set_ylabel('NIF', fontsize=18)
                axes[i,j].set_yticklabels([])
                axes[i,j].set_xticklabels([])
                axes[i,j].tick_params(axis='both', which='both',length=0)
            else:
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

def vary_s(data_key, key, experiment, n_samples, save_path, n_samples_per_batch=8, reuse_key=True):
    """ Compare the same sample for different values of t """
    exp, sampler, encoder, decoder = experiment

    # Define the samples we'll be using
    sigmas = jnp.linspace(0.0, 1.0, n_samples)

    # We will vmap over temperature. Also will be sharing the same random key everywhere
    def sigma_decode(decoder, z, s):
        _, fz = decoder(z, key, sigma=s)
        return fz

    # Encode the image
    z = random.normal(data_key, exp.model.z_shape)*1.5

    # Decode at different sigmas
    sigma_samples = vmap(partial(sigma_decode, decoder, z))(sigmas)

    # Create the axes
    n_rows = 1
    n_cols = n_samples

    fig, axes = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(2*n_cols, 2*n_rows)

    # Plot the samples
    for j, (im, s) in enumerate(zip(sigma_samples, sigmas)):
        im = im[:,:,0] if im.shape[-1] == 1 else im
        axes[j].imshow(im)
        axes[j].set_axis_off()
        axes[j].set_title('s=%5.3f'%s, fontsize=18)

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

################################################################################################################################################

def samples_vary_s(data_key, key, experiments, n_samples, save_path, n_samples_per_batch=8, reuse_key=True):
    """ Compare the same sample for different values of t """

    # Use a sample from the NF model for these plots
    x = None
    for exp, sampler, encoder, decoder in experiments:
        if(exp.is_nf == False):
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

# def save_probability_difference(key, experiment1, experiment2, save_path, n_samples_per_batch=8):
#     """ Compute the KL divergences and total variation between exp1 and exp2 """

#     # Compute the validation nll for each experiment
#     kl_qp = 0.0
#     kl_pq = 0.0
#     differences = []

#     exp1, sampler1, encoder1, _ = experiment1
#     exp2, sampler2, encoder2, _ = experiment2

#     filled_encoder1 = jit(partial(encoder1, sigma=1.0))
#     filled_encoder2 = jit(partial(encoder2, sigma=1.0))

#     # KL[p||q]
#     p_over_q = []
#     for key in random.split(key, 100):
#         k1, k2, k3 = random.split(key, 3)

#         # Sample from p(x)
#         _, x = sampler1(n_samples_per_batch, k1, 1.0, 1.0)

#         # Compute log p(x) and log q(x)
#         log_px, _ = filled_encoder1(x, k2)
#         log_qx, _ = filled_encoder2(x, k3)

#         p_over_q.append(log_px - log_qx)

#     p_over_q = jnp.concatenate(p_over_q, axis=0)
#     kl_pq = jax.scipy.special.logsumexp(p_over_q)

#     # KL[q||p]
#     q_over_p = []
#     for key in random.split(key, 100):
#         k1, k2, k3 = random.split(key, 3)

#         # Sample from q(x)
#         _, x = sampler2(n_samples_per_batch, key, 1.0, 1.0)

#         # Compute log q(x) and log p(x)
#         log_qx, _ = filled_encoder2(x, k2)
#         log_px, _ = filled_encoder1(x, k1)

#         q_over_p.append(log_qx - log_px)

#     q_over_p = jnp.concatenate(q_over_p, axis=0)
#     kl_qp = jax.scipy.special.logsumexp(q_over_p)

#     metrics = {'kl_qp': kl_qp,
#                'kl_pq': kl_pq}

#     print(metrics)

#     # Save the best values of s
#     with open(save_path, 'w') as f:
#         yaml.dump(metrics, f)


def save_probability_difference(key, experiment1, experiment2, save_path, n_samples_per_batch=8):
    """ Compute the KL divergences and total variation between exp1 and exp2 """

    # Compute the validation nll for each experiment
    kl_qp = 0.0
    kl_pq = 0.0
    differences = []

    exp1, _, encoder1, _ = experiment1
    exp2, _, encoder2, _ = experiment2

    filled_encoder1 = jit(partial(encoder1, sigma=1.0))
    filled_encoder2 = jit(partial(encoder2, sigma=1.0))

    # This should be small enough to fit in memory
    n_validation = exp1.split_shapes[2]
    validation_data = exp1.data_loader((n_validation,), start=0, split='validation')

    # Compute the log likelihoods
    _, log_likelihoods1 = batched_evaluate(key, filled_encoder1, validation_data, n_samples_per_batch)
    _, log_likelihoods2 = batched_evaluate(key, filled_encoder2, validation_data, n_samples_per_batch)

    differences = log_likelihoods1 - log_likelihoods2
    kl_qp = util.scaled_logsumexp(differences, log_likelihoods1)
    kl_pq = util.scaled_logsumexp(-differences, log_likelihoods2)
    abs_difference = jnp.abs(differences)

    mean_abs_diff = jnp.mean(abs_difference)
    std_abs_diff = jnp.std(abs_difference)

    total_variation = jnp.max(abs_difference)

    metrics = {'kl_qp': kl_qp,
               'kl_pq': kl_pq,
               'mean_abs_diff': mean_abs_diff,
               'std_abs_diff': std_abs_diff,
               'total_variation': total_variation}

    print(metrics)

    # Save the best values of s
    with open(save_path, 'w') as f:
        yaml.dump(metrics, f)

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

# def embeddings_to_df():
#     path = 'Results/cifar_128_225000.npz'
#     with np.load(path) as data:
#         z, y, u = data['z'], data['y'], data['u']


################################################################################################################################################

def save_test_embeddings(key, experiment, save_path, n_samples_per_batch=64):

    # Load the full datset
    exp, sampler, encoder, decoder = experiment
    n_train, n_test, n_validation = exp.split_shapes
    data_loader = exp.data_loader

    # Compute our embeddings
    x, y = data_loader((n_test + n_validation,), start=0, split='tpv', return_labels=True, onehot=False)
    z, _ = batched_evaluate(key, encoder, x, n_samples_per_batch)

    # Compute the UMAP embeddings
    u = umap.UMAP(random_state=0).fit_transform(z, y=y)

    z, y = np.array(z), np.array(y)
    np.savez(save_path, z=z, y=y, u=u)

def plot_embeddings(embedding_paths, titles, save_path):

    def outlier_mask(data, m=5.0):
        return jnp.all((data - jnp.median(data, keepdims=True))**2 < m*jnp.std(data, keepdims=True), axis=1)

    # Load the embeddings
    ys, us = [], []
    for path in embedding_paths:
        with np.load(path) as data:
            z, y, u = data['z'], data['y'], data['u']
            mask = outlier_mask(u)
            # ys.append(y[mask])
            # us.append(u[mask])
            ys.append(y)
            us.append(u)

            # df1 = pd.DataFrame()

    fig, axes = plt.subplots(1, 2); axes = axes.ravel()
    fig.set_size_inches(10, 5)

    us = [us[0], us[3]]
    ys = [ys[0], ys[3]]

    for i, (ax, u, y, title) in enumerate(zip(axes, us, ys, titles)):
        scatter = ax.scatter(*u.T, s=3.0, c=y, cmap='Spectral', alpha=0.6)
        ax.set_title(title, fontsize=20)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(axis='both', which='both',length=0)
        if(i == 0):
            ax.set_xlim(-5.5, 4.8)
            ax.set_ylim(5, 15)
        else:
            ax.set_xlim(-4, 7)
            ax.set_ylim(-6, 8)



    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    cbar = fig.colorbar(scatter, boundaries=jnp.arange(11) - 0.5)
    cbar.set_ticks(jnp.arange(10))
    cbar.ax.set_yticklabels(['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'])
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()

