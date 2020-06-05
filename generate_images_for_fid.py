import argparse
from datasets import celeb_dataset_loader, cifar10_data_loader
from jax import random
from experiments import Experiment
import os
import yaml
import evaluate_experiments
import pathlib
import numpy as np
from tqdm import tqdm
import glob
import shutil

if(__name__ == '__main__'):

    # Load the command line arguments
    parser = argparse.ArgumentParser(description='Plot comparisons between models')

    parser.add_argument('--names', nargs='+', help='Experiments to compare', required=True)

    parser.add_argument('--quantize',
                        action='store',
                        type=int,
                        help='The number of bits to use in quantization',
                        default=3)

    parser.add_argument('--experiment_root',
                        action='store',
                        type=str,
                        help='The root directory of the experiments folder',
                        default='Experiments')

    parser.add_argument('--fid_root',
                        action='store',
                        type=str,
                        help='The root directory of the FID folder',
                        default='FID')

    args = parser.parse_args()

    # The FID folder should already exist with TTUR in it
    fid_path = args.fid_root
    assert os.path.exists(fid_path)

    jax_key = random.PRNGKey(0)
    n_samples = 25000

    # Define the combinations of sigma and temperatures we will use
    sigmas = np.linspace(0.0, 1.2, 15) # Will be used with temp of 1.0
    configurations = []
    for s in sigmas:
        s = float(s)
        configurations.append((s, 1.0))

    original_configurations = configurations

    # temps  = np.linspace(0.0, 3.0, 30) # Will be used with sigma of 0.0
    # for t in temps:
    #     t = float(t)
    #     configurations.append((0.0, t))

    # configurations = [(0.3, 1.0)]


    # In the meta file, we need:
    # - iteration number
    # - settings
    #    - path
    #    - s
    #    - t
    #    - score


    # Load all of the experiments
    for name in tqdm(args.names):
        exp = Experiment(name,
                         args.quantize,
                         None,
                         start_it=-1,
                         experiment_root=args.experiment_root)
        exp.load_experiment()
        sampler = exp.get_jitted_sampler()

        # Create the folder for the FID score
        experiment_fid_folder = os.path.join(fid_path, exp.experiment_name)
        pathlib.Path(experiment_fid_folder).mkdir(parents=True, exist_ok=True)

        # Extract the meta data
        meta_path = os.path.join(experiment_fid_folder, 'meta.yaml')
        if(os.path.exists(meta_path) == False):
            meta = {}
        else:
            # If the iteration number has changed, then we need to re-initialize
            with open(meta_path) as f:
                meta = yaml.safe_load(f)

        # If it doesn't have iteration number, or the iteration number is wrong, then reset
        it = meta.get('iteration_number', None)
        if(it != exp.current_iteration):
            meta['iteration_number'] = exp.current_iteration

        # Create the settings
        settings = meta.get('settings', None)
        if(settings is None):
            meta['settings'] = []

        # Also map to the pre-computed stats of the dataset.  This should never be different
        stats_name = exp.model.dataset_name+'_stats.npz'
        meta['stats'] = os.path.join(fid_path, stats_name)
        assert os.path.exists(meta['stats']), 'The statistics for the dataset %s do not exist'%exp.model.dataset_name

        # Loop through the configurations and assign them
        if(exp.is_nf):
            configurations = [(1.0, 1.0)] # Only need one here
        else:
            configurations = original_configurations

        for s, t in configurations:

            # If the path exists, it must contain the correct number of images
            for setting in meta['settings']:
                if(setting['s'] == s and setting['t'] == t):
                    path = setting['path']
                    n_images = len(glob.glob(path+'/*.jpg'))
                    if(n_images == n_samples):
                        continue
                    else:
                        # Delete the images folder
                        shutil.rmtree(path, ignore_errors=False, onerror=None)

            # Add a new configuration
            meta['settings'].append(dict(s=s, t=1.0, path=None, score=None))

        # Save the meta file
        with open(meta_path, 'w') as f:
            yaml.dump(meta, f)

        # Generate the images for the FID score
        settings = list(meta['settings'])
        for i, setting in enumerate(settings):
            if(setting['path'] is None):
                temp  = setting['t']
                sigma = setting['s']

                # Generate the images we'll need to compute the FID score
                images_folder = os.path.join(experiment_fid_folder, 'images_%d'%i)
                pathlib.Path(images_folder).mkdir(parents=True, exist_ok=True)
                evaluate_experiments.generate_images_for_fid(jax_key, sampler, temp, sigma, n_samples, images_folder)

                # Update the meta file
                meta['settings'][i] = dict(s=sigma, t=temp, path=images_folder, score=None)
                with open(meta_path, 'w') as f:
                    yaml.dump(meta, f)


        del exp
        del sampler
