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

if(__name__ == '__main__'):

    # Load the command line arguments
    parser = argparse.ArgumentParser(description='Plot comparisons between models')

    parser.add_argument('--names', nargs='+', help='Experiments to compare', required=True)

    parser.add_argument('--quantize',
                        action='store',
                        type=int,
                        help='The number of bits to use in quantization',
                        default=5)

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

    # Load all of the experiments
    all_experiments = []
    for name in args.names:
        exp = Experiment(name,
                         args.quantize,
                         None,
                         start_it=-1,
                         experiment_root=args.experiment_root)
        exp.load_experiment()
        sampler = exp.get_jitted_sampler()
        encoder = exp.get_jitted_forward()
        decoder = exp.get_jitted_inverse()
        all_experiments.append((exp, sampler, encoder, decoder))

    # The FID folder should already exist with TTUR in it
    fid_path = args.fid_root
    assert os.path.exists(fid_path)

    jax_key = random.PRNGKey(0)
    n_samples = 25000

    # Define the combinations of sigma and temperatures we will use
    # sigmas = np.linspace(0.0, 1.0, 30) # Will be used with temp of 1.0
    # temps  = np.linspace(0.0, 3.0, 30) # Will be used with sigma of 0.0

    # configurations = []
    # for s in sigmas:
    #     s = float(s)
    #     configurations.append((s, 1.0))
    # for t in temps:
    #     t = float(t)
    #     configurations.append((0.0, t))

    configurations = [(0.3, 1.0)]

    # Loop over the experiments
    for exp, sampler, encoder, decoder in all_experiments:
        initialize = False

        # Create the folder for the FID score
        experiment_fid_folder = os.path.join(fid_path, exp.experiment_name)
        if(os.path.exists(experiment_fid_folder) == False):
            initialize = True
        pathlib.Path(experiment_fid_folder).mkdir(parents=True, exist_ok=True)

        # Extract the meta data
        meta_path = os.path.join(experiment_fid_folder, 'meta.yaml')
        if(os.path.exists(meta_path) == False):
            initialize = True
        else:
            # If the iteration number has changed, then we need to re-initialize
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            if(('iteration_number' in meta == False) or (meta['iteration_number'] != exp.current_iteration)):
                initialize = True

        # Initialize if needed
        if(initialize):
            print('Initializing %s'%exp.experiment_name)

            # Create a file that will map a temperature/sigma setting with a folder of samples and the FID score
            meta = {}
            meta['iteration_number'] = exp.current_iteration
            meta['settings'] = []

            # Also map to the pre-computed stats of the dataset
            stats_name = exp.model.dataset_name+'_stats.npz'
            meta['stats'] = os.path.join(fid_path, stats_name)
            assert os.path.exists(meta['stats']), 'The statistics for the dataset %s do not exist'%exp.model.dataset_name

            for s, t in configurations:
                meta['settings'].append(dict(s=s, t=t, path=None, score=None))

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
