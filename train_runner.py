import argparse
from datasets import celeb_dataset_loader, cifar10_data_loader
from jax import random
from experiments import Experiment
import os
import yaml

if(__name__ == '__main__'):

    # Load the command line arguments
    parser = argparse.ArgumentParser(description='Train an NIF model')
    parser.add_argument('--name',
                        action='store',
                        type=str,
                        help='Name of model.  Is used to load existing checkpoints.',
                        default='GLOW')

    parser.add_argument('--dataset',
                        action='store',
                        type=str,
                        help='Dataset to load.',
                        default='CIFAR10')

    parser.add_argument('--quantize',
                        action='store',
                        type=int,
                        help='The number of bits to use in quantization',
                        default=5)

    parser.add_argument('--start_it',
                        action='store',
                        type=int,
                        help='Sets the training iteration to start on.  -1 finds most recent',
                        default=-1)

    parser.add_argument('--checkpoint_interval',
                        action='store',
                        type=int,
                        help='Sets the number of iterations between each test',
                        default=500)

    parser.add_argument('--experiment_root',
                        action='store',
                        type=str,
                        help='The root directory of the experiments folder',
                        default='Experiments')

    parser.add_argument('--experiment_def_path',
                        action='store',
                        type=str,
                        help='The root directory of the experiments definitions folder',
                        default='experiment_definitions')

    parser.add_argument('--optimizer_settings',
                        action='store',
                        type=str,
                        help='Settings for the optimizer',
                        default='adam')
    args = parser.parse_args()

    # Load the dataset
    data_key = random.PRNGKey(0)
    if(args.dataset == 'CelebA'):
        data_loader, x_shape = celeb_dataset_loader(data_key, quantize_level_bits=args.quantize, split=(0.6, 0.2, 0.2))
    elif(args.dataset == 'CIFAR10'):
        data_loader, x_shape = cifar10_data_loader(data_key, quantize_level_bits=args.quantize, split=(0.6, 0.2, 0.2))
    else:
        assert 0, 'Invalid dataset'

    # Load the experiment object
    exp = Experiment(args.name,
                     x_shape,
                     data_loader,
                     args.quantize,
                     args.checkpoint_interval,
                     start_it=args.start_it,
                     experiment_root=args.experiment_root)

    # Initialize the experiment from scratch or from a checkpoint
    if(exp.current_iteration is None):
        key = random.PRNGKey(0)
        # Load the model and optimizer definitions
        model_def_path = os.path.join(args.experiment_def_path, args.name+'.yaml')
        opt_def_path = os.path.join(args.experiment_def_path, args.optimizer_settings+'.yaml')

        with open(model_def_path) as f:
            model_meta_data = yaml.safe_load(f)
        model_meta_data['x_shape'] = x_shape

        with open(opt_def_path) as f:
            opt_meta_data = yaml.safe_load(f)

        # Initialize the experiment
        exp.create_experiment_from_meta_data(key, model_meta_data, opt_meta_data)
    else:
        exp.load_experiment()

    # Train
    exp.train()