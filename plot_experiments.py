import argparse
from datasets import celeb_dataset_loader, cifar10_data_loader
from jax import random
from experiments import Experiment
import os
import yaml
import evaluate_experiments
import pathlib

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

    parser.add_argument('--results_root',
                        action='store',
                        type=str,
                        help='The root directory of the results folder',
                        default='Results')

    parser.add_argument('--compare_samples',
                        action='store_true',
                        help='')

    parser.add_argument('--reconstructions',
                        action='store_true',
                        help='')

    parser.add_argument('--vary_t',
                        action='store_true',
                        help='')

    parser.add_argument('--vary_s',
                        action='store_true',
                        help='')

    parser.add_argument('--compare_t',
                        action='store_true',
                        help='')

    parser.add_argument('--interpolations',
                        action='store_true',
                        help='')

    parser.add_argument('--best_s_for_nll',
                        action='store_true',
                        help='')

    parser.add_argument('--nll_comparison',
                        action='store_true',
                        help='')


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

    # Save the plots to the results folder
    folder_name = '_'.join(['%s_%d'%(exp.experiment_name, exp.current_iteration) for exp, _, _, _ in all_experiments])
    results_folder = os.path.join(args.results_root, folder_name)
    pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

    # Compare samples between the all of the models
    if(args.compare_samples):
        n_samples = 16
        key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'compare_samples.pdf')
        evaluate_experiments.compare_samples(key, all_experiments, n_samples, save_path)

    # Plot reconstructions for each of the models
    if(args.reconstructions):
        n_samples = 16
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)

        for exp, _, encoder, decoder in all_experiments:
            save_path = os.path.join(results_folder, 'reconstructions_%s.pdf'%exp.experiment_name)
            evaluate_experiments.reconstructions(data_key, key, exp.data_loader, encoder, decoder, save_path, n_samples, exp.quantize_level_bits)

    # Compare different temperature samples
    if(args.vary_t):
        n_samples = 16
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'vary_t.pdf')
        evaluate_experiments.samples_vary_t(data_key, key, all_experiments, n_samples, save_path, reuse_key=True)

    # Compare different sigma samples
    if(args.vary_s):
        n_samples = 16
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'vary_s.pdf')
        evaluate_experiments.samples_vary_s(data_key, key, all_experiments, n_samples, save_path, reuse_key=True)

    # Compare different temperature samples
    if(args.compare_t):
        n_samples = 16
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'compare_t.pdf')
        evaluate_experiments.compare_t(key, all_experiments, n_samples, save_path)

    # Plot some interpolations
    if(args.interpolations):
        n_pairs = 10
        n_interp = 10
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)

        for experiment in all_experiments:
            exp = experiment[0]
            save_path = os.path.join(results_folder, 'interpolation_%s.pdf'%exp.experiment_name)
            evaluate_experiments.interpolate_pairs(data_key, key, experiment, n_pairs, n_interp, save_path)

    # Compute the best value of s
    if(args.best_s_for_nll):
        key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'best_s_for_nll.yaml')
        evaluate_experiments.save_best_s_for_nll(key, all_experiments, save_path)

    # Compare the log likelihoods
    if(args.nll_comparison):
        key = random.PRNGKey(0)
        best_s_path = os.path.join(results_folder, 'best_s_for_nll.yaml')
        save_path = os.path.join(results_folder, 'nll_comparison.yaml')
        evaluate_experiments.validation_nll_from_best_s(key, all_experiments, best_s_path, save_path)
