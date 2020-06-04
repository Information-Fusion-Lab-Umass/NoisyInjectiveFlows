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
                        default=3)

    parser.add_argument('--experiment_root',
                        action='store',
                        type=str,
                        help='The root directory of the experiments folder',
                        default='Experiments')

    parser.add_argument('--checkpoint',
                        action='store',
                        type=int,
                        help='What checkpoint to use',
                        default=-1)

    parser.add_argument('--results_root',
                        action='store',
                        type=str,
                        help='The root directory of the results folder',
                        default='Results')

    parser.add_argument('--compare_baseline_samples',
                        action='store_true',
                        help='')

    parser.add_argument('--compare_vertical',
                        action='store_true',
                        help='')

    parser.add_argument('--compare_samples',
                        action='store_true',
                        help='')

    parser.add_argument('--compare_full_samples',
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

    parser.add_argument('--manifold_penalty',
                        action='store_true',
                        help='')
    parser.add_argument('--save_embedding',
                        action='store_true',
                        help='')
    parser.add_argument('--print_embedding',
                        action='store_true',
                        help='')

    args = parser.parse_args()

    # Load all of the experiments
    all_experiments = []
    results_folders = []
    for name in args.names:
        exp = Experiment(name,
                         args.quantize,
                         None,
                         start_it=args.checkpoint,
                         experiment_root=args.experiment_root)
        exp.load_experiment()
        results_folders.append(os.path.join(args.results_root, '%s_%d'%(exp.experiment_name, exp.current_iteration)))

        sampler = exp.get_jitted_sampler()
        encoder = exp.get_jitted_forward()
        decoder = exp.get_jitted_inverse()
        all_experiments.append((exp, sampler, encoder, decoder))

    # Save the plots to the results folder
    folder_name = '_'.join(['%s_%d'%(exp.experiment_name, exp.current_iteration) for exp, _, _, _ in all_experiments])
    results_folder = os.path.join(args.results_root, folder_name)
    pathlib.Path(results_folder).mkdir(parents=True, exist_ok=True)

    # Save Embeddings
    if(args.save_embedding):
        save_path = os.path.join(results_folder, 'embedding')
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)

        for exp, _, encoder, decoder in all_experiments:
            data_loader = exp.data_loader
            print('starting_save_embeddings')
            embedding_path = os.path.join(save_path, str(exp.experiment_name))
            pathlib.Path(embedding_path).mkdir(parents=True, exist_ok=True)
            evaluate_experiments.save_embeddings(key, data_loader, encoder, embedding_path, test=True, n_samples_per_batch=4)
            print('test_done')
            evaluate_experiments.save_embeddings(key, data_loader, encoder, embedding_path, test=False, n_samples_per_batch=4)

    if(args.print_embedding):
        save_paths = [os.path.join(results_folders[0], 'embedding'), os.path.join(results_folders[1], 'embedding')]
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)
        exp1, _, encoder1, decoder1 = all_experiments[0]
        exp2, _, encoder2, decoder2 = all_experiments[1]

        data_loader = exp1.data_loader
        path_1 = os.path.join(save_paths[0], str(exp1.experiment_name))
        path_2 = os.path.join(save_paths[1], str(exp2.experiment_name))

        evaluate_experiments.print_reduced_embeddings(key, data_loader, encoder1, encoder2, path_1, path_2, results_folder, test=True, n_samples_per_batch=4)


    # Compare samples between a model and a baseline
    if(args.compare_baseline_samples):
        n_samples = 8
        key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'compare_baseline_samples.pdf')
        baseline_sampler = all_experiments[0][1]
        sampler = all_experiments[1][1]
        evaluate_experiments.compare_manifold_vs_full_samples(key, sampler, baseline_sampler, n_samples, save_path)

    # Compare samples between the all of the models vertically
    if(args.compare_vertical):
        n_samples = 3
        key = random.PRNGKey(1)
        save_path = os.path.join(results_folder, 'compare_vertical.pdf')
        evaluate_experiments.compare_vertical(key, all_experiments, n_samples, save_path)


    # Compare samples between the all of the models
    if(args.compare_samples):
        n_samples = 8
        key = random.PRNGKey(3)
        save_path = os.path.join(results_folder, 'compare_samples.pdf')
        evaluate_experiments.compare_samples(key, all_experiments, n_samples, save_path)

    # Compare samples between the all of the models
    if(args.compare_full_samples):
        n_samples = 8
        key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'compare_full_samples.pdf')
        evaluate_experiments.compare_samples(key, all_experiments, n_samples, save_path, sigma=1.0)

    # Plot reconstructions for each of the models
    if(args.reconstructions):
        n_samples = 8
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)

        for exp, _, encoder, decoder in all_experiments:
            save_path = os.path.join(results_folder, 'reconstructions_%s.pdf'%exp.experiment_name)
            evaluate_experiments.reconstructions(data_key, key, exp.data_loader, encoder, decoder, save_path, n_samples, exp.quantize_level_bits)

    # Compare different temperature samples
    if(args.vary_t):
        n_samples = 8
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'vary_t.pdf')
        evaluate_experiments.samples_vary_t(data_key, key, all_experiments, n_samples, save_path, reuse_key=True)

    # Compare different sigma samples
    if(args.vary_s):
        n_samples = 8
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)
        save_path = os.path.join(results_folder, 'vary_s.pdf')
        evaluate_experiments.samples_vary_s(data_key, key, all_experiments, n_samples, save_path, reuse_key=True)

    # Compare different temperature samples
    if(args.compare_t):
        n_samples = 8
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

    # See what images correspond to the varying manifold penalties
    if(args.manifold_penalty):
        key = random.PRNGKey(0)
        evaluate_experiments.manifold_penalty(key, all_experiments[0][0], None)