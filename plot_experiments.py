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

    parser.add_argument('--basic_samples',
                        action='store_true',
                        help='')

    parser.add_argument('--figure2',
                        action='store_true',
                        help='')

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

    parser.add_argument('--compare_reconstructions',
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

    parser.add_argument('--plot_embeddings',
                        action='store_true',
                        help='')

    parser.add_argument('--prob_diff',
                        action='store_true',
                        help='')

    args = parser.parse_args()

    # Load all of the experiments
    all_experiments = []
    results_folders = []
    for i, name in enumerate(args.names):
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

    # Basic plots for appendix
    if(args.basic_samples):
        nf = all_experiments[0]
        nif = all_experiments[1]
        nf_path = os.path.join(results_folder, 'basic_samples_nf.pdf')
        nif_s0_path = os.path.join(results_folder, 'basic_samples_nif_s0.pdf')
        nif_s1_path = os.path.join(results_folder, 'basic_samples_nif_s1.pdf')

        key = random.PRNGKey(0)
        n_rows = 8
        n_cols = 8

        evaluate_experiments.plot_samples(key, nf, nf_path, n_rows, n_cols, n_samples_per_batch=8, sigma=1.0)
        evaluate_experiments.plot_samples(key, nif, nif_s0_path, n_rows, n_cols, n_samples_per_batch=8, sigma=0.0)
        evaluate_experiments.plot_samples(key, nif, nif_s1_path, n_rows, n_cols, n_samples_per_batch=8, sigma=1.0)

    # Save Embeddings
    if(args.save_embedding):
        save_path = os.path.join(results_folder, 'embedding')
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)

        for experiment in all_experiments:
            name = experiment[0].experiment_name
            save_path = os.path.join(results_folder, '%s_test_embeddings.npz'%name)
            evaluate_experiments.save_test_embeddings(key, experiment, save_path, n_samples_per_batch=64)

    if(args.plot_embeddings):
        embedding_paths = []
        for experiment in all_experiments:
            name = experiment[0].experiment_name
            path = os.path.join(results_folder, '%s_test_embeddings.npz'%name)
            embedding_paths.append(path)
        save_path = os.path.join(results_folder, 'embedding_plot.pdf')
        assert all_experiments[3][0].experiment_name == 'cifar_64'
        titles = ['NF', 'NIF-64']
        evaluate_experiments.plot_embeddings(embedding_paths, titles, save_path)

    # Create figure 2
    if(args.figure2):
        n_samples = 8
        key = random.PRNGKey(0)
        keys = random.split(key, 10)
        nf_exp = all_experiments[0]
        assert nf_exp[0].is_nf == True, 'Need to pass the normalizing flow first'

        for nif_exp in all_experiments[1:]:
            for j, key in enumerate(keys):
                z_dim = nif_exp[0].model.z_shape[0]
                save_path = os.path.join(results_folder, 'dim_%d_figure_2_%d.pdf'%(z_dim, j))
                evaluate_experiments.figure_2_plots(key, nf_exp, nif_exp, n_samples, save_path)

    # Compare samples between a model and a baseline
    if(args.compare_baseline_samples):
        n_samples = 16
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
        n_samples = 12
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
    if(args.compare_reconstructions):
        n_samples = 12
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(3)

        save_path = os.path.join(results_folder, 'compare_reconstructions.pdf')
        evaluate_experiments.compare_reconstructions(data_key, key, all_experiments, save_path, n_samples, exp.quantize_level_bits)

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
        keys = random.split(key, 2)
        fixed_key = random.PRNGKey(0)

        for exp in all_experiments:
            if(exp[0].is_nf):
                continue
            for j, key in enumerate(keys):
                z_dim = exp[0].model.z_shape[0]
                save_path = os.path.join(results_folder, 'dim_%d_vary_s_%d.pdf'%(z_dim, j))
                evaluate_experiments.vary_s(key, fixed_key, exp, n_samples, save_path, reuse_key=True)

    # # Compare different sigma samples
    # if(args.vary_s):
    #     n_samples = 8
    #     key = random.PRNGKey(0)
    #     data_key = random.PRNGKey(0)
    #     save_path = os.path.join(results_folder, 'vary_s.pdf')
    #     evaluate_experiments.samples_vary_s(data_key, key, all_experiments, n_samples, save_path, reuse_key=True)

    # Compare different temperature samples
    if(args.compare_t):
        n_samples = 8
        key = random.PRNGKey(0)
        data_key = random.PRNGKey(0)
        glow = all_experiments[0]
        for exp in all_experiments[1:]:
            name = exp[0].experiment_name
            for i, key in enumerate(random.split(key, 6)):
                save_path = os.path.join(results_folder, '%s_compare_t_%d.pdf'%(name, i))
                evaluate_experiments.compare_t(key, [glow, exp], n_samples, save_path)

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

    if(args.prob_diff):
        key = random.PRNGKey(0)
        exp1, exp2 = all_experiments[0], all_experiments[1]
        save_path = os.path.join(results_folder, 'test.yaml')
        evaluate_experiments.save_probability_difference(key, exp1, exp2, save_path)