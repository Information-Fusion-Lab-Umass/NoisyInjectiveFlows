import numpy as np
import os
import tensorflow as tf
import tqdm
from tqdm import tqdm
import argparse
import yaml
# from TTUR.fid import calculate_fid_given_paths
from TTUR.fid import check_or_download_inception, _handle_path, calculate_frechet_distance, create_inception_graph

if(__name__ == '__main__'):

    # Load the command line arguments
    parser = argparse.ArgumentParser(description='Compute the FID scores using folders of images')
    parser.add_argument('--names', nargs='+', help='Experiments to compare', required=True)
    args = parser.parse_args()

    # Download the inception network
    inception_path = None
    inception_path = check_or_download_inception(inception_path)
    create_inception_graph(str(inception_path))

    results = {}

    # Start the tf session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Compute the FID scores
        for name in tqdm(args.names):
            results[name] = []

            # Load the settings file
            experiment_fid_folder = name
            meta_path = os.path.join(experiment_fid_folder, 'meta.yaml')
            assert os.path.exists(meta_path)

            with open(meta_path) as f:
                meta = yaml.safe_load(f)

            # Load the path to the stats
            stats_path = meta['stats'][4:]

            # Loop through all of the configurations
            all_settings = meta['settings']
            for i, setting in enumerate(tqdm(all_settings)):

                # If we've done this, continue
                if(setting['score'] is not None):
                    results[name].append(dict(s=setting['s'], t=setting['t'], score=setting['score']))
                    continue

                # Compute the FID score
                samples_path = setting['path'][4:]
                m1, s1 = _handle_path(samples_path, sess, low_profile=True)
                m2, s2 = _handle_path(stats_path, sess, low_profile=True)
                fid_score = calculate_frechet_distance(m1, s1, m2, s2)

                fid_score = float(fid_score)
                setting['score'] = fid_score
                results[name].append(dict(s=setting['s'], t=setting['t'], score=setting['score']))

                # Update the meta file
                meta['settings'][i] = setting
                with open(meta_path, 'w') as f:
                    yaml.dump(meta, f)



    print(results)