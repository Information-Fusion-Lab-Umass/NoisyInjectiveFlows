import numpy as np
import os
import gzip, pickle
import tensorflow as tf
from imageio import imread
from scipy import linalg
import pathlib
import urllib
import warnings
import tqdm
from TTUR.fid import check_or_download_inception, _handle_path, calculate_frechet_distance, create_inception_graph

if(__name__ == '__main__'):
    images_path = 'fashion_mnist_images'
    assert os.path.exists(images_path), 'Need to run ../datasets.save_fashion_mnist_to_samples()'

    inception_path = None
    inception_path = check_or_download_inception(inception_path)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _handle_path(images_path, sess, low_profile=True, stats_path='FashionMNIST.npz')