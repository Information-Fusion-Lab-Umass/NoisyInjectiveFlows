import zipfile
import glob
import tarfile
import array
import gzip
import os
from os import path
import struct
from six.moves.urllib.request import urlretrieve

from six.moves import cPickle as pickle
from imageio import imread
import platform

import numpy as np
import pandas as pd
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
from functools import partial

from jax import random, vmap, jit, value_and_grad
import jax.numpy as jnp
from tqdm import tqdm
import pathlib

def download_url(data_folder, filename, url):
    # language=rst
    """
    Download a url to a specified location

    :param data_folder: Target folder location.  Will be created if doesn't exist
    :param filename: What to name the file
    :param url: url to download
    """
    if(path.exists(data_folder) == False):
        os.makedirs(data_folder)

    out_file = path.join(data_folder, filename)
    if(path.isfile(out_file) == False):
        print('Downloading {} to {}'.format(url, data_folder))
        urlretrieve(url, out_file)
        print('Done.')

    return out_file

def parse_mnist_struct(filename, struct_format='>II'):
    # language=rst
    """
    Unpack the data in the mnist files

    :param filename: MNIST .gz filename
    :param struct_format: How to read the files
    """
    struct_size = struct.calcsize(struct_format)
    with gzip.open(filename, 'rb') as file:
        header = struct.unpack(struct_format, file.read(struct_size))
        return header, np.array(array.array("B", file.read()), dtype=np.uint8)

def download_mnist(data_folder, base_url):
    # language=rst
    """
    Get the raw mnist data

    :param data_folder: Where to download the data to
    :param base_url: Where to download the files from
    """
    mnist_filenames = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for filename in mnist_filenames:
        download_url(data_folder, filename, base_url + filename)

    (_, n_train_data, n_rows, n_cols), train_images = parse_mnist_struct(path.join(data_folder, "train-images-idx3-ubyte.gz"), struct_format='>IIII')
    (_, n_test_data, n_rows, n_cols), test_images = parse_mnist_struct(path.join(data_folder, "t10k-images-idx3-ubyte.gz"), struct_format='>IIII')
    train_images = train_images.reshape((n_train_data, n_rows, n_cols))
    test_images = test_images.reshape((n_test_data, n_rows, n_cols))

    _, train_labels = parse_mnist_struct(path.join(data_folder, "train-labels-idx1-ubyte.gz"), struct_format='>II')
    _, test_labels = parse_mnist_struct(path.join(data_folder, "t10k-labels-idx1-ubyte.gz"), struct_format='>II')

    return train_images, train_labels, test_images, test_labels

def get_mnist_data(quantize_level_bits=2, data_folder='data/mnist/', kind='digits'):
    # language=rst
    """
    Retrive an mnist dataset.  Either get the digits or fashion datasets.

    :param data_folder: Where to download the data to
    :param kind: Choice of dataset to retrieve
    """
    if(kind == 'digits'):
        base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    elif(kind == 'fashion'):
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

    # Download and get the raw dataset
    train_images, train_labels, test_images, test_labels = download_mnist(data_folder, base_url)

    # Add a dummy channel dimension
    train_images = train_images[...,None]
    test_images = test_images[...,None]

    # Turn the labels to one hot vectors
    train_labels = train_labels == np.arange(10)[:,None]
    test_labels = test_labels == np.arange(10)[:,None]

    train_labels = train_labels.astype(np.int32).T
    test_labels = test_labels.astype(np.int32).T

    # Quantize
    factor = 256/(2**quantize_level_bits)
    train_images = train_images//factor
    test_images = test_images//factor

    return train_images, train_labels, test_images, test_labels

def mnist_data_loader(key, quantize_level_bits=8, split=(0.6, 0.2, 0.2), data_folder='data/mnist/', kind='fashion'):
    # language=rst
    """
    Load the mnist dataset.
    :param data_folder: Where to download the data to
    """
    train_images, train_labels, test_images, test_labels = get_mnist_data(quantize_level_bits=quantize_level_bits, data_folder=data_folder, kind=kind)
    x_shape = train_images.shape[1:]

    # Turn the (train, test) split into a (train, test, validation) split
    images, labels = np.concatenate([train_images, test_images]), np.concatenate([train_labels, test_labels])
    images = images.astype(np.int32)
    total_examples = images.shape[0]

    assert sum(split) == 1.0
    n_train      = int(total_examples*split[0])
    remaining    = total_examples - n_train
    n_test       = int(total_examples*split[1])
    n_validation = total_examples - n_train - n_test

    train_images, train_labels = images[:n_train], labels[:n_train]
    test_images, test_labels = images[n_train:n_train + n_test], labels[n_train:n_train + n_test]
    validation_images, validation_labels = images[n_train + n_test:], labels[n_train + n_test:]

    # def data_loader(batch_shape, key=None, start=None, train=True, labels=False):
    def data_loader(batch_shape, key=None, start=None, split='train', return_labels=False, onehot=True, return_if_at_end=False):
        assert (key is None)^(start is None)
        at_end = False

        if(split == 'train'):
            images, labels = train_images, train_labels
        elif(split == 'test'):
            images, labels = test_images, test_labels
        elif(split == 'validation'):
            images, labels = validation_images, validation_labels
        else:
            assert 0, 'Invalid split name.  Choose from \'train\', \'test\' or \'validation\''

        # See if we want to take a random batch or not
        if(key is not None):
            batch_idx = random.randint(key, batch_shape, minval=0, maxval=images.shape[0])
        else:
            n_files = batch_shape[-1]
            batch_idx = start + jnp.arange(n_files)

            # Trim the batch indices so that we don't exceed the size of the dataset
            batch_idx = batch_idx[batch_idx < images.shape[0]]
            batch_shape = batch_shape[:-1] + (batch_idx.shape[0],)

            # Check if we're at the end of the dataset
            if(batch_idx.shape[0] < n_files):
                at_end = True

            batch_idx = np.broadcast_to(batch_idx, batch_shape)

        image_batch, label_batch = images[batch_idx,...], labels[batch_idx,...]
        if(onehot == False):
            label_batch = np.nonzero(label_batch)[-1].reshape(batch_shape)

        ret = (image_batch, label_batch) if return_labels else image_batch
        if(return_if_at_end):
            return ret, at_end
        return ret

    return data_loader, x_shape, (n_train, n_test, n_validation)

############################################################################################################################################################

def download_cifar10(data_folder, base_url):
    # language=rst
    """
    Get the raw cifar data

    :param data_folder: Where to download the data to
    :param base_url: Where to download the files from
    """
    # Download the cifar data
    filename = 'cifar-10-python.tar.gz'
    download_filename = download_url(data_folder, filename, base_url)

    # Extract the batches
    with tarfile.open(download_filename) as tar_file:
        tar_file.extractall(data_folder)

    # Remove the tar file
    os.remove(download_filename)

def load_cifar_batch(filename):
    # language=rst
    """
    Load a single batch of the cifar dataset

    :param filename: Where the batch is located
    """
    version = platform.python_version_tuple()
    py_version = version[0]
    assert py_version == '2' or py_version == '3', 'Invalid python version'
    with open(filename, 'rb') as f:
        # Load the data into a dictionary
        datadict = pickle.load(f) if py_version == '2' else pickle.load(f, encoding='latin1')
        images, labels = datadict['data'], datadict['labels']

        # Reshape the images so that the channel dim is at the end
        images = images.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1).astype(np.float32)

        # Turn the labels into onehot vectors
        labels = np.array(labels)
        return images, labels

def load_cifar10(batches_data_folder):
    # language=rst
    """
    Load a single batch of the cifar dataset

    :param filename: Where the batch is located
    """
    # Load the cifar training data batches
    xs, ys = [], []
    for batch_idx in range(1,6):
        filename = os.path.join(batches_data_folder, 'data_batch_%d'%batch_idx)
        images, labels = load_cifar_batch(filename)
        xs.append(images)
        ys.append(labels)
    train_images = np.concatenate(xs)
    train_labels = np.concatenate(ys) == np.arange(10)[:,None]

    # Load the test data
    test_images, test_labels = load_cifar_batch(os.path.join(batches_data_folder, 'test_batch'))
    test_labels = test_labels == np.arange(10)[:,None]

    train_labels = train_labels.astype(np.int32).T
    test_labels = test_labels.astype(np.int32).T
    return train_images, train_labels, test_images, test_labels

def get_cifar10_data(quantize_level_bits=2, data_folder='data/cifar10/'):
    # language=rst
    """
    Load the cifar 10 dataset.

    :param data_folder: Where to download the data to
    """
    cifar10_dir = os.path.join(data_folder, 'cifar-10-batches-py')

    if(os.path.exists(cifar10_dir) == False):
        # Download the cifar dataset
        cifar_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        download_cifar10(data_folder, cifar_url)

    # Load the raw cifar-10 data
    train_images, train_labels, test_images, test_labels = load_cifar10(cifar10_dir)

    # Quantize
    factor = 256/(2**quantize_level_bits)
    train_images = train_images//factor
    test_images = test_images//factor

    return train_images, train_labels, test_images, test_labels

def cifar10_data_loader(key, quantize_level_bits=8, split=(0.6, 0.2, 0.2), data_folder='data/cifar10/'):
    # language=rst
    """
    Load the cifar 10 dataset.
    :param data_folder: Where to download the data to
    """
    train_images, train_labels, test_images, test_labels = get_cifar10_data(quantize_level_bits=quantize_level_bits, data_folder=data_folder)
    x_shape = train_images.shape[1:]

    # Turn the (train, test) split into a (train, test, validation) split
    images, labels = np.concatenate([train_images, test_images]), np.concatenate([train_labels, test_labels])
    images = images.astype(np.int32)
    total_examples = images.shape[0]

    assert sum(split) == 1.0
    n_train      = int(total_examples*split[0])
    remaining    = total_examples - n_train
    n_test       = int(total_examples*split[1])
    n_validation = total_examples - n_train - n_test

    train_images, train_labels = images[:n_train], labels[:n_train]
    test_images, test_labels = images[n_train:n_train + n_test], labels[n_train:n_train + n_test]
    validation_images, validation_labels = images[n_train + n_test:], labels[n_train + n_test:]
    tpv_images, tpv_label = images[n_train:], labels[n_train:]

    # def data_loader(batch_shape, key=None, start=None, train=True, labels=False):
    def data_loader(batch_shape, key=None, start=None, split='train', return_labels=False, onehot=True, return_if_at_end=False):
        assert (key is None)^(start is None)
        at_end = False

        if(split == 'train'):
            images, labels = train_images, train_labels
        elif(split == 'test'):
            images, labels = test_images, test_labels
        elif(split == 'validation'):
            images, labels = validation_images, validation_labels
        elif(split == 'tpv'):
            images, labels = tpv_images, tpv_label
        else:
            assert 0, 'Invalid split name.  Choose from \'train\', \'test\' or \'validation\''

        # See if we want to take a random batch or not
        if(key is not None):
            batch_idx = random.randint(key, batch_shape, minval=0, maxval=images.shape[0])
        else:
            n_files = batch_shape[-1]
            batch_idx = start + jnp.arange(n_files)

            # Trim the batch indices so that we don't exceed the size of the dataset
            batch_idx = batch_idx[batch_idx < images.shape[0]]
            batch_shape = batch_shape[:-1] + (batch_idx.shape[0],)

            # Check if we're at the end of the dataset
            if(batch_idx.shape[0] < n_files):
                at_end = True

            batch_idx = np.broadcast_to(batch_idx, batch_shape)

        image_batch, label_batch = images[batch_idx,...], labels[batch_idx,...]
        if(onehot == False):
            label_batch = np.nonzero(label_batch)[-1].reshape(batch_shape)


        ret = (image_batch, label_batch) if return_labels else image_batch
        if(return_if_at_end):
            return ret, at_end

        return ret

    return data_loader, x_shape, (n_train, n_test, n_validation)

############################################################################################################################################################

def get_celeb_dataset(quantize_level_bits=8, strides=(5, 5), crop=(12, 4), n_images=20000, data_folder='data/img_align_celeba/'):
    # language=rst
    """
    Load the celeb A dataset.

    :param data_folder: Where to download the data to
    """
    celeb_dir = data_folder

    if(os.path.exists(celeb_dir) == False):
        assert 0, 'Need to manually download the celeb-A dataset.  Download the zip file from here: %s'%('https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM')

    all_files = glob.glob('%s*.jpg'%celeb_dir)
    all_files = all_files[:n_images]

    quantize_factor = 256/(2**quantize_level_bits)

    images = []
    for path in tqdm(all_files):
        im = plt.imread(path, format='jpg')
        im = im[::strides[0],::strides[1]][crop[0]:,crop[1]:]
        images.append(im//quantize_factor)

    np_images = np.array(images, dtype=np.int32)

    return np_images

############################################################################################################################################################

def celeb_dataset_loader(key,
                         quantize_level_bits=8,
                         strides=(2, 2),
                         crop=((26, -19), (12, -13)),
                         n_images=-1,
                         split=(0.6, 0.2, 0.2),
                         data_folder='data/img_align_celeba/'):
    # language=rst
    """
    Load the celeb A dataset.

    :param data_folder: Where the data exists.  We expect the user to manually download this!!!
    """
    celeb_dir = data_folder
    all_files = glob.glob('%s*.jpg'%celeb_dir)

    # Sort these the images
    all_files = sorted(all_files, key=lambda x: int(os.path.split(x)[1].strip('.jpg')))

    if(n_images == -1 or n_images is None):
        n_images = len(all_files)

    # Load the file paths
    all_files = all_files[:n_images]
    total_files = len(all_files)
    quantize_factor = 256/(2**quantize_level_bits)
    all_files = np.array(all_files)

    # Load a single file so that we can get the shape
    im = plt.imread(all_files[0], format='jpg')
    im = im[::strides[0],::strides[1]][crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
    x_shape = jnp.array(im).shape

    # Generate the indices of the training, test and validation sets
    file_indices = jnp.arange(total_files)
    shuffled_file_indices = random.permutation(key, file_indices)

    assert sum(split) == 1.0
    n_train      = int(total_files*split[0])
    remaining    = total_files - n_train
    n_test       = int(total_files*split[1])
    n_validation = total_files - n_train - n_test

    train_indices      = shuffled_file_indices[:n_train]
    test_indices       = shuffled_file_indices[n_train:n_train + n_test]
    validation_indices = shuffled_file_indices[n_train + n_test:]

    # The loader will be used to pull batches of data
    def data_loader(batch_shape, key=None, start=None, split='train', return_if_at_end=False):
        assert (key is None)^(start is None)
        at_end = False

        # We have the option to choose the index of the images
        if(key is None):
            data_indices = file_indices
            n_files = batch_shape[-1]
            batch_idx = start + jnp.arange(n_files)

            # Trim the batch indices so that we don't exceed the size of the dataset
            batch_idx = batch_idx[batch_idx < validation_indices.shape[0]]
            batch_shape = batch_shape[:-1] + (batch_idx.shape[0],)

            # Check if we're at the end of the dataset
            if(batch_idx.shape[0] < n_files):
                at_end = True

            batch_idx = np.broadcast_to(batch_idx, batch_shape)
        else:
            if(split == 'train'):
                data_indices = train_indices
            elif(split == 'test'):
                data_indices = test_indices
            elif(split == 'validation'):
                data_indices = validation_indices
            else:
                assert 0, 'Invalid split name.  Choose from \'train\', \'test\' or \'validation\''

            batch_idx = random.randint(key, batch_shape, minval=0, maxval=data_indices.shape[0])
            batch_idx = data_indices[batch_idx]

        batch_files = all_files[np.array(batch_idx)]

        images = np.zeros(batch_shape + x_shape, dtype=np.int32).reshape((-1,) + x_shape)
        for k, path in enumerate(batch_files.ravel()):
            im = plt.imread(path, format='jpg')
            im = im[::strides[0],::strides[1]][crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
            im = im//quantize_factor

            images[k] = im

        ret = images.reshape(batch_shape + x_shape)
        if(return_if_at_end):
            return ret, at_end
        return ret

    return data_loader, x_shape, (n_train, n_test, n_validation)

############################################################################################################################################################

def get_BSDS300_data(quantize_level_bits=8, strides=(1, 1), crop=(0, 0), data_folder='/tmp/BSDS300/'):
    # language=rst
    """
    Load the BSDS300 dataset.

    :param data_folder: Where to download the data to
    """

    filename = 'BSDS300-images'
    full_filename = os.path.join(data_folder, filename)
    if(os.path.exists(full_filename) == False):
        bsds300_url = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz'
        download_filename = download_url(data_folder, filename, bsds300_url)
        assert full_filename == download_filename

        # Extract the batches
        with tarfile.open(download_filename) as tar_file:
            tar_file.extractall(data_folder)

    # Find the files from the folder
    train_images = glob.glob(data_folder + 'BSDS300/images/train/*.jpg')
    test_images = glob.glob(data_folder + 'BSDS300/images/test/*.jpg')

    quantize_factor = 256/(2**quantize_level_bits)

    # Load the files
    images = []
    shape = None
    for path in tqdm(train_images):
        im = plt.imread(path, format='jpg')
        if(shape is None):
            shape = im.shape
        if(im.shape != shape):
            im = im.transpose((1, 0, 2))
        im = im[::strides[0],::strides[1]][crop[0]:,crop[1]:]
        images.append(im//quantize_factor)

    train_images = np.array(images, dtype=np.int32)

    images = []
    for path in tqdm(test_images):
        im = plt.imread(path, format='jpg')
        if(im.shape != shape):
            im = im.transpose((1, 0, 2))
        im = im[::strides[0],::strides[1]][crop[0]:,crop[1]:]
        images.append(im//quantize_factor)

    test_images = np.array(images, dtype=np.int32)

    return train_images, test_images

############################################################################################################################################################

def make_train_test_split(x, percentage):
    n_train = int(x.shape[0]*percentage)
    np.random.shuffle(x)
    return x[n_train:], x[:n_train]

def decorrelate_data(data, threshold=0.98):
    # language=rst
    """
    Drop highly correlated columns.
    Adapted from NSF repo https://github.com/bayesiains/nsf/blob/master/data/gas.py

    :param threshold: Threshold where columns are considered correlated
    """
    # Find the correlation between each column
    col_correlation = np.sum(data.corr() > threshold, axis=1)

    while np.any(col_correlation > 1):
        # Remove columns that are highly correlated with more than 1 other column
        col_to_remove = np.where(col_correlation > 1)[0][0]
        col_name = data.columns[col_to_remove]
        data.drop(col_name, axis=1, inplace=True)

        # Find the correlation again
        col_correlation = np.sum(data.corr() > threshold, axis=1)
    return data

def remove_outliers_in_df(df, max_z_score=2):
    z_scores = scipy.stats.zscore(df)
    return df[np.all(np.abs(z_scores) < max_z_score, axis=1)]

def whiten_data(x):
    U, _, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, VT)

############################################################################################################################################################

def get_gas_data(train_test_split=True, decorrelate=True, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, co_only=True, data_folder='/tmp/gas/', **kwargs):
    # language=rst
    """
    Load the gas dataset.  Adapted from NSF repo https://github.com/bayesiains/nsf/tree/master/data

    :param data_folder: Where to download the data to
    """
    filename = 'data.zip'
    full_filename = os.path.join(data_folder, filename)

    # Download the dataset is we haven't already
    if(os.path.exists(full_filename) == False):
        gas_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00322/data.zip'
        download_filename = download_url(data_folder, filename, gas_url)
        assert full_filename == download_filename

        with zipfile.ZipFile(full_filename, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

    # Load the datasets
    co_data = pd.read_csv(os.path.join(data_folder, 'ethylene_CO.txt'), delim_whitespace=True, header=None, skiprows=1)
    methane_data = pd.read_csv(os.path.join(data_folder, 'ethylene_methane.txt'), delim_whitespace=True, header=None, skiprows=1)

    # 0 is time, 1 and 2 are labels
    co_data.drop(columns=[0, 1, 2], inplace=True)
    methane_data.drop(columns=[0, 1, 2], inplace=True)

    # Turn everything to numeric values
    co_data = co_data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)
    methane_data = methane_data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)

    # Remove columns that are highly correlated
    if(decorrelate == True):
        threshold = kwargs.get('threshold', 0.98)
        co_data = decorrelate_data(co_data, threshold=threshold)
        methane_data = decorrelate_data(methane_data, threshold=threshold)

    # Normalize the data.  If we're going to use dequantization, don't do this.  Instead
    # seed an actnorm layer with the mean and std.
    if(normalize):
        co_data = (co_data - co_data.mean())/co_data.std()
        methane_data = (methane_data - methane_data.mean())/methane_data.std()

    # Column 4 is really weird
    co_data.drop(columns=[4], inplace=True)

    # The data only contains 2 decimals, so do uniform dequantization
    co_dequantization_scale = np.ones(co_data.shape[1])*0.01*0.0
    methane_dequantization_scale = np.ones(methane_data.shape[1])*0.01*0.0

    # Remove outliers
    if(remove_outliers):
        remove_outliers_in_df(co_data)

    # Switch from pandas to numpy
    co_data = co_data.to_numpy(dtype=np.float32)
    methane_data = methane_data.to_numpy(dtype=np.float32)

    # Whiten the data
    if(whiten):
        co_data = whiten_data(co_data)

    # Train test split
    if(train_test_split):
        train_percentage = kwargs.get('train_percentage', 0.7)
        co_data = make_train_test_split(co_data, train_percentage)
        methane_data = make_train_test_split(methane_data, train_percentage)

    # Only return the co data
    if(co_only == True):
        data = co_data
        dequant = co_dequantization_scale
    else:
        data = co_data, methane_data
        dequant = co_dequantization_scale, methane_dequantization_scale

    if(return_dequantize_scale):
        return data, dequant
    return data

############################################################################################################################################################

def get_miniboone_data(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, data_folder='/tmp/miniboone', **kwargs):
    # language=rst
    """
    Load the miniboone dataset.  No dequantization is needed here, they use a lot of decimals.

    :param data_folder: Where to download the data to
    """
    filename = 'MiniBooNE_PID.txt'
    full_filename = os.path.join(data_folder, filename)

    # Download the dataset if we haven't already
    if(os.path.exists(full_filename) == False):
        miniboone_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00199/MiniBooNE_PID.txt'
        download_filename = download_url(data_folder, filename, miniboone_data_url)
        assert full_filename == download_filename

    # Load the dataset
    data = pd.read_csv(full_filename, delim_whitespace=True, header=None, skiprows=1)

    # Remove some invalid entries
    data.drop(data[data[0] < -100].index, inplace=True)

    # Turn everything to numeric values
    data = data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)

    # Remove columns that are highly correlated
    if(decorrelate == True):
        threshold = kwargs.get('threshold', 0.99)
        data = decorrelate_data(data, threshold=threshold)

    # Remove outliers
    if(remove_outliers):
        remove_outliers_in_df(data)

    # Normalize the data.  If we're going to use dequantization, don't do this.  Instead
    # seed an actnorm layer with the mean and std.
    if(normalize == True):
        data = (data - data.mean())/data.std()

    # Switch from pandas to numpy
    data = data.to_numpy(dtype=np.float32)

    # Whiten the data
    if(whiten):
        data = whiten_data(data)

    # Train test split
    if(train_test_split):
        train_percentage = kwargs.get('train_percentage', 0.7)
        data = make_train_test_split(data, train_percentage)

    # For consistency, return a dummy dequantization array
    if(return_dequantize_scale):
        n_cols = data[0].shape[1] if train_test_split else data.shape[1]
        return data, np.zeros(n_cols)

    return data

############################################################################################################################################################

def get_power_data(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, data_folder='/tmp/power/', **kwargs):
    # language=rst
    """
    Load the power dataset.

    :param data_folder: Where to download the data to
    """
    filename = 'household_power_consumption.zip'
    full_filename = os.path.join(data_folder, filename)

    # Download the dataset if we haven't already
    if(os.path.exists(full_filename) == False):
        power_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip'
        download_filename = download_url(data_folder, filename, power_url)
        assert full_filename == download_filename

        with zipfile.ZipFile(full_filename, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

    # Load the dataset
    data = pd.read_csv(full_filename, sep=';')

    # Combine the time stamp into a single time
    time_ns = pd.to_datetime(data['Time']).astype(np.int64)
    date_ns = pd.to_datetime(data['Date']).astype(np.int64)
    data = data.drop(columns=['Time', 'Date'])
    data['Time'] = (date_ns + time_ns - time_ns[0])/1e16

    # Turn everything to numeric values
    data = data.apply(partial(pd.to_numeric, errors='coerce')).dropna(axis=0)

    # Remove columns that are highly correlated
    if(decorrelate == True):
        threshold = kwargs.get('threshold', 0.99)
        data = decorrelate_data(data, threshold=threshold)

    # Remove outliers
    if(remove_outliers):
        remove_outliers_in_df(data)

    # We have different dequantization scales
    dequantize_scale = np.array([0.001, 0.001, 0.01, 0.1, 1.0, 1.0, 1.0, 0.0])*0.0

    # Normalize the data.  If we're going to use dequantization, don't do this.  Instead
    # seed an actnorm layer with the mean and std.
    if(normalize == True):
        data = (data - data.mean())/data.std()

    # Switch from pandas to numpy
    data = data.to_numpy(dtype=np.float32)

    # Whiten the data
    if(whiten):
        data = whiten_data(data)

    # Train test split
    if(train_test_split):
        train_percentage = kwargs.get('train_percentage', 0.7)
        data = make_train_test_split(data, train_percentage)

    if(return_dequantize_scale):
        return data, dequantize_scale
    return data

############################################################################################################################################################

def get_hepmass_data(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=True, whiten=True, retrieve_files=['1000_train', '1000_test'], data_folder='/tmp/hepmass/'):
    # language=rst
    """
    Load the HEPMASS dataset.

    :param data_folder: Where to download the data to
    """
    # There are a bunch of files in the hepmass dataset.  Only want some of them
    all_filenames = ['1000_test.csv.gz',
                     '1000_train.csv.gz',
                     'all_test.csv.gz',
                     'all_train.csv.gz',
                     'not1000_test.csv.gz',
                     'not1000_train.csv.gz']

    filenames = []
    for fname in all_filenames:
        for ret in retrieve_files:
            if(fname.startswith(ret)):
                filenames.append(fname)

    # Get each dataset
    data_dict = {}
    for filename in filenames:
        full_filename = os.path.join(data_folder, filename)

        if(os.path.exists(full_filename) == False):
            # Download the cifar dataset
            hepmass_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00347/%s'%filename
            download_filename = download_url(data_folder, filename, hepmass_url)
            assert full_filename == download_filename

        data_dict[filename.strip('.csv.gz')] = pd.read_csv(full_filename, compression='gzip')

    # Assumes we have chosen only 1 kind of dataset
    if(train_test_split):
        assert len(filenames) == 2
        train_file = [fname for fname in retrieve_files if 'train' in fname][0]
        test_file = [fname for fname in retrieve_files if 'test' in fname][0]
        train_data, test_data = data_dict[train_file], data_dict[test_file]

        # The train data is columns are messed up!
        train_data = train_data.reset_index()
        train_data.columns = test_data.columns

        # Remove the data associated with background noise
        train_data = train_data[train_data['# label'] == 1.0]
        test_data = test_data[test_data['# label'] == 1.0]

        train_data.drop(columns='# label', inplace=True)
        test_data.drop(columns='# label', inplace=True)

        # Remove outliers
        if(remove_outliers):
            remove_outliers_in_df(train_data)

        # We don't have to dequantize
        dequantize_scale = np.zeros(train_data.shape[1])

        # Normalize the data
        if(normalize):
            train_data = (train_data - train_data.mean())/train_data.std()
            test_data = (test_data - test_data.mean())/test_data.std()

        # Switch from pandas to numpy
        train_data = train_data.to_numpy(dtype=np.float32)
        test_data = test_data.to_numpy(dtype=np.float32)

        # Whiten the data
        if(whiten):
            whitened_data = whiten_data(np.concatenate([train_data, test_data], axis=0))
            train_data, test_data = np.split(whitened_data, np.array([train_data.shape[0]]), axis=0)

        if(return_dequantize_scale):
            return (train_data, test_data), dequantize_scale
        return train_data, test_data

    return data_dict

############################################################################################################################################################

def uci_loader(datasets=['hepmass', 'gas', 'miniboone', 'power'], data_root='data/'):
    kwargs = dict(train_test_split=True, decorrelate=False, normalize=False, return_dequantize_scale=True, remove_outliers=False, whiten=True)

    for d in datasets:
        if(d == 'hepmass'):
            data_folder = os.path.join(data_root, 'hepmass')
            (hepmass_train_data, hepmass_test_data), hepmass_noise_scale = get_hepmass_data(data_folder=data_folder, **kwargs)
            yield hepmass_train_data, hepmass_test_data, hepmass_noise_scale, 'hepmass'
            del hepmass_train_data
            del hepmass_test_data
        elif(d == 'gas'):
            data_folder = os.path.join(data_root, 'gas')
            (gas_train_data, gas_test_data), gas_noise_scale = get_gas_data(data_folder=data_folder, **kwargs)
            yield gas_train_data, gas_test_data, gas_noise_scale, 'gas'
            del gas_train_data
            del gas_test_data
        elif(d == 'miniboone'):
            data_folder = os.path.join(data_root, 'miniboone')
            (miniboone_train_data, miniboone_test_data), miniboone_noise_scale = get_miniboone_data(data_folder=data_folder, **kwargs)
            yield miniboone_train_data, miniboone_test_data, miniboone_noise_scale, 'miniboone'
            del miniboone_train_data
            del miniboone_test_data
        elif(d == 'power'):
            data_folder = os.path.join(data_root, 'power')
            (power_train_data, power_test_data), power_noise_scale = get_power_data(data_folder=data_folder, **kwargs)
            yield power_train_data, power_test_data, power_noise_scale, 'power'
            del power_train_data
            del power_test_data

############################################################################################################################################################

def STL10_dataset_loader(quantize_level_bits=8, strides=(1, 1), crop=(0, 0), data_folder='data/STL10/'):
    # language=rst
    """
    Load the STL10 dataset.

    :param data_folder: Where to download the data to
    """
    filename = 'STL10-images'
    full_filename = os.path.join(data_folder, filename)
    if(os.path.exists(full_filename) == False):
        bsds300_url =  'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
        download_filename = download_url(data_folder, filename, bsds300_url)
        assert full_filename == download_filename

        # Extract the batches
        with tarfile.open(download_filename) as tar_file:
            tar_file.extractall(data_folder)

    quantize_factor = 256/(2**quantize_level_bits)

    data_file = os.path.join(data_folder, 'stl10_binary/unlabeled_X.bin')
    data = np.fromfile(data_file, dtype=jnp.uint8)
    data = data.reshape((-1, 3, 96, 96)).transpose((0, 3, 2, 1))
    data = data[:,::strides[0],::strides[1]][crop[0]:,crop[1]:]

    x_shape = data.shape[1:]

    def data_loader(key, n_gpus, batch_size):
        batch_idx = random.randint(key, (n_gpus, batch_size), minval=0, maxval=data.shape[0])
        return data[batch_idx,...]//quantize_factor

    return data_loader, x_shape

############################################################################################################################################################

def save_fashion_mnist_to_samples():
    train_images, _, test_images, _ = get_mnist_data(quantize_level_bits=8, data_folder='data/mnist/', kind='fashion')
    images = np.concatenate([train_images, test_images])

    # Make sure we've created the image folder
    save_folder = 'FID/fashion_mnist_images/'
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Convert to jpg files
    for i, im in enumerate(images):
        path = os.path.join(save_folder, '%s.jpg'%i)
        matplotlib.image.imsave(path, im[:,:,0])

############################################################################################################################################################

def save_train_splits_for_fid(key=random.PRNGKey(0), dataset='CelebA', quantize_level_bits=8, split_percentage=0.1, n_splits=1, save_folder='FID/celeba_splits'):
    if(dataset == 'CelebA'):
        dataset_getter = celeb_dataset_loader
    elif(dataset == 'CIFAR10'):
        dataset_getter = cifar10_data_loader
    elif(dataset == 'FashionMNIST'):
        dataset_getter = partial(mnist_data_loader, kind='fashion')
    else:
        assert 0, 'Invalid dataset'

    data_loader, x_shape, (n_train, n_test, n_validation) = dataset_getter(key, quantize_level_bits)
    split_size = int(n_train*split_percentage)

    # Create the folder to save to
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Going to make 5 random batches from the training data
    keys = random.split(key, n_splits)
    for split_index, key in tqdm(list(enumerate(keys))):

        # Make a folder for the split
        split_folder = os.path.join(save_folder, 'split_%d'%split_index)
        pathlib.Path(split_folder).mkdir(parents=True, exist_ok=True)

        x = data_loader((split_size,), key=key)

        # Convert to jpg files
        for i, im in tqdm(list(enumerate(x))):
            path = os.path.join(split_folder, '%s.jpg'%i)
            im = im[:,:,0] if im.shape[-1] == 1 else im
            matplotlib.image.imsave(path, im/(2.0**quantize_level_bits))

############################################################################################################################################################

def save_test_for_fid(key=random.PRNGKey(0), dataset='FashionMNIST', quantize_level_bits=8, save_folder_name='test_images'):
    if(dataset == 'CelebA'):
        dataset_getter = celeb_dataset_loader
    elif(dataset == 'CIFAR10'):
        dataset_getter = cifar10_data_loader
    elif(dataset == 'FashionMNIST'):
        dataset_getter = partial(mnist_data_loader, kind='fashion')
    else:
        assert 0, 'Invalid dataset'

    data_loader, x_shape, (n_train, n_test, n_validation) = dataset_getter(key, quantize_level_bits)

    # Create the folder to save to
    save_folder = 'FID/%s_%s'%(dataset, save_folder_name)
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)

    x_test = data_loader((n_test,), start=0, split='test')

    # Convert to jpg files
    for i, im in tqdm(list(enumerate(x_test))):
        path = os.path.join(save_folder, '%s.jpg'%i)
        im = im[:,:,0] if im.shape[-1] == 1 else im
        matplotlib.image.imsave(path, im/(2.0**quantize_level_bits))

    del x_test
    x_validation = data_loader((n_validation,), start=0, split='validation')

    for i, im in tqdm(list(enumerate(x_validation))):
        path = os.path.join(save_folder, '%s.jpg'%(i + n_test))
        im = im[:,:,0] if im.shape[-1] == 1 else im
        matplotlib.image.imsave(path, im/(2.0**quantize_level_bits))
