#!/usr/bin/env python
# -*- coding: utf-8 -*-

from urllib.request import urlretrieve
from os.path import isfile, isdir
from sklearn import preprocessing
from tqdm import tqdm
import numpy as np
import pickle
import tarfile

CIFER10_DATASET_FOLDER_PATH = 'cifar-10-batches-py'


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_cifar10():
    if not isfile('cifar-10-python.tar.gz'):
        print('Download CIFAR-10 python dataset')
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'cifar-10-python.tar.gz',
                pbar.hook)
    else:
        print('Skip Downloading')

    if not isdir(CIFER10_DATASET_FOLDER_PATH):
        print('Extract the dataset')
        with tarfile.open('cifar-10-python.tar.gz') as tar:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()
    else:
        print('Skip Extracting')


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(x)

    encodings = label_binarizer.transform(x)
    return encodings


def preprocess_cifar10():
    n_batches = 5
    valid_features = []
    valid_labels = []

    print('Preprocess batches of training data...')
    for batch_id in range(1, n_batches + 1):
        with open(CIFER10_DATASET_FOLDER_PATH + '/data_batch_' + str(batch_id), mode='rb') as f:
            batch = pickle.load(f, encoding='latin1')

        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']

        validation_count = int(len(features) * 0.1)

        features = features / 255
        labels = one_hot_encode(labels)

        dumpname = 'preprocess_batch_' + str(batch_id) + '.p'

        pickle.dump((features[:-validation_count], labels[:-validation_count]), open(dumpname, 'wb'))

        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    print('Save a portion of training batch for validation...')
    pickle.dump((np.array(valid_features), np.array(valid_labels)), open('preprocess_validation.p', 'wb'))

    print('Preprocess the training data...')
    with open(CIFER10_DATASET_FOLDER_PATH + '/test_batch', mode='rb') as f:
        test_batch = pickle.load(f, encoding='latin1')

    test_features = test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = test_batch['labels']

    test_features = test_features / 255
    test_labels = one_hot_encode(test_labels)

    pickle.dump((np.array(test_features), np.array(test_labels)), open('preprocess_training.p', 'wb'))


def main():
    download_cifar10()
    preprocess_cifar10()


if __name__ == '__main__':
    main()
