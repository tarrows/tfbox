#!/usr/bin/env python
# -*- coding: utf-8 -*-

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
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
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'cifar-10-python.tar.gz',
                pbar.hook)

    if not isdir(CIFER10_DATASET_FOLDER_PATH):
        with tarfile.open('cifar-10-python.tar.gz') as tar:
            tar.extractall()
            tar.close()


def preprocess_cifar10():
    print('PREPROCESSING...')


def main():
    download_cifar10()
    preprocess_cifar10()


if __name__ == '__main__':
    main()
