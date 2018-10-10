#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example from
bstriner/keras-adversarial/blob/master/examples/example_gan_convolutional.py
"""

import os

import matplotlib as mpl
import numpy as np
import pandas as pd
import keras.backend as K

from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from keras.optimizers import Adam

from keras_adversarial import AdversarialModel
from keras_adversarial import AdversarialOptimizerSimultaneous
from keras_adversarial import gan_targets
from keras_adversarial import normal_latent_sampling
from keras_adversarial import simple_gan
from keras_adversarial.image_grid_callback import ImageGridCallback

from image_utils import dim_ordering_fix
from image_utils import dim_ordering_input
from image_utils import dim_ordering_reshape
from image_utils import dim_ordering_unfix

# allows mpl to run with no DISPLAY defined
mpl.use("Agg")


def model_generator():
    return Model()


def model_discriminator():
    return Model()


def mnist_process(x):
    return x


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


if __name__ == '__main__':
    # z in R^100
    latent_dim = 100


