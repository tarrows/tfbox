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
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch * 14 * 14)(g_input)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)
    H = Conv2D(int(nch / 2), (3, 3), padding='same')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(int(nch / 4), (3, 3), padding='same')(H)
    H = BatchNormalization()(H)
    H = Activation('relu')(H)
    H = Conv2D(1, (1, 1), padding='same')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input, g_V)


def model_discriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    nch = 512
    d_input = dim_ordering_input(input_shape, name="input_x")
    H = Conv2D(int(nch / 2), (5, 5), strides=(2, 2), padding='same',
               activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Conv2D(nch, (5, 5), strides=(2, 2), padding='same',
               activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(int(nch / 2))(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1, activation='sigmoid')
    return Model(d_input, d_V)


def mnist_process(x):
    return x.astype(np.float32) / 255.0


def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


if __name__ == '__main__':
    # z in R^100
    latent_dim = 100
    # x in R^{28 * 28}
    input_shape = (1, 28, 28)
    # generator (z -> x)
    generator = model_generator()
    # discriminator (x -> y)
    discriminator = model_discriminator(input_shape=input_shape)
    # gan (x -> yfake, yreal), z generated on GPU
    gan = simple_gan(generator, discriminator,
                     normal_latent_sampling(latent_dim,))
    # print summary of models
    generator.summary()
    discriminator.summary()
    gan.summary()

    # build adversarial model
    model = AdversarialModel(
      base_model=gan,
      player_params=[
        generator.trainable_weights, discriminator.trainable_weights
      ],
      player_names=["generator", "discriminator"]
    )
    model.adversarial_compile(
      adversarial_optimizer=AdversarialOptimizerSimultaneous(),
      player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
      loss="binary_crossentropy"
    )
