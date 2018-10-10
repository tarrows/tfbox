#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Reshape
from keras.layers.convolutional import UpSampling2D
from keras.models import Sequential


def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_shape=(100, ), activation='tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization)
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(7 * 7 * 128,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same', activation='tanh',
                     data_format='channels_last'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same', activation='tanh',
                     data_format='channels_last'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1),
                     activation='tanh', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same',
                     activation='tanh', data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    return model
