#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune InceptionV3 on a new set of classes
https://keras.io/applications/
"""

# from keras import backend as K
from keras.application.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
# from keras.preprocessing import image

base_model = InceptionV3(weights="imagenet", include_top=False)

# removed layer
#   where m1 ... m4 = (None, 8, 8, n) for n in [320, 768, 768, 192]
# layer.name, layer.input_shape, layer_output.shape
# ("mixed10", [m1, m2, m3, m4], (None, 8, 8, 2048))
# ("avg_pool", (None, 8, 8, 2048), (None, 1, 1, 2048))
# ("flatten", (None, 1, 1, 2048), (None, 2048))
# ("predictions", (None, 2048), (None, 1000))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")
predictions = Dense(200, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# freeze all convolutional Inception V3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done after setting layers to non-trainable)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

# train the model on the new data for a few epochs
model.fit_generator(...)

# train the 2 inception blocks
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# recompile the model for these modifications to take effect
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss="categorical_crossentropy")

# train our model again (this time fine-tuning the top 2 inception blocks)
# alongside the top Dense layers
model.fit_generator(...)
