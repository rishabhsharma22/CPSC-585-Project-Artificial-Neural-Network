#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 21:23:04 2018

@author: anushree-ankola
"""

from keras.applications.resnet50 import ResNet50
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

conv_base = ResNet50(weights=None, include_top=False, input_shape=(197, 197, 3))

conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

conv_base.trainable = False

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.5,
                                   rotation_range = 180,
                                   fill_mode = 'nearest',
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/Training_Data',
                                                 target_size = (197,197),
                                                 batch_size = 32,
                                                 class_mode = 'binary',
                                                 shuffle = True)

test_set = test_datagen.flow_from_directory('dataset/Test_Data',
                                            target_size = (197, 197),
                                            batch_size = 32,
                                            shuffle = False,
                                            class_mode = 'binary')

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
loss='binary_crossentropy',
metrics=['acc'])

history = model.fit_generator(training_set,
                         steps_per_epoch = 500,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 100)


model.save("Anushree_model.h5", overwrite=True)

model.summary()