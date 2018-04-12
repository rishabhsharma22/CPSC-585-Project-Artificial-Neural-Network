# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:51:54 2018

@author: Rishabh Sharma
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout,AveragePooling2D,GlobalAveragePooling2D,Reshape
import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


img_width,img_height = 224,224
train_data_dir = "dataset\Training_Data"
validation_data_dir = "dataset\Test_Data"
batch_size = 8
nb_validation_samples = 100

model = MobileNet(weights = 'imagenet', include_top=False, input_shape=(img_width,img_height,3) )

model.summary()

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
output = Dense(2, activation="softmax")(x)

model_final = Model(input = model.input, output = output)

model_final.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=["accuracy"])

train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

model_final.fit_generator(
train_generator,
samples_per_epoch = 1000,
epochs = 5,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples)


