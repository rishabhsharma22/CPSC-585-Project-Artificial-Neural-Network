"""
                          
    FileName:   cnn_model_aB9       
    Date:       4/26/18 4:18 PM   
    Author:     aB9           
                                
"""

# Import libraries
import os, shutil

from keras import models, optimizers
from keras import layers
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import numba

# Directories
dataset = 'dataset'
dataset_train = dataset + '/Training_Data'
dataset_test = dataset + '/Test_Data'
train_waterbodies = dataset_train + '/With_Water'
train_no_waterbodies = dataset_train + '/Without_Water'
test_waterbodies = dataset_test + '/With_Water'
test_no_waterbodies = dataset_test + '/Without_water'

# #Data Augmentation config
# datagen = ImageDataGenerator(rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode='nearest')

# Creating the model (Sequential)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the network
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Configure model for training
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# Adding Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=45, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(dataset_train, target_size=(64,64), batch_size=32, class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(dataset_test, target_size=(64,64), batch_size=32, class_mode='binary')

#Fitting the model
model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,validation_data=validation_generator, validation_steps=50)

model.save('cnn_model_aB9.h5')