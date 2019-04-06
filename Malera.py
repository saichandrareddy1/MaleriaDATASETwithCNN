#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:06:49 2019

@author: sai
"""

import keras 
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing import image
from keras.layers import BatchNormalization, Flatten
from keras.layers import ZeroPadding2D

#DataGenerator

train_gen = image.ImageDataGenerator(rescale = 1./255,
                                    featurewise_center=True, 
                                    samplewise_center=True, 
                                    featurewise_std_normalization=True, 
                                    samplewise_std_normalization=True, 
                                    zca_whitening=False, zca_epsilon=1e-06, 
                                    rotation_range=40, width_shift_range=0.0,
                                    height_shift_range=0.0, brightness_range=(0.2, 0.2),
                                    shear_range=0.2, zoom_range=0.2, 
                                    channel_shift_range=0.0, fill_mode='nearest', 
                                    cval=0.0, horizontal_flip=True, vertical_flip=True,
                                    data_format=None, validation_split=0.0, dtype=None)

test_gen = image.ImageDataGenerator(rescale = 1./255,
                                    featurewise_center=True, 
                                    samplewise_center=True, 
                                    featurewise_std_normalization=True, 
                                    samplewise_std_normalization=True, 
                                    zca_whitening=False, zca_epsilon=1e-06, 
                                    rotation_range=40, width_shift_range=0.0,
                                    height_shift_range=0.0, brightness_range=(0.2, 0.2),
                                    shear_range=0.2, zoom_range=0.2, 
                                    channel_shift_range=0.0, fill_mode='nearest', 
                                    cval=0.0, horizontal_flip=True, vertical_flip=True,
                                    data_format=None, validation_split=0.0, dtype=None)


val_gen = image.ImageDataGenerator(rescale = 1./255,
                                    featurewise_center=True, 
                                    samplewise_center=True, 
                                    featurewise_std_normalization=True, 
                                    samplewise_std_normalization=True, 
                                    zca_whitening=False, zca_epsilon=1e-06, 
                                    rotation_range=40, width_shift_range=0.0,
                                    height_shift_range=0.0, brightness_range=(0.2, 0.2),
                                    shear_range=0.2, zoom_range=0.2, 
                                    channel_shift_range=0.0, fill_mode='nearest', 
                                    cval=0.0, horizontal_flip=True, vertical_flip=True,
                                    data_format=None, validation_split=0.0, dtype=None)

#Data Train

data_train = train_gen.flow_from_directory("/home/sai/Documents/data/Cell_images/Train",
                                           target_size = (32, 32),
                                           classes = ['Parasitized', 'Uninfected'],
                                           class_mode = 'categorical',
                                           batch_size = 64,
                                           seed = 1)



#DataTest

data_test = test_gen.flow_from_directory("/home/sai/Documents/data/Cell_images/Test",
                                         target_size=(32, 32),
                                         classes = ['Parasitized', 'Uninfected'],
                                         class_mode='categorical',
                                         batch_size=64,
                                         seed=1)

#DataVal

data_val = val_gen.flow_from_directory("/home/sai/Documents/data/Cell_images/Val",
                                         target_size=(32, 32),
                                         classes = ['Parasitized', 'Uninfected'],
                                         class_mode='categorical',
                                         batch_size=64,
                                         seed=1)



#Model
model = Sequential()

#CONV1
model.add(ZeroPadding2D(padding = (2, 2), input_shape= (32, 32, 3)))
model.add(Conv2D(5, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                 activation =  'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding = 'SAME'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.3))

#CONV2
model.add(ZeroPadding2D(padding = (2, 2)))
model.add(Conv2D(5, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                 activation =  'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding = 'SAME'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.3))

#CONV3
model.add(ZeroPadding2D(padding = (2, 2)))
model.add(Conv2D(3, kernel_size=(2, 2), strides=(1, 1), padding='SAME',
                 activation =  'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding = 'SAME'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.3))

#CONV4
model.add(ZeroPadding2D(padding = (2, 2)))
model.add(Conv2D(3, kernel_size=(2, 2), strides=(1, 1), padding='SAME',
                 activation =  'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding = 'SAME'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

#CONV5
model.add(ZeroPadding2D(padding = (2, 2)))
model.add(Conv2D(3, kernel_size=(2, 2), strides=(1, 1), padding='SAME',
                 activation =  'relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding = 'SAME'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.5))

#FLATTEN
model.add(Flatten())

#Dense1
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.2))

#Dense2
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.3))

#Dense3
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization(axis = -1))
model.add(Dropout(0.5))

#Dense4
model.add(Dense(2, activation='sigmoid'))

#model summary
model.summary()

#Optimizers
Ad = Adam(lr = 0.01)
#Compile Model
model.compile(optimizer = Ad,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

#Fitting Model
history = model.fit_generator(data_train, steps_per_epoch=100,
                    epochs=30, validation_data=data_val,
                    validation_steps=30)


import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Evaluating Model
score  = model.evaluate_generator(data_test, 
                                  steps = 100)


#printAccuracy
print("Accuracyloss:-", score[0])
print("AccuracyScore",score[1] )


model.save("/home/sai/Documents/data/Cell_images/weights.h5")
