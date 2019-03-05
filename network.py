from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.layers import Activation,GlobalAveragePooling2D
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

batch_size = 128
num_classes = 2
epochs = 100
# input image dimensions
img_rows, img_cols = 64, 64

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')
x_test=np.load('x_test.npy')
y_test=np.load('y_test.npy')

x_train=x_train.transpose(0,2,3,1)
x_test=x_test.transpose(0,2,3,1)
if K.image_data_format() == 'channels_first':
    #x_train = x_train.reshape(x_train.shape[0], 95, img_rows, img_cols)
    #x_test = x_test.reshape(x_test.shape[0], 95, img_rows, img_cols)
    input_shape = (95, img_rows, img_cols)
else:
    #x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 95)
#input_shape = (img_rows, img_cols,95)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 2
x_test /= 2
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(128, (3,3),strides=(2,2),input_shape=input_shape))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3),strides=(1, 1)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3),strides=(2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3),strides=(1, 1)))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

# model.add(Conv2D(512, (3, 3),strides=(2, 2)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))
#
# model.add(Conv2D(512, (3, 3),strides=(1, 1)))
# model.add(BatchNormalization(axis=3))
# model.add(Activation('relu'))

model.add(GlobalAveragePooling2D())

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])