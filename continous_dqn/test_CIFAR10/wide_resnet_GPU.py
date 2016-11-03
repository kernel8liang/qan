import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization, Dropout
from keras.engine import merge, Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy as accuracy
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
import keras.backend as K
import json
import time

nb_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reorder dimensions for tensorflow
x_train = np.transpose(x_train.astype('float32'), (0, 1, 2, 3))
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train)
x_train = (x_train - mean) / std
x_test = np.transpose(x_test.astype('float32'), (0, 1, 2, 3))
x_test = (x_test - mean) / std
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

sess = tf.Session()
K.set_session(sess)

def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


def residual_block(x, nb_filters=16, subsample_factor=1):
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample)(x)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x

    if nb_filters > prev_nb_channels:
        shortcut = Lambda(zero_pad_channels,
                          arguments={'pad': nb_filters - prev_nb_channels})(shortcut)

    y = BatchNormalization(axis=3)(x)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same')(y)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same')(y)

    out = merge([y, shortcut], mode='sum')

    return out

img_rows, img_cols = 32, 32
img_channels = 3

blocks_per_group = 4
widening_factor = 4

inputs = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, img_channels))

x = Convolution2D(16, 3, 3, init='he_normal', border_mode='same')(inputs)

for i in range(0, blocks_per_group):
    nb_filters = 16 * widening_factor
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

for i in range(0, blocks_per_group):
    nb_filters = 32 * widening_factor
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

for i in range(0, blocks_per_group):
    nb_filters = 64 * widening_factor
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid')(x)
x = Flatten()(x)

preds = Dense(nb_classes, activation='softmax')(x)

labels = tf.placeholder(tf.float32, shape=(None, 10))

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with sess.as_default():
    X_batch = x_train[:64]
    y_batch = y_train[:64]
    train_step.run(feed_dict={inputs: X_batch, labels: y_batch})


acc_value = accuracy(labels, preds)

with sess.as_default():
    print acc_value.eval(feed_dict={inputs: x_test, labels: y_test})

