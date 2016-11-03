# This is the CNN we want to run and change our LR continuously
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras.engine import merge, Input, Model
import math
from scipy import stats
import tensorflow as tf

class CNNEnv:
    def __init__(self):
        # Basic information on run
        self.nb_epoch = 3
        self.batch_size = 10000

        # Basic information on CNN
        self.nb_classes = 10
        self.img_rows, self.img_cols = 32, 32
        self.img_channels = 3

        # Other information for step function
        self.losses = []
        self.log_sum = 0
        self.reward = 0

        # Random seed for reproducibility
        seed = 1337
        np.random.seed(seed)

        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()

        # TensorFlow dimension ordering issue fix
        self.X_train = np.transpose(self.X_train.astype('float32'), (0, 1, 2, 3))
        mean = np.mean(self.X_train, axis=0, keepdims=True)
        std = np.std(self.X_train)
        self.X_train = (self.X_train - mean) / std
        self.X_test = np.transpose(self.X_test.astype('float32'), (0, 1, 2, 3))
        self.X_test = (self.X_test - mean) / std

        # One-hot-encode outputs
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        # self.num_classes = self.y_test.shape[1]

        # Total iterations we want to run
        self.tot_iter = self.X_train.shape[0] / self.batch_size

    def reset(self, episodes):
        # Get weight features before training
        self.episodes = episodes

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
                                  arguments={
                                      'pad': nb_filters - prev_nb_channels})(
                    shortcut)

            y = BatchNormalization(axis=3)(x)
            y = Activation('relu')(y)
            y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                              init='he_normal', border_mode='same')(
                y)
            y = BatchNormalization(axis=3)(y)
            y = Activation('relu')(y)
            y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                              init='he_normal', border_mode='same')(
                y)

            out = merge([y, shortcut], mode='sum')

            return out

        # Save weights if episode = 1, else load weights
        if self.episodes == 1:

            blocks_per_group = 4
            widening_factor = 4

            inputs = Input(shape=(self.img_rows, self.img_cols, self.img_channels))

            x = Convolution2D(16, 3, 3,
                              init='he_normal', border_mode='same')(inputs)

            for i in range(0, blocks_per_group):
                nb_filters = 16 * widening_factor
                x = residual_block(x, nb_filters=nb_filters,
                                   subsample_factor=1)

            for i in range(0, blocks_per_group):
                nb_filters = 32 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                x = residual_block(x, nb_filters=nb_filters,
                                   subsample_factor=subsample_factor)

            for i in range(0, blocks_per_group):
                nb_filters = 64 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                x = residual_block(x, nb_filters=nb_filters,
                                   subsample_factor=subsample_factor)

            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=(8, 8), strides=None,
                                 border_mode='valid')(x)
            x = Flatten()(x)

            predictions = Dense(self.nb_classes, activation='softmax')(x)

            self.model = Model(input=inputs, output=predictions)

            sgd = SGD(lr=0.1, decay=5e-4, momentum=0.9, nesterov=True)

            self.model.compile(optimizer=sgd,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            # There are 96 layers, 91 is the last convolution layer
            a = []
            for layer in self.model.layers:
                # weights = layer.get_weights()
                weights = layer.get_weights()
                a.append(len(weights))

            b =[]
            for i in range(96):
                b.append(len(self.model.layers[i].get_weights()))
            return a, self.model.layers[89].get_weights()

    def step(self):
        loss = self.model.train_on_batch(self.X_train[:128], self.y_train[:128])

        return loss