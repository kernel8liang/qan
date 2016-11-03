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
import tensorflow as tf
from scipy import stats
from numba import cuda

class CNNEnv:
    def __init__(self):
        # Basic information on run
        self.nb_epoch = 1
        self.batch_size = 64

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

        # For generator
        self.num_examples = self.X_train.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # Total iterations we want to run
        self.tot_iter = self.X_train.shape[0] / self.batch_size

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self.batch_size = batch_size

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size

        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.X_train = self.X_train[perm]
            self.y_train = self.y_train[perm]

            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.X_train[start:end], self.y_train[start:end]

    def reset(self, episodes):
        # Get weight features before training
        self.episodes = episodes
        self.index_in_epoch = 0
        self.epochs_completed = 0

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

            # Save Weights
            self.model.save_weights('weights.h5')
        else:
            # Re-create model
            blocks_per_group = 4
            widening_factor = 4

            inputs = Input(
                shape=(self.img_rows, self.img_cols, self.img_channels))

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

            sgd = SGD(lr=0.1, nesterov=True)

            self.model.compile(optimizer=sgd,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            # Load Original Weights
            self.model.load_weights('weights.h5')

        # Get weight features before training
        self.w = self.model.layers[89].get_weights()[0]
        self.w = self.w.reshape(self.w.shape[0], self.w.shape[1],
                                self.w.shape[2] * self.w.shape[3])

        # Get gc
        def get_gc(m):
            r = m.reshape(-1)
            r_sort = sorted(r)
            n = r.size
            n1 = int(math.floor(n * 0.25))
            n2 = int(math.floor(n * 0.5))
            n3 = int(math.floor(n * 0.75))
            gc = np.zeros((12), dtype='float32')
            gc[0] = r.mean()
            gc[2] = r_sort[n1]
            gc[2] = r_sort[n2]
            gc[3] = r_sort[n3]
            gc[4] = np.std(r)
            gc[5] = stats.skew(r)
            gc[6] = stats.kurtosis(r)
            gc[7] = stats.moment(r, 1)
            gc[8] = stats.moment(r, 2)
            gc[9] = stats.moment(r, 3)
            gc[10] = stats.moment(r, 4)
            gc[11] = stats.moment(r, 5)
            return gc

        # Get hd
        def get_hd(s_param, type):
            s = s_param
            row = s.shape[0]
            col = s.shape[1]
            size = row

            if type == 1:
                size = row * col
                s = s.reshape(size, s.shape[2])

            g = np.zeros((size, 12), dtype='float32')

            for i in range(0, size):
                gc = get_gc(s[i])
                g[i] = gc

            g = np.transpose(g)

            h = np.zeros((12, 5), dtype='float32')

            for i in range(0, 12):
                hd = np.zeros((5), dtype='float32')
                hd[0] = np.mean(g[i])
                hd[1] = np.median(g[i])
                hd[2] = np.std(g[i])
                hd[3] = np.min(g[i])
                hd[4] = np.min(g[i])
                h[i] = hd
            return h

        self.res = {}

        self.res[len(self.res) + 1] = get_gc(self.w)
        self.res[len(self.res) + 1] = get_hd(self.w, 0)[:]
        self.res[len(self.res) + 1] = get_hd(np.transpose(self.w), 0)[:]
        self.res[len(self.res) + 1] = get_hd(self.w, 1)[:]

        self.state = self.res[1].reshape(12, 1)

        for i in range(2, len(self.res) + 1):
            self.state = np.concatenate((self.state, self.res[i]), axis=1)

        self.state = self.state.reshape(1, -1)

        '''
        def zca_whitening(inputs):
            # Correlation matrix
            sigma = np.dot(inputs, inputs.T) / inputs.shape[1]

            # Singular Value Decomposition
            U, S, V = np.linalg.svd(sigma)

            # Whitening constant, it prevents division by zero
            epsilon = 0.1

            # ZCA Whitening matrix
            ZCAMatrix = np.dot(
                np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))),
                U.T)

            # Data whitening
            return np.dot(ZCAMatrix, inputs)

        self.state = zca_whitening(self.state).reshape(1, -1)
        '''
        print(self.state)
        return self.state, self.tot_iter, self.nb_epoch

    def step(self, a_t, i):
        # Spike
        self.a_t = math.exp(a_t[0][0] + a_t[0][1])

        # If testing with test script use the following a_t
        # self.a_t = a_t
        self.tot_steps = i
        self.i = i  # step, starts from 1

        # Logic for looping through without limits to the call
        # This has to be improved on to ensure we don't need to manually
        # add elifs for more epoch
        if self.i <= self.tot_iter:
            self.i = self.i
        elif self.i > self.tot_iter and self.i <= (2 * self.tot_iter):
            self.i -= self.tot_iter
        elif self.i > (2 * self.tot_iter) and self.i <= (3 * self.tot_iter):
            self.i -= (2 * self.tot_iter)
        elif self.i > (3 * self.tot_iter) and self.i <= (4 * self.tot_iter):
            self.i -= (3 * self.tot_iter)
        elif self.i > (4 * self.tot_iter) and self.i <= (5 * self.tot_iter):
            self.i -= (4 * self.tot_iter)
        elif self.i > (5 * self.tot_iter) and self.i <= (6 * self.tot_iter):
            self.i -= (5 * self.tot_iter)
        elif self.i > (6 * self.tot_iter) and self.i <= (7 * self.tot_iter):
            self.i -= (6 * self.tot_iter)
        elif self.i > (7 * self.tot_iter) and self.i <= (8 * self.tot_iter):
            self.i -= (7 * self.tot_iter)
        elif self.i > (8 * self.tot_iter) and self.i <= (9 * self.tot_iter):
            self.i -= (8 * self.tot_iter)
        elif self.i > (9 * self.tot_iter) and self.i <= (10 * self.tot_iter):
            self.i -= (9 * self.tot_iter)
        elif self.i > (10 * self.tot_iter) and self.i <= (11 * self.tot_iter):
            self.i -= (10 * self.tot_iter)
        elif self.i > (11 * self.tot_iter) and self.i <= (12 * self.tot_iter):
            self.i -= (11 * self.tot_iter)
        elif self.i > (12 * self.tot_iter) and self.i <= (13 * self.tot_iter):
            self.i -= (12 * self.tot_iter)
        elif self.i > (13 * self.tot_iter) and self.i <= (14 * self.tot_iter):
            self.i -= (13 * self.tot_iter)
        elif self.i > (14 * self.tot_iter) and self.i <= (15 * self.tot_iter):
            self.i -= (14 * self.tot_iter)
        elif self.i > (15 * self.tot_iter) and self.i <= (16 * self.tot_iter):
            self.i -= (15 * self.tot_iter)
        elif self.i > (16 * self.tot_iter) and self.i <= (17 * self.tot_iter):
            self.i -= (16 * self.tot_iter)
        elif self.i > (17 * self.tot_iter) and self.i <= (18 * self.tot_iter):
            self.i -= (17 * self.tot_iter)
        elif self.i > (18 * self.tot_iter) and self.i <= (19 * self.tot_iter):
            self.i -= (18 * self.tot_iter)
        elif self.i > (19 * self.tot_iter) and self.i <= (20 * self.tot_iter):
            self.i -= (19 * self.tot_iter)
        elif self.i > (20 * self.tot_iter) and self.i <= (21 * self.tot_iter):
            self.i -= (20 * self.tot_iter)
        else:
            self.i -= (21 * self.tot_iter)

        # Batch size of 32
        # 1875 * 32 = 60000 -> # of training samples
        self.indexing = self.i - 1
        batch = self.next_batch(self.batch_size)
        self.X_batch = batch[0]
        self.y_batch = batch[1]

        '''BATCH BEGINS'''
        # Change LR
        K.set_value(self.model.optimizer.lr, self.a_t)

        # Train & evaluate one iteration
        # self.model.train_on_batch(self.X_batch, self.y_batch)
        self.train_loss = self.model.train_on_batch(self.X_batch, self.y_batch)

        self.train_err = 0
        self.test_err = 0
        if self.tot_iter == self.i:
            self.scores_test = self.model.evaluate(self.X_test, self.y_test, batch_size=32)
            self.scores_train = self.model.test_on_batch(self.X_batch, self.y_batch)
            self.train_err = (1 - self.scores_train[1])*100
            self.test_err = (1 - self.scores_test[1])*100

        # Accuracy can be obtained from self.scores_test[1] * 100
        # Loss can be obtained from self.train_loss[0]

        '''BATCH ENDS'''

        '''Weight Features Start'''
        # Get weight features on batch_end
        self.w = self.model.layers[89].get_weights()[0]
        self.w = self.w.reshape(self.w.shape[0], self.w.shape[1], self.w.shape[2] * self.w.shape[3])

        # Get gc
        def get_gc(m):
            r = m.reshape(-1)
            r_sort = sorted(r)
            n = r.size
            n1 = int(math.floor(n * 0.25))
            n2 = int(math.floor(n * 0.5))
            n3 = int(math.floor(n * 0.75))
            gc = np.zeros((12), dtype='float32')
            gc[0] = r.mean()
            gc[2] = r_sort[n1]
            gc[2] = r_sort[n2]
            gc[3] = r_sort[n3]
            gc[4] = np.std(r)
            gc[5] = stats.skew(r)
            gc[6] = stats.kurtosis(r)
            gc[7] = stats.moment(r, 1)
            gc[8] = stats.moment(r, 2)
            gc[9] = stats.moment(r, 3)
            gc[10] = stats.moment(r, 4)
            gc[11] = stats.moment(r, 5)

            return gc

        # Get hd
        def get_hd(s_param, type):
            s = s_param
            row = s.shape[0]
            col = s.shape[1]
            size = row

            if type == 1:
                size = row * col
                s = s.reshape(size, s.shape[2])

            g = np.zeros((size, 12), dtype='float32')

            for i in range(0, size):
                gc = get_gc(s[i])
                g[i] = gc

            g = np.transpose(g)

            h = np.zeros((12, 5), dtype='float32')

            for i in range(0, 12):
                hd = np.zeros((5), dtype='float32')
                hd[0] = np.mean(g[i])
                hd[1] = np.median(g[i])
                hd[2] = np.std(g[i])
                hd[3] = np.min(g[i])
                hd[4] = np.min(g[i])
                h[i] = hd
            return h

        self.res = {}

        self.res[len(self.res) + 1] = get_gc(self.w)
        self.res[len(self.res) + 1] = get_hd(self.w, 0)[:]
        self.res[len(self.res) + 1] = get_hd(np.transpose(self.w), 0)[:]
        self.res[len(self.res) + 1] = get_hd(self.w, 1)[:]

        self.state = self.res[1].reshape(12, 1)

        for i in range(2, len(self.res) + 1):
            self.state = np.concatenate((self.state, self.res[i]), axis=1)

        '''Weight Features End'''

        '''Rewards Start'''
        # BUG with accessing loss history
        self.losses.append(self.train_loss[0])

        # Reward from step 2 onwards
        if self.tot_steps > 1:
            self.batch_loss = self.losses[-1]
            self.last_loss = self.losses[-2]
            self.log_sum += math.log(self.batch_loss) - math.log(self.last_loss)
            self.reward = (-1 / (self.nb_epoch*self.tot_iter - 1)) * self.log_sum

        # Reward from step 1
        if self.tot_steps == 1:
            self.batch_loss = self.losses[-1]
            self.log_sum += math.log(self.batch_loss)
            self.reward = self.log_sum

        '''Done'''
        if self.tot_steps < (self.tot_iter * self.nb_epoch):
            self.done = False

        if self.tot_steps == (self.tot_iter * self.nb_epoch):
            self.done = True

        '''RETURN FOR DDPG'''

        self.state = self.state.reshape(1, -1)

        '''
        def zca_whitening(inputs):
            # Correlation matrix
            sigma = np.dot(inputs, inputs.T) / inputs.shape[1]

            # Singular Value Decomposition
            U, S, V = np.linalg.svd(sigma)

            # Whitening constant, it prevents division by zero
            epsilon = 0.1

            # ZCA Whitening matrix
            ZCAMatrix = np.dot(
                np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))),
                U.T)

            # Data whitening
            return np.dot(ZCAMatrix, inputs)

        self.state = zca_whitening(self.state).reshape(1, -1)
        '''

        # return s_t1, r_t, done

        return self.state, self.reward, self.done, self.test_err, self.train_err, self.train_loss[0]

