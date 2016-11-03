import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.datasets import cifar10
from keras.utils import np_utils
import numpy as np


class CNNEnv:
    def __init__(self):
        # the data, shuffled and split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # reorder dimensions for tensorflow
        self.x_train = np.transpose(self.x_train.astype('float32'), (0, 1, 2, 3))
        self.mean = np.mean(self.x_train, axis=0, keepdims=True)
        self.std = np.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = np.transpose(self.x_test.astype('float32'), (0, 1, 2, 3))
        self.x_test = (self.x_test - self.mean) / self.std

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1]*self.x_train.shape[2]*self.x_train.shape[3])
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1]*self.x_test.shape[2]*self.x_test.shape[3])
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)


        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)

        self.num_examples = self.x_train.shape[0]

        self.index_in_epoch = 0
        self.epochs_completed = 0

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
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]

            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]


    def step(self):
        sess = tf.Session()

        K.set_session(sess)

        # this placeholder will contain our input digits, as flat vectors
        img = tf.placeholder(tf.float32, shape=(None, 3072))

        # Keras layers can be called on TensorFlow tensors:
        x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
        x = Dense(128, activation='relu')(x)
        preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

        labels = tf.placeholder(tf.float32, shape=(None, 10))

        loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        with sess.as_default():
            for i in range(3000):
                batch = self.next_batch(128)
                _, l = sess.run([optimizer, loss],
                                feed_dict={img: batch[0], labels: batch[1]})
                print(l)

        acc_value = accuracy(labels, preds)

        with sess.as_default():
            print acc_value.eval(feed_dict={img: self.x_test, labels: self.y_test})

a = CNNEnv()
a.step()