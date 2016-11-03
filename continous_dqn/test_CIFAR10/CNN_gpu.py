import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np
from numba import cuda
import math

class CNNEnv:
    def __init__(self):

        # The data, shuffled and split between train and test sets
        self.x_train, self.y_train, self.x_test, self.y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

        # Reorder dimensions for tensorflow
        self.mean = np.mean(self.x_train, axis=0, keepdims=True)
        self.std = np.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        # For generator
        self.num_examples = self.x_train.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # For wide resnets
        self.blocks_per_group = 4
        self.widening_factor = 4

        # Basic info
        self.batch_num = 64
        self.img_row = 32
        self.img_col = 32
        self.img_channels = 3
        self.nb_classes = 10

        # Basic information on run
        self.nb_epoch = 50
        self.batch_size = 64

        # Total iterations we want to run
        self.tot_iter = self.x_train.shape[0] / self.batch_size

        # Other information for step function
        self.losses = 0

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

    def reset(self, first):
        self.first = first
        self.tot_steps = 0
        if self.first is False:
            self.sess.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)

        def zero_pad_channels(x, pad=0):
            """
            Function for Lambda layer
            """
            pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
            return tf.pad(x, pattern)

        def residual_block(x, count, nb_filters=16, subsample_factor=1):
            prev_nb_channels = x.outputs.get_shape().as_list()[3]

            if subsample_factor > 1:
                subsample = [1, subsample_factor, subsample_factor, 1]
                # shortcut: subsample + zero-pad channel dim
                name_pool = 'pool_layer' + str(count)
                shortcut = tl.layers.PoolLayer(x,
                                               ksize=subsample,
                                               strides=subsample,
                                               padding='VALID',
                                               pool=tf.nn.avg_pool,
                                               name=name_pool)

            else:
                subsample = [1, 1, 1, 1]
                # shortcut: identity
                shortcut = x

            if nb_filters > prev_nb_channels:
                name_lambda = 'lambda_layer' + str(count)
                shortcut = tl.layers.LambdaLayer(
                    shortcut,
                    zero_pad_channels,
                    fn_args={'pad': nb_filters - prev_nb_channels},
                    name=name_lambda)

            name_norm = 'norm' + str(count)
            y = tl.layers.BatchNormLayer(x,
                                         decay=0.999,
                                         epsilon=1e-05,
                                         is_train=True,
                                         name=name_norm)

            name_conv = 'conv_layer' + str(count)
            y = tl.layers.Conv2dLayer(y,
                                      act=tf.nn.relu,
                                      shape=[3, 3, prev_nb_channels, nb_filters],
                                      strides=subsample,
                                      padding='SAME',
                                      name=name_conv)

            name_norm_2 = 'norm_second' + str(count)
            y = tl.layers.BatchNormLayer(y,
                                         decay=0.999,
                                         epsilon=1e-05,
                                         is_train=True,
                                         name=name_norm_2)

            prev_input_channels = y.outputs.get_shape().as_list()[3]
            name_conv_2 = 'conv_layer_second' + str(count)
            y = tl.layers.Conv2dLayer(y,
                                      act=tf.nn.relu,
                                      shape=[3, 3, prev_input_channels, nb_filters],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      name=name_conv_2)

            name_merge = 'merge' + str(count)
            out = tl.layers.ElementwiseLayer([y, shortcut],
                                             combine_fn=tf.add,
                                             name=name_merge)


            return out

        # Placeholders
        self.learning_rate = tf.placeholder(tf.float32)
        self.spikes = tf.placeholder(tf.float32)
        lr = tf.add(tf.exp(self.learning_rate), tf.exp(self.spikes))

        self.img = tf.placeholder(tf.float32, shape=[self.batch_num, 32, 32, 3])
        self.labels = tf.placeholder(tf.int32, shape=[self.batch_num, ])

        x = tl.layers.InputLayer(self.img, name='input_layer')
        x = tl.layers.Conv2dLayer(x,
                                  act=tf.nn.relu,
                                  shape=[3, 3, 3, 16],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='cnn_layer_first')

        for i in range(0, self.blocks_per_group):
            nb_filters = 16 * self.widening_factor
            count = i
            x = residual_block(x, count, nb_filters=nb_filters, subsample_factor=1)

        for i in range(0, self.blocks_per_group):
            nb_filters = 32 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            count = i + self.blocks_per_group
            x = residual_block(x, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

        for i in range(0, self.blocks_per_group):
            nb_filters = 64 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            count = i + 2*self.blocks_per_group
            x = residual_block(x, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

        x = tl.layers.BatchNormLayer(x,
                                     decay=0.999,
                                     epsilon=1e-05,
                                     is_train=True,
                                     name='norm_last')

        x = tl.layers.PoolLayer(x,
                                ksize=[1, 8, 8, 1],
                                strides=[1, 8, 8, 1],
                                padding='VALID',
                                pool=tf.nn.avg_pool,
                                name='pool_last')

        x = tl.layers.FlattenLayer(x, name='flatten')

        self.x = tl.layers.DenseLayer(x,
                                 n_units=self.nb_classes,
                                 act=tf.identity,
                                 name='fc')

        output = self.x.outputs

        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, self.labels))
        self.cost = ce

        correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), self.labels)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_params = x.all_params

        self.train_op = tf.train.GradientDescentOptimizer(
            lr, use_locking=False).minimize(self.cost, var_list=train_params)

        '''WEIGHT'''
        with tf.variable_scope("", reuse=True):
            w = tf.get_variable("conv_layer_second11/W_conv2d")

        # tl.layers.print_all_variables()
        # print(w.get_shape()) gives (3, 3, 256, 256)
        shape = w.get_shape().as_list()
        w = tf.reshape(w, (shape[0], shape[1], shape[2] * shape[3]))

        def get_gc(m):
            r = tf.reshape(m, [-1])
            size = r.get_shape().as_list()[0]
            # First moment: mean
            m1 = tf.reduce_mean(r)
            # Second Moment: standard deviation
            m2 = tf.div(tf.reduce_mean(tf.square(tf.sub(r, m1))), size)
            # Third moment
            m3 = tf.div(tf.reduce_mean(tf.pow((tf.sub(r, m1)), 3)), size)
            # Fourth Moment
            m4 = tf.div(tf.reduce_mean(tf.pow((tf.sub(r, m1)), 4)), size)
            # Skewness
            m5 = tf.pow(tf.div(m3, m2), 2)
            # Kurtosis
            m6 = tf.div(m4, tf.square(m2))
            gc = tf.pack([m1, m2, m3, m4, m5, m6])
            return gc

        def get_hd(s_param, type):
            s = s_param
            row = s.get_shape().as_list()[0]
            col = s.get_shape().as_list()[1]
            dim1 = s.get_shape().as_list()[2]
            size = row

            if type == 1:
                size = row * col
                s = tf.reshape(s, [size, dim1])

            lst = []

            for i in range(0, size):
                gc = get_gc(s[i])
                lst.append(gc)

            g = tf.pack(lst)

            lst_2 = []
            for i in range(0, 6):
                size = g[i].get_shape().as_list()[0]
                # First moment: mean
                hd1 = tf.reduce_mean(g[i])
                # Second Moment: standard deviation
                hd2 = tf.div(tf.reduce_mean(tf.square(tf.sub(g[i], hd1))), size)
                # Third moment
                hd3 = tf.div(tf.reduce_mean(tf.pow((tf.sub(g[i], hd1)), 3)), size)
                # Fourth Moment
                hd4 = tf.div(tf.reduce_mean(tf.pow((tf.sub(g[i], hd1)), 4)), size)
                # Skewness
                hd5 = tf.pow(tf.div(hd3, hd2), 1.5)
                # Kurtosis
                hd6 = tf.div(hd4, tf.square(hd2))
                # Pack
                hd = tf.pack([hd1, hd2, hd3, hd4, hd5, hd6])
                lst_2.append(hd)

            h = tf.pack(lst_2)
            return h

        res = {}
        res[len(res) + 1] = get_gc(w)
        # res[len(res) + 1] = get_hd(w, 0)[:]
        # res[len(res) + 1] = get_hd(tf.transpose(w), 0)[:]
        res[len(res) + 1] = get_hd(w, 1)[:]

        state = tf.reshape(res[1], [6, 1])

        for i in range(2, len(res) + 1):
            state = tf.concat(1, [state, res[i]])

        state_vec = tf.reshape(state, [-1])
        state_size = state_vec.get_shape().as_list()[0]
        mean_state = tf.reduce_mean(state_vec)
        var_state = tf.div(tf.reduce_mean(tf.square(tf.sub(state_vec, mean_state))), state_size)
        sd_state = tf.sqrt(var_state)
        state_norm = tf.div(tf.sub(state_vec, mean_state), sd_state)
        self.st = tf.reshape(state_norm, [1, -1])

        '''
        self.st = tf.reshape(state, [1, -1])
        '''

        self.nb_epochs = tf.placeholder(tf.float32)
        self.tot_iters = tf.placeholder(tf.float32)
        self.loss_prev = tf.placeholder(tf.float32)
        self.log_sum = tf.Variable([0.00], tf.float32)
        self.reward = tf.Variable([0.00], tf.float32)

        # Getting loss
        batch_loss = self.cost
        last_loss = self.loss_prev
        if self.tot_steps > 1:
            self.log_sum = tf.add(self.log_sum, tf.sub(tf.log(batch_loss), tf.log(last_loss)))
            self.reward = tf.mul((-1 / (tf.mul(self.nb_epochs, self.tot_iters) - 1)), self.log_sum)

        if self.tot_steps <= 1:
            self.log_sum = tf.add(self.log_sum, tf.log(batch_loss))
            self.reward = tf.mul((-1 / (tf.mul(self.nb_epochs, self.tot_iters) - 1)), self.log_sum)

        self.sess.run(tf.initialize_all_variables())

        wt = self.sess.run(self.st)
        # print(wt)
        return wt, self.tot_iter, self.nb_epoch

    def step(self, lrate, spike, i):
        self.lrate = lrate
        self.spike = spike
        self.tot_steps = i
        self.i = i

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

        batch = self.next_batch(self.batch_num)

        feed_dict = {self.img: batch[0],
                     self.labels: batch[1],
                     self.learning_rate: self.lrate,
                     self.spikes: self.spike,
                     self.nb_epochs: self.nb_epoch,
                     self.tot_iters: self.tot_iter,
                     self.loss_prev: self.losses}

        feed_dict.update(self.x.all_drop)
        _, l, ac, wt, rt = self.sess.run(
            [self.train_op, self.cost, self.acc, self.st, self.log_sum], feed_dict=feed_dict)

        self.losses = l  # For assigning in the next round

        if self.tot_steps < (self.tot_iter * self.nb_epoch):
            done = False
        if self.tot_steps == (self.tot_iter * self.nb_epoch):
            done = True

        test_loss, test_acc, n_batch = 0, 0, 0
        tot_test_loss, tot_test_acc = 0, 0

        if self.tot_iter == self.i:
            for X_test_a, y_test_a in tl.iterate.minibatches(self.x_test, self.y_test, self.batch_num, shuffle=True):
                # Disable all dropout/dropconnect/denoising layers
                dp_dict = tl.utils.dict_to_one(self.x.all_drop)
                feed_dict = {self.img: X_test_a, self.labels: y_test_a}
                feed_dict.update(dp_dict)
                err, ac = self.sess.run([self.cost, self.acc], feed_dict=feed_dict)
                test_loss += err
                test_acc += ac
                n_batch += 1

            tot_test_loss = test_loss / n_batch
            tot_test_acc = test_acc / n_batch
        return wt, rt, done, tot_test_loss, tot_test_acc, l, ac

'''
# Testing with 3 steps
env = CNNEnv()
env.reset(True)
env.step(0.001, 0.001, 1)
env.step(0.001, 0.001, 2)
env.step(0.001, 0.001, 3)
'''
