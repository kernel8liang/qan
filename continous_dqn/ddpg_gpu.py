# from CNN import CNNEnv
import numpy as np
import hickle as hkl
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json
import math

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from CNN_gpu import CNNEnv
from OU import OU
import timeit

# Instantiate Ornstein-Uhlenbeck Process and assign to object
OU = OU()

# 1 to train and 0 to run
def runCNN(train_indicator=1):
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001  # Target Network HyperParameters (TAU)
    LRA = 0.0001  # Learning rate for Actor (LRA)
    LRC = 0.001  # Learning rate for Critic (LRC)

    action_dim = 2  # Learning rate from CNN that is running on MNIST/CIFAR
    state_dim = 42  # Number of weight features

    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 10
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    # Variables for graphing
    lrate_graph = []
    spike_graph = []
    training_loss_graph = []
    training_acc_graph = []
    testing_loss_graph = []
    testing_acc_graph = []

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)

    # Create replay buffer
    buff = ReplayBuffer(BUFFER_SIZE)

    # Now load the weight if we are not training
    '''
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")
    '''

    print("Start RL-LR experiment.")
    episodes = 0
    # Instantiate CNN running on CIFAR Environment only once
    env = CNNEnv()

    for i in range(episode_count):
        episodes += 1
        print(
        "Episode : " + str(episodes) + " Replay Buffer " + str(buff.count()))

        if episodes == 1:
            first = True
        else:
            first = False

        s_t, tot_iter, nb_epoch = env.reset(first)

        max_steps = tot_iter * nb_epoch
        total_reward = 0.

        # Reset step
        num_episode = 0
        for j in range(max_steps):
            loss = 0
            num_episode += 1  # This would start from 1
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t)
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(
                a_t_original[0][0], -4.61, 0.40, 0.30)

            # Stochastic spikes
            if random.random() <= 0.1:
                noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(
                    a_t_original[0][1], -3.00, 1.00, 0.5)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]

            # Given Action at step t, get at step t:
            # (1) Weight Features: s_t1
            # (2) Reward: r_t
            # (3) Done - True or False, if True, terminates
            s_t1, r_t, done, test_loss, test_acc, train_loss, train_acc = env.step(a_t[0][0], a_t[0][1], num_episode)

            # Add replay buffer
            buff.add(s_t, a_t[0], r_t[0], s_t1, done)

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            states = states.reshape(states.shape[0], states.shape[2])
            new_states = new_states.reshape(new_states.shape[0], new_states.shape[2])

            target_q_values = critic.target_model.predict(
                [new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            step += 1

            '''
            print("Episode", i, "Step", step, "Action LR", a_t[0][0],
                  "Action Spike", a_t[0][1],
                  "Reward", r_t, "Loss", loss, "State", s_t1)
            '''

            # Better monitor
            if (step % 500) == 0:
                print("Episode", i, "Step", step, "Action LR", a_t[0][0],
                      "Action Spike", a_t[0][1],
                      "Reward", r_t, "Loss", loss)

            '''Quick fix to appending epochs'''

            iter_cnn = num_episode

            if iter_cnn <= tot_iter:
                iter_cnn = iter_cnn
            elif iter_cnn > tot_iter and iter_cnn <= (2 * tot_iter):
                iter_cnn -= tot_iter
            elif iter_cnn > (2 * tot_iter) and iter_cnn <= (3 * tot_iter):
                iter_cnn -= (2 * tot_iter)
            elif iter_cnn > (3 * tot_iter) and iter_cnn <= (4 * tot_iter):
                iter_cnn -= (3 * tot_iter)
            elif iter_cnn > (4 * tot_iter) and iter_cnn <= (5 * tot_iter):
                iter_cnn -= (4 * tot_iter)
            elif iter_cnn > (5 * tot_iter) and iter_cnn <= (6 * tot_iter):
                iter_cnn -= (5 * tot_iter)
            elif iter_cnn > (6 * tot_iter) and iter_cnn <= (7 * tot_iter):
                iter_cnn -= (6 * tot_iter)
            elif iter_cnn > (7 * tot_iter) and iter_cnn <= (8 * tot_iter):
                iter_cnn -= (7 * tot_iter)
            elif iter_cnn > (8 * tot_iter) and iter_cnn <= (9 * tot_iter):
                iter_cnn -= (8 * tot_iter)
            elif iter_cnn > (9 * tot_iter) and iter_cnn <= (10 * tot_iter):
                iter_cnn -= (9 * tot_iter)
            elif iter_cnn > (10 * tot_iter) and iter_cnn <= (
                11 * tot_iter):
                iter_cnn -= (10 * tot_iter)
            elif iter_cnn > (11 * tot_iter) and iter_cnn <= (
                12 * tot_iter):
                iter_cnn -= (11 * tot_iter)
            elif iter_cnn > (12 * tot_iter) and iter_cnn <= (
                13 * tot_iter):
                iter_cnn -= (12 * tot_iter)
            elif iter_cnn > (13 * tot_iter) and iter_cnn <= (
                14 * tot_iter):
                iter_cnn -= (13 * tot_iter)
            elif iter_cnn > (14 * tot_iter) and iter_cnn <= (
                15 * tot_iter):
                iter_cnn -= (14 * tot_iter)
            elif iter_cnn > (15 * tot_iter) and iter_cnn <= (
                16 * tot_iter):
                iter_cnn -= (15 * tot_iter)
            elif iter_cnn > (16 * tot_iter) and iter_cnn <= (
                17 * tot_iter):
                iter_cnn -= (16 * tot_iter)
            elif iter_cnn > (17 * tot_iter) and iter_cnn <= (
                18 * tot_iter):
                iter_cnn -= (17 * tot_iter)
            elif iter_cnn > (18 * tot_iter) and iter_cnn <= (
                19 * tot_iter):
                iter_cnn -= (18 * tot_iter)
            elif iter_cnn > (19 * tot_iter) and iter_cnn <= (
                20 * tot_iter):
                iter_cnn -= (19 * tot_iter)
            elif iter_cnn > (20 * tot_iter) and iter_cnn <= (
                21 * tot_iter):
                iter_cnn -= (20 * tot_iter)
            else:
                iter_cnn -= (21 * tot_iter)

            # Append training error and testing error at every epoch
            if iter_cnn == tot_iter:
                training_acc_graph.append(train_acc)
                training_loss_graph.append(train_loss)
                testing_acc_graph.append(test_loss)
                testing_loss_graph.append(test_acc)

            # Append lr and spikes at every iteration
            lrate_graph.append(a_t[0][0])
            spike_graph.append(a_t[0][1])

            # Break if done
            if done:
                break

        '''
        if train_indicator:
            print("Now we save model")
            actor.model.save_weights("actormodel.h5", overwrite=True)
            with open("actormodel.json", "w") as outfile:
                json.dump(actor.model.to_json(), outfile)

            critic.model.save_weights("criticmodel.h5", overwrite=True)
            with open("criticmodel.json", "w") as outfile:
                json.dump(critic.model.to_json(), outfile)


        print("TOTAL REWARD @ " + str(i) + "-th Episode : Reward " + str(
            total_reward))
        print("Total Step: " + str(step))
        print("")
        '''

    print('Saving arrays for graphing')

    # Save arrays for graphing
    training_acc_graph = np.asarray(training_acc_graph)
    hkl.dump(training_acc_graph, 'train_acc.hkl', mode='w')

    testing_acc_graph = np.asarray(testing_acc_graph)
    hkl.dump(testing_acc_graph, 'test_acc.hkl', mode='w')

    training_loss_graph = np.asarray(training_loss_graph)
    hkl.dump(training_loss_graph, 'train_loss.hkl', mode='w')

    testing_loss_graph = np.asarray(testing_loss_graph)
    hkl.dump(testing_loss_graph, 'test_loss.hkl', mode='w')

    lrate_graph = np.asarray(lrate_graph)
    hkl.dump(lrate_graph, 'lrate.hkl', mode='w')

    spike_graph = np.asarray(spike_graph)
    hkl.dump(spike_graph, 'spikes.hkl', mode='w')

    print("Now we save model")
    actor.model.save_weights("actormodel.h5", overwrite=True)
    with open("actormodel.json", "w") as outfile:
        json.dump(actor.model.to_json(), outfile)

    critic.model.save_weights("criticmodel.h5", overwrite=True)
    with open("criticmodel.json", "w") as outfile:
        json.dump(critic.model.to_json(), outfile)

    print("Finished saving and training.")


if __name__ == "__main__":
    runCNN()