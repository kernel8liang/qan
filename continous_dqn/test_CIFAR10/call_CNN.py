from CNN_with_class import CNNEnv

# Instantiate class and assign to object env
env = CNNEnv()


# Test reset
a, b = env.reset(1)

loss = env.step()
print(loss)


'''
s_t, max_epsiode, nb_epochs = env.reset(1)
print('State\'s shape: {}'.format(s_t.shape))
print('Max episode or total iterations: {}'.format(max_epsiode))
print('Number of epochs: {}'.format(nb_epochs))
'''

# Test ddpg
'''
i = 0
for j in range(2):
    i += 1
    s_t1, r_t, done = env.step(0.001, i)
    print('Iteration {}'.format(i))
    print('State\'s shape: {}'.format(s_t1))
    print('Rewards: {}'.format(r_t))
    print('Done: {}'.format(done))
'''