import hickle as hkl

train_acc = hkl.load('train_acc.hkl')
print(train_acc.shape)
print(train_acc)

test_acc = hkl.load('test_acc.hkl')
print(test_acc.shape)
print(test_acc)

train_loss = hkl.load('train_loss.hkl')
print(train_loss.shape)
print(train_loss)

test_loss = hkl.load('test_loss.hkl')
print(test_loss.shape)
print(test_loss)

lrate = hkl.load('lrate.hkl')
print(lrate.shape)
print(lrate)

spikes = hkl.load('spikes.hkl')
print(spikes.shape)
print(spikes)