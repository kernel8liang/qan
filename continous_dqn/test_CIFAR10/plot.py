import hickle as hkl

test_err = hkl.load('test_err.hkl')
print(test_err.shape)
print(test_err)

train_err = hkl.load('train_err.hkl')
print(train_err.shape)
print(train_err)

train_loss = hkl.load('train_loss.hkl')
print(train_loss.shape)
print(train_loss)

lrate = hkl.load('lrate.hkl')
print(lrate.shape)
print(lrate)
