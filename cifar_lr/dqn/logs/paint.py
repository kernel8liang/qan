import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

fin = open('torchnet_test_loss.log')

ar = []
for i in fin:
    ar.append(i[:-1])

ar = [float(x) for x in ar]
##baseline VS last episode

t = range(0, len(ar))

plt.title('')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.plot(t, ar, label='qan')
plt.legend(loc='lower right')

plt.savefig('acc.pdf')


