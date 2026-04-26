from testH import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

numbers = pd.read_csv('numbers.csv')
names = numbers['class'].values
numbers = numbers.values[:,:-1].T
numbers = (numbers > 150) * 1 + (numbers <= 150) * -1

net = Hopfield(784, mode='pinv')
net.fit(numbers)

digit= numbers[:,0].reshape((28,28)) 

plt.matshow(digit, cmap=plt.cm.gray)
plt.axis('equal')

noise = 0.5
A = (np.random.rand(28,28) < noise)
A = (A*-1) + (1*A==False)
d = digit * A
plt.matshow(d, cmap=plt.cm.gray)

new = net.prediction(d.reshape(-1,1), max_iter=10)
plt.matshow(new.reshape(28,28), cmap=plt.cm.gray)
plt.show()