# x = [x(1), x(2)....x(n)...x(p)]
# y = [y(1), y(2)....y(n)...y(p)]
import numpy as np
import matplotlib.pyplot as plt

#Funcion para tener el numero de patrón
class linearNeuron:
    def __init__(self, n_inputs):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()

    def predict(self, x):
        Yest = np.dot(self.w, x) + self.b
        return Yest
    
    def batcher(self, x, y, batch_size):
        p = x.shape[1]
        li, ui = 0, batch_size
        while True:
            if li < p:
                yield x[:, li:ui], y[:, li:ui]
                li, ui = li + batch_size, ui + batch_size
            else:
                return None
    def MSE(self, x, y):
        p = x.shape[1]
        Yest = self.predict(x)
        return (1/(2*p))*np.sum((y-Yest)**2)
    
    def fit(self, x, y, lr=0.1, epochs = 200, batch_size = 32):
        errorHistory = []
        for _ in range(epochs):
            minibatch = self.batcher(x, y, batch_size= batch_size)
            for mX, mY in minibatch:
                p = mX.shape[1]
                Yest = self.predict(mX)
                self.w += (lr/p) * ((mY-Yest) @ mX.T).ravel()
                self.b += (lr/p) * np.sum(mY-Yest)
            errorHistory.append(self.MSE(x,y))
        return errorHistory
    
#Ejemplo
p = 100
x =-1 + 2*np.random.rand(p).reshape(1,-1)
y = -18 * x + 6+ np.random.randn(p)
neuron = linearNeuron(1)
error = neuron.fit(x,y, batch_size=10)

plt.plot(x,y, '.b')
xn = np.array([[-1,1]])
plt.plot(xn.ravel(), neuron.predict(xn), '--r')
plt.figure()
plt.plot(error)
plt.show()
