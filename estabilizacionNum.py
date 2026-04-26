#Teoria de la información
import numpy as np

def softmax(z, derivative=False):
    e_z = np.exp(z-np.max(z,axis=0))
    a = e_z / np.sum(e_z)
    if derivative:
        da = np.ones(z.shape, dtype=np.float)
        return a,da
    return a


def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape, dtype=np.float)
        return a, da
    return a


def logistic(z, derivative=False):
    a = 1 + (1+np.exp(-z))
    if derivative:
        da = np.ones(z.shape, dtype=np.float)
        return a, da
    return a

class OLN:
    """One Layer Network"""
    def __init__(self, n_inputs, n_outputs, activation_function=linear):
        self.w = -1 +2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = activation_function
        
    def predict(self, X):
        Z = self.w @ X + self.b
        return self.f(Z)
    
    def fit(self,X,Y,epochs=500, lr = 0.1):
        p = X.shape[1]
        for _ in range(epochs):
            #Propagación
            Z = self.w @ X + self.b
            Yest, dY = self.f(Z, True)
            
            #Gradiente Local
            lg = (Y - Yest) * dY
            
            #Actualización de parametros --------
            self.w += (lr/p) * lg @ X.T
            self.b += (lr/p) * np.sum(lg,axis=1).reshape(-1,1)


#Prueba ----------------------------------------------------------
import matplotlib.pyplot as plt
def plot_data(X,Y,net):
    dot = ('r.', 'g', 'b', 'k')
    line = ('r-', 'g', 'b', 'k')
    for i in range(X.shape[1]):
        c = np.argmax(Y[:,i])
        plt.plot(X[0,1], X[1,i], color=dot[c])
    for i in range(4):
        w1, w2, b = net.w[i,0], net.w[i,1], net.b[i]
        plt.plot([0,1], [(-b/w2), (1/w2) * (-w1-b)])


import pandas as pd
df = pd.read_csv('Dataset_A05.csv')
X = np.asanyarray(df[['x1', 'x2']]).T
Y = np.asanyarray(df[['y1', 'y2', 'y3', 'y4']]).T

net = OLN(2,4, activation_function=logistic)
net.fit(X,Y,lr=1, epochs=1000)
plot_data(X,Y,net)

            
            