import numpy as np
import matplotlib.pyplot as plt

def linear (z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    a = 1/(1+np.exp(-z))
    if derivative:
        da = np.ones(z.shape)
        return a,da
    return a

def softmax(z, derivative= False):
    e = np.exp(z-np.max(z, axis = 0))
    a = e/np.sum(e, axis=0)
    if derivative:
        da = np.ones(z.shape)
        return a,da
    return a


def tanh(z, derivative = False):
    a = np.tanh(z)
    if derivative:
        da = (1-a) * (1+a)
        return a,da
    return a

def relu(z, deribative=False):
    a = z * (z >= 0)
    if deribative:
        da = np.array(z>=0, dtype=float)
        return a, da
    return a

def logistic_hidden(z, derivative=False):
    a = 1/(1+np.exp(-z))
    if derivative:
        da = a *(1-a)
        return a,da
    return a


def MLP_binary_classification(X,Y,net):
    plt.figure()
    for i in range(X.shape[1]):
       
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], 'ro', markersize=9)
        else:
            plt.plot(X[0,i], X[1,i], 'bo', markersize=9)
            
    xmin, ymin = np.min(X[0, :])-0.5, np.min(X[1,:])-0.5
    xmax, ymax = np.max(X[0, :])+0.5, np.max(X[1,:])+0.5
    xx,yy = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax, 100))

    data = np.array([xx.ravel(), yy.ravel()])
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    
    plt.contourf(xx,yy,zz, alpha = 0.8, cmap = plt.cm.RdBu) 
    plt.contour(xx,yy,zz,[0.5], colors='k', linestyles='--', linewidths=2)
    
    plt.title('Clasificación de red neuronal (MLP)')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.grid()
    plt.show() # Se mantiene plt.show() para ejecución local