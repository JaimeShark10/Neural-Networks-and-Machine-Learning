import numpy as np
import matplotlib.pyplot as plt

def linear(z, derivate =False):
    a = z
    #Calcular la derivada
    if derivate:
        da = np.ones(z.shape) #Funcion en caso de un vector o matriz
        return a, da
    return a

def logistic(z, derivate=False):
    a = 1/(1+np.exp(-z))
    if derivate:
        da = a * (1-a)
        return a, da
    return a

def tanh(z, derivate=False):
    a = np.tanh(z)
    if derivate:
        da =(1 + a) * (1 - a)
        return a, da
    return a

def relu(z, derivate=False):
    a = z * (z>=0)
    if derivate:
        da = 1.0*(z>=0)
        return a, da
    return a

#Si queremos hacer una neurona, tengamos la funcion para ir hacia adelante y atras

class neuron:
    def __init__(self,n_inputs, actFunction=linear):
        self.w = -1+2 * np.random.rand(n_inputs)
        self.b = -1+2 * np.random.rand()
        self.f = actFunction
    #def predict(self, x,):