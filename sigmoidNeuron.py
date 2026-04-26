# x = [x(1), x(2)....x(n)...x(p)]
# y = [y(1), y(2)....y(n)...y(p)]
import numpy as np
import matplotlib.pyplot as plt

#Una funcion esta vectorizada si f: R -> R
# f(x1,x2...xn) = [f(x1,x2...nx)] En vectorial 


class LogisticNeuron:
    def __init__(self, n_inputs):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()

    def predict(self, x):                             #Categorias
        z = np.dot(self.w, x) + self.b
        Yest = 1/(np.exp(-z))
        return Yest
    
    def predictClass(self, x, umbral, treshold = 0.5): #Probability
        return 1.0*(self.predict(x) > treshold) 
    
    def fit(self, x, y, lr=0.1, epochs = 100):
        p = x.shape[1]
        for _ in range(epochs):
            Yest = self.predict(x)
            self.w += (lr/p) * ((y-Yest) @x.T).ravel()
            self.b += (lr/p) * np.sum(y-Yest)
    
#Ejemplo
x = np.array([[0,1,0,1],
              [0,0,1,1]])
y = np.array([[0,0,0,1]])

neuron = LogisticNeuron(2)
neuron.fit(x, y, epochs = 500, lr = 1)
print(neuron.predict(x))
print(neuron.predictClass(x))


#Tarea - Clasificacion de Diabetes
#Dataset Se recomienda leer con Pandas
#Pregnant, Blood Pressure, Glucosa, BMT, Pedigree, Age, [Diagnostic (0 v 1)]



    