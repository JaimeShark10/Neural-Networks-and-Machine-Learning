import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Creamos ejemplo
p = 50
x = np.linspace(-5,5,p).reshape(1,-1)
y = 2 * np.cos(x) + np.sin(3*x) + 5

#Grafica
plt.plot(x,y, 'or')

k = 3

model = KMeans(n_clusters=k)
model.fit(x.T)

c = model.cluster_centers_
