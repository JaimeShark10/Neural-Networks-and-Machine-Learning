import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
import os

def linear(z, derivative=False):
    """Función de activación lineal (Identidad)."""
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    """
    Función de activación logística (Sigmoide) - Numéricamente estable.
   """
    a = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
    
    if derivative:
        da = a * (1 - a)
        return a, da
    return a

def softmax(z, derivative=False):
    """Función de activación Softmax."""
    
    e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    a = e_z / np.sum(e_z, axis=0, keepdims=True)
    if derivative:
        da = np.ones(z.shape) 
        return a, da
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



# Multilayer Perceptron or dense network
class denseNetwork:

    def __init__(self, layer_dims, hidden_activation=tanh, 
                 output_activation = logistic):
        
        # Atributtes
        self.L = len(layer_dims) - 1
        self.w = [None] * (self.L + 1) 
        self.b = [None] * (self.L + 1)
        self.f = [None] * (self.L + 1)
        self.output_activation = output_activation 

        # Inicialize
        for l in range(1,self.L + 1):
            self.w[l] = -1+2 * np.random.rand(layer_dims[l], layer_dims[l-1])
            self.b[l] = -1+2 * np.random.rand(layer_dims[l], 1)
            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation
            
    def predict(self, X):
        A = X.copy()
        for l in range(1, self.L + 1):
            Z = self.w[l] @ A + self.b[l]
            A = self.f[l](Z)
        return A

    def fit(self, X, Y, epochs = 10000, lr=0.01): # Aumento epochs y bajo lr para estabilidad
        p = X.shape[1]
        for _  in range(epochs):
            # Inicialize containers
            A = [None] * (self.L + 1)
            dA = [None] * (self.L + 1)
            lg = [None] * (self.L + 1)
            dw = [None] * (self.L + 1)
            db = [None] * (self.L + 1)

            # Propagation
            A[0] = X.copy()
            for l in range(1, self.L + 1):
                Z = self.w[l] @ A[l-1] + self.b[l]
                A[l], dA[l] = self.f[l](Z, derivative=True)

            # Backpropagation
            for l in range(self.L, 0, -1):
                if l == self.L:
                    # FIX: Manejo correcto del error para Softmax (Cross-Entropy)
                    if self.output_activation == softmax:
                        lg[l] = A[l] - Y
                    else:
                        # Original (Y-A) * dA para otras pérdidas/activaciones
                        lg[l] = (Y - A[l]) * dA[l]
                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * dA[l]
                
                # Cálculo de gradientes
                dw[l] = (1/p) * (lg[l] @ A[l-1].T)
                db[l] = (1/p) * np.sum(lg[l], axis=1, keepdims=True)

                self.w[l] = self.w[l] - lr * dw[l]
                self.b[l] = self.b[l] - lr * db[l]

if __name__ == "__main__":
    
    # 2. Extrae de los datos la matriz X y Y
    file_name = "Dataset_A05.csv"
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: Asegúrate de que el archivo de datos '{file_name}' esté en el mismo directorio.")
        exit()

    X = df[['x1', 'x2']].values.T 
    Y = df[['y1', 'y2', 'y3', 'y4']].values.T 
    n_features, n_samples = X.shape
    n_outputs = Y.shape[0]
    
    # El ejercicio pide "Red Neuronal de una capa", así que layer_dims es [2, 4]
    layer_dims = [n_features, n_outputs] 
    
    print("--- INICIO DE ACTIVIDAD ---")
    
    print("\n[PARTE 1] Entrenando red Multi-Etiqueta (Logistic)...")
    # 5. Entrena la red con logistic (sigmoide)
    net_multi_label = denseNetwork(
        layer_dims = layer_dims, 
        output_activation = logistic
    )
    net_multi_label.fit(X, Y, epochs=5000, lr=0.1)

    # 6, 7, 8. Ploteo de datos y fronteras lineales
    def plot_multi_label_lines(X, Y, net, filename='multi_label_lines_final.png'):
        plt.figure(figsize=(10, 8))
        colors = ['r', 'g', 'b', 'm']
        y_class = np.argmax(Y, axis=0) # Clase "ganadora" original para colorear

        # 6. Dibuja los datos
        for i in range(Y.shape[0]):
            idx = np.where(y_class == i)[0]
            plt.scatter(X[0, idx], X[1, idx], color=colors[i], marker='o', s=50, alpha=0.6, label=f'Clase {i+1}')

        x_min, y_min = np.min(X[0, :]), np.min(X[1,:])
        x_max, y_max = np.max(X[0, :]), np.max(X[1,:])
        margin = 50000
        x_range = np.linspace(x_min - margin, x_max + margin, 100)

        W1 = net.w[1] 
        B1 = net.b[1] 

        for i in range(W1.shape[0]): 
            w1_i, w2_i = W1[i, 0], W1[i, 1]
            b_i = B1[i, 0]

            if np.abs(w2_i) > 1e-6:
                y_line = (-b_i - w1_i * x_range) / w2_i
                plt.plot(x_range, y_line, color=colors[i], linestyle='-', linewidth=2, label=f'Límite Neurona {i+1}')
            else:
                x_val = -b_i / w1_i
                plt.axvline(x=x_val, color=colors[i], linestyle='-', linewidth=2, label=f'Límite Neurona {i+1}')

        plt.title('Parte 1: Clasificación Multi-Etiqueta (Fronteras Lineales)')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(filename)
        plt.close()
        return filename

    plot_1_filename = plot_multi_label_lines(X, Y, net_multi_label)
    print(f"  -> Gráfico de Parte 1 guardado como: {plot_1_filename}")

    
    print("\n[PARTE 2] Entrenando red Single-Winner (Softmax)...")
    # 9. Entrena la red con softmax
    net_single_winner = denseNetwork(
        layer_dims = layer_dims, 
        output_activation = softmax
    )
    net_single_winner.fit(X, Y, epochs=10000, lr=0.01)

    # 10, 11, 12. Ploteo de datos y regiones de decisión
    def plot_decision_areas(X, Y, net, filename='single_winner_areas_final.png'):
        plt.figure(figsize=(10, 8))

        # Colormaps para fondo y puntos
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFAAFF']) 
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FF00FF']) 

        x_min, x_max = X[0, :].min() - 500, X[0, :].max() + 500
        y_min, y_max = X[1, :].min() - 500, X[1, :].max() + 500
        
        # Malla para predicción
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))

        X_mesh = np.c_[xx.ravel(), yy.ravel()].T
        Z_pred = net.predict(X_mesh) 
        Z_class = np.argmax(Z_pred, axis=0) 
        Z_class = Z_class.reshape(xx.shape)

        plt.pcolormesh(xx, yy, Z_class, cmap=cmap_light, shading='auto', alpha=0.8)

        # 10. Dibuja los datos
        y_class = np.argmax(Y, axis=0)
        for i in range(Y.shape[0]):
            idx = np.where(y_class == i)[0]
            plt.scatter(X[0, idx], X[1, idx], color=cmap_bold(i), marker='o', s=50, alpha=1.0, edgecolor='k', label=f'Clase {i+1}')

        plt.title('Parte 2: Clasificación de un Solo Ganador (Regiones de Decisión Softmax)')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(filename)
        plt.close()
        return filename

    plot_2_filename = plot_decision_areas(X, Y, net_single_winner)
    print(f"  -> Gráfico de Parte 2 guardado como: {plot_2_filename}")
    print("\n--- ACTIVIDAD COMPLETADA ---")