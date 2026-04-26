import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Funciones

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
    pass 


# --- Graficación  ---

def single_dataset(df, ax, title):
    """
    Función auxiliar para graficar un único dataset.
    """
    df['y_class'] = np.where(df['y'] > 0.5, 1, 0)
    
    class_0 = df[df['y_class'] == 0]
    class_1 = df[df['y_class'] == 1]
    
    ax.scatter(class_0['x1'], class_0['x2'], c='red', marker='o', s=20, label='Clase 0')
    ax.scatter(class_1['x1'], class_1['x2'], c='blue', marker='o', s=20, label='Clase 1')
    
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) 

def plot_all_datasets():
    """
    Cargando los cuatro datasets
    """
    file_names = [
        "blobs.csv",
        "circles.csv",
        "moons.csv",
        "XOR.csv"
    ]

    titles = [
        "Datos 'Blobs'",
        "Datos 'Circles'",
        "Datos 'Moons'",
        "Datos 'XOR'"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten() 

    for i, file_name in enumerate(file_names):

        df = pd.read_csv(file_name, encoding='latin-1', decimal=',') 
        single_dataset(df, axes[i], titles[i])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title="Clase", fontsize='large', title_fontsize='x-large')

    # Ajuste y guardado
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("cuatro_datasets_clasificacion.png")
    print("\n✅ Gráfico de los 4 datasets generado como 'cuatro_datasets_clasificacion.png' correctamente.")
    plt.show()


# --- Graficación de los datasets y función principal ---
if __name__ == '__main__':
    plot_all_datasets()