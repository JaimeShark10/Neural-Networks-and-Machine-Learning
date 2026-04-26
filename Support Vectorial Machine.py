import cvxopt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class SVM():
    def __init__(self, kernel='linear', C=0.001, gamma=0.001, degree=3):
        # Parámetros de funciones kernel
        self.C = float(C)
        self.gamma = float(gamma)
        self.d = int(degree)

        if kernel == 'linear':
            self.kernel = self.linear
        elif kernel == 'polynomial':
            self.kernel = self.polynomial
        elif kernel == 'gaussian':
            self.kernel = self.gaussian
        else:
            raise NameError('Kernel no reconocido')

    # Funciones Kernel
    def linear(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial(self, x1, x2):
        return (np.dot(x1, x2) + 1) ** self.d

    def gaussian(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    # Algoritmo de entrenamiento
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Matriz Gram
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # Resolver problema con cvxopt (Quadratic Programming)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        # Matriz G y vector h (constraints)
        if self.C == 0:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.identity(n_samples) * -1
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))

            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Extraer multiplicadores de Lagrange (alphas)
        alphas = np.ravel(solution['x'])
        
        # Guardar atributos necesarios para predicción
        sv = alphas > 1e-5
        self.a = alphas[sv]
        self.sv_x = X[sv]
        self.sv_y = y[sv]
        self.X = X
        self.y = y

        # Calcular bias
        self.compute_bias()

    # Calcular b (término de sesgo)
    def compute_bias(self):
        self.b = 0
        for i in range(len(self.a)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.a * self.sv_y *
                             np.array([self.kernel(self.sv_x[i], x) for x in self.sv_x]))
        self.b /= len(self.a)

    # Proyección de los puntos (f(x))
    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv_x in zip(self.a, self.sv_y, self.sv_x):
                s += a * sv_y * self.kernel(X[i], sv_x)
            y_predict[i] = s
        return y_predict + self.b

    # Predicción binaria
    def predict(self, X):
        return np.sign(self.project(X))

# --------------------------------------------------------------
# Visualización
# --------------------------------------------------------------

def plot_svm(X, y, model):
    plt.figure(figsize=(10, 10))
    
    # Dibujar datos de cada clase
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'bo', markersize=9)
    plt.plot(X[y == -1, 0], X[y == -1, 1], 'ro', markersize=9)
    
    # Calcular límites
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Dibujar superficie de decisión y márgenes
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                         np.linspace(ymin, ymax, 100))
    
    data = np.array([xx.ravel(), yy.ravel()]).T
    zz = model.project(data).reshape(xx.shape)
    
    plt.contour(xx, yy, zz, [0.0], colors='k', linewidths=2)
    plt.contour(xx, yy, zz, [-1.0, 1.0], colors='grey',
                linestyles='--', linewidths=2)

    plt.contourf(xx, yy, zz, [min(zz.ravel()), 0.0, max(zz.ravel())],
                 cmap=plt.cm.RdBu, alpha=0.5)
    plt.title("Máquina de Soporte Vectorial (SVM)")
    plt.show()

# --------------------------------------------------------------
# Datos de ejemplo (problema 1)
# --------------------------------------------------------------

np.random.seed(24)
mean1 = np.array([0, 2])
mean2 = np.array([2, 0])
cov = np.array([[1.5, 1], [1, 1.5]])
X1 = np.random.multivariate_normal(mean1, cov, 150)
X2 = np.random.multivariate_normal(mean2, cov, 150)
X = np.vstack((X1, X2))
y = np.hstack((np.ones(len(X1)), np.ones(len(X2)) * -1))  # etiquetas 1 y -1

# --------------------------------------------------------------
# Entrenar y graficar
# --------------------------------------------------------------

model = SVM(kernel='gaussian', C=2000, gamma=1)
model.fit(X, y)
plot_svm(X, y, model)