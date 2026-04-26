import numpy as np
import matplotlib.pyplot as plt

class DeltaNeuron:
    def __init__(self, n_inputs, activation_function):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.activation = activation_function

    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        return self.activation(z)

    def fit(self, x, y, lr=0.1, epochs=200):
        p = x.shape[1]
        error_history = []
        for _ in range(epochs):
            Y_est = self.predict(x)
            # Delta Rule
            delta = (Y_est - y)
            self.w -= (lr/p) * (delta @ x.T).ravel()
            self.b -= (lr/p) * np.sum(delta)
            error_history.append(self.cost(Y_est, y))
        return error_history

    def cost(self, Y_est, y):
        p = y.shape[1]
        return (1/(2*p)) * np.sum((y - Y_est)**2)

# Funciones de Activación
def linear_activation(z):
    return z

def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

# --- Ejemplos de uso ---

# 1. Problema de Regresión Lineal
# Generar datos
p = 100
x_lr = -1 + 2 * np.random.rand(p).reshape(1, -1)
y_lr = -18 * x_lr + 6 + np.random.randn(p)

# Entrenar la neurona
linear_neuron = DeltaNeuron(1, linear_activation)
error_history_lr = linear_neuron.fit(x_lr, y_lr, lr=0.1, epochs=200)

# Graficar resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_lr.ravel(), y_lr.ravel(), 'o', color='blue')
xn_lr = np.array([[-1, 1]])
plt.plot(xn_lr.ravel(), linear_neuron.predict(xn_lr), '--', color='red')
plt.title('Regresión Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend(['Datos de entrenamiento', 'Recta de regresión'])

plt.subplot(1, 2, 2)
plt.plot(error_history_lr, color='green')
plt.title('Historial de Error (MSE)')
plt.xlabel('Épocas')
plt.ylabel('MSE')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Problema de Compuertas Lógicas (AND, OR, XOR)
# Compuerda AND
x_and = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y_and = np.array([[0, 0, 0, 1]])
and_neuron = DeltaNeuron(2, sigmoid_activation)
and_neuron.fit(x_and, y_and, lr=1.0, epochs=5000)
predictions_and = (and_neuron.predict(x_and) > 0.5) * 1.0
print(f"Predicciones para AND: {predictions_and}")

# Compuerda OR
x_or = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y_or = np.array([[0, 1, 1, 1]])
or_neuron = DeltaNeuron(2, sigmoid_activation)
or_neuron.fit(x_or, y_or, lr=1.0, epochs=5000)
predictions_or = (or_neuron.predict(x_or) > 0.5) * 1.0
print(f"Predicciones para OR: {predictions_or}")

# Compuerda XOR
x_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
y_xor = np.array([[0, 1, 1, 0]])
xor_neuron = DeltaNeuron(2, sigmoid_activation)
xor_neuron.fit(x_xor, y_xor, lr=1.0, epochs=5000)
predictions_xor = (xor_neuron.predict(x_xor) > 0.5) * 1.0
print(f"Predicciones para XOR: {predictions_xor}")