import numpy as np
import matplotlib.pyplot as plt
class Adaline:
    "ADALINE con métodos de entrenamiento por descenso de gradiente y pseudoinversa."
    
    def __init__(self, lr=0.01, epochs=5000, tol=1e-6, batch_size=None, random_state=None):
        self.lr = lr  
        self.epochs = epochs  
        self.tol = tol  
        self.batch_size = batch_size 
        self.random_state = random_state  
        self.w_ = None  
        self.cost_ = []  

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _net_input(self, X):

        return np.dot(X, self.w_)

    def fit_bgd(self, X, y):
        "Batch Gradient Descent (Descenso de Gradiente por Lotes)"
        rgen = np.random.RandomState(self.random_state)
        
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        X_b = self._add_bias(X)

        for i in range(self.epochs):
            
            output = self._net_input(X_b)
            errors = y - output
            self.w_ += self.lr * X_b.T.dot(errors)
            cost = (errors**2).mean()
            self.cost_.append(cost)
            if len(self.cost_) > 1 and abs(self.cost_[-2] - self.cost_[-1]) < self.tol:
                break
        return self

    def fit_sgd(self, X, y):
        "Stochastic Gradient Descent (Descenso de Gradiente Estocástico)"
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        X_b = self._add_bias(X)

        for i in range(self.epochs):
            indices = rgen.permutation(len(y))
            for idx in indices:
                xi, yi = X_b[idx:idx+1], y[idx:idx+1]
                output = self._net_input(xi)
                errors = yi - output
                self.w_ += self.lr * xi.T.dot(errors)
            
            output = self._net_input(X_b)
            errors = y - output
            cost = (errors**2).mean()
            self.cost_.append(cost)
            if len(self.cost_) > 1 and abs(self.cost_[-2] - self.cost_[-1]) < self.tol:
                break
        return self

    def fit_mbgd(self, X, y, batch_size=16):
        "Mini-batch Gradient Descent (Descenso de Gradiente por Mini-lotes)"
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)
        X_b = self._add_bias(X)
        self.batch_size = batch_size
        n_samples = len(y)

        for i in range(self.epochs):
            indices = rgen.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                X_batch, y_batch = X_b[batch_indices], y[batch_indices]
                

                output = self._net_input(X_batch)
                errors = y_batch - output
                self.w_ += self.lr * X_batch.T.dot(errors)
            
            output = self._net_input(X_b)
            errors = y - output
            cost = (errors**2).mean()
            self.cost_.append(cost)
            if len(self.cost_) > 1 and abs(self.cost_[-2] - self.cost_[-1]) < self.tol:
                break
        return self

    def fit_pinv(self, X, y): #Pseudoinversa

        X_b = self._add_bias(X)
        self.w_ = np.linalg.pinv(X_b).dot(y)
        self.cost_ = [((y - self._net_input(X_b))**2).mean()]
        return self

    def predict(self, X):
        """Realiza predicciones después del entrenamiento."""
        return self._net_input(self._add_bias(X))

if __name__ == '__main__':
    # Genera datos de ejemplo para la demostración
    np.random.seed(42)
    X = np.linspace(0, 100, 100).reshape(-1, 1) + np.random.randn(100, 1) * 2
    y = 50 + 2 * X.flatten() + np.random.randn(100) * 15
    
    # Normaliza la variable independiente (x)
    X_norm = (X - X.mean()) / X.std()

    # Entrena cada modelo
    adalines = {
        'BGD': Adaline(lr=0.01, epochs=5000, random_state=1),
        'SGD': Adaline(lr=0.001, epochs=5000, random_state=1),
        'mBGD': Adaline(lr=0.01, epochs=5000, batch_size=16, random_state=1),
        'Pseudoinversa': Adaline()
    }

    results = {}
    for name, model in adalines.items():
        if name == 'Pseudoinversa':
            model.fit_pinv(X_norm, y)
        else:
            model.fit_bgd(X_norm, y) if name == 'BGD' else (
                model.fit_sgd(X_norm, y) if name == 'SGD' else model.fit_mbgd(X_norm, y)
            )
        results[name] = {
            'weights': model.w_,
            'cost': model.cost_[-1],
            'predictions': model.predict(X_norm)
        }
        
    # Des-normaliza los datos para las gráficas
    X_unnorm = X.flatten()
    
    # Grafica los resultados de cada método
    plt.figure(figsize=(15, 10))
    for i, (name, res) in enumerate(results.items()):
        plt.subplot(2, 2, i + 1)
        plt.scatter(X_unnorm, y, marker='x', color='orange')
        
        # Usa los pesos para generar la línea de regresión
        intercept = res['weights'][0]
        slope = res['weights'][1]
        line = intercept + slope * X_norm.flatten() * X.std() + X.mean()
        plt.plot(X_unnorm, line, color='darkorange', linewidth=2)
        
        plt.title(f'{name} (MSE: {res["cost"]:.4f})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Grafica la evolución del MSE
    plt.figure(figsize=(8, 6))
    for name, model in adalines.items():
        if name != 'Pseudoinversa':
            plt.plot(range(len(model.cost_)), model.cost_, label=f'{name}')
    plt.title('Evolución del MSE por época (BGD, SGD, mBGD)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
