#Red de Hopfield, 04 de nov, 2025 - JAIME
import numpy as np

class Hopfield:

    def _hebbian(self, S):
        n,p = S.shape
        self.W += (1/n) * S @ S.T
        np.fill_diagonal(self.W, 0)
    
    def _pinv(self, S):
        n,p = S.shape
        self.W += (1/n) * S @ np.linalg.pinv(S)
        np.fill_diagonal(self.W, 0)
    
    def _storkey(self, S):
        n, p = S.shape
        h = self.W @ S
        self.W += (1/n) * (S @ S.T - S @ h.T + h @ S.T)
        np.fill_diagonal(self.W, 0)

    def __init__(self, neurons, mode='hebbian'):
        self.n = neurons
        self.W = np.zeros((neurons, neurons))
        modes = {'hebbian': self._hebbian,
                 'pinv': self._pinv,
                 'storkey': self._storkey}
        self.fit = modes[mode]

    def prediction(self, S, max_iter=1):
        for i in range(max_iter):
            S = np.sign(self.W @ S)
            return S

         