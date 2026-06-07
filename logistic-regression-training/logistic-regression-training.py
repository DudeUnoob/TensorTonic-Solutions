import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    N, D = X.shape
    w = np.zeros(D)      # instead of X.shape[1]
    b = 0.0

    for _ in range(steps):
        z = X @ w + b
        p = _sigmoid(z)

        dLdw = (1.0 / N) * (X.T @ (p - y))
        dldb = (1.0 / N) * np.sum(p - y)

        w = w - lr * dLdw
        b = b - lr * dldb

    return w, b