import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x = np.array(x, dtype=float)

    for i in range(len(x)):
        if x[i] < 0:
            x[i] = alpha * x[i]

    return x