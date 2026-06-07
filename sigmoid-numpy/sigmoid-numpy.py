import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """

    x = np.array(x, dtype=float)
    z = (1) / (1 + np.exp(-x))
    

    return z