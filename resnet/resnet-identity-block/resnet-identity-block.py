import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE

    W1 = np.asarray(W1)
    W2 = np.asarray(W2)

    x = np.asarray(x)

    h = np.maximum(0, x @ W1.transpose())

    return np.maximum(0, h @ W2.T + x)
    pass
