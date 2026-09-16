import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Returns a float64 zero hidden-state matrix.
    """

    size = [[0] * hidden_dim] * batch_size

    return np.array(size).astype(np.float64)