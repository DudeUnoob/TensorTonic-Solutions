import numpy as np

def conv_block(x, W1, W2, Ws):
    h = np.maximum(np.array(x) @ np.array(W1), 0)
    z = h @ np.array(W2)
    s = np.array(x) @ np.array(Ws)

    return np.maximum(z + s, 0)
