import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)

    main = np.maximum(0, x @ W1)
    main = np.maximum(0, main @ W2)
    main = main @ W3

    shortcut = x @ Ws

    out = np.maximum(0, main + shortcut)

    return out