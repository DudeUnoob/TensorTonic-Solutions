import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.array(x)
    conv1 = np.array(conv1)
    W1_b1 = np.array(W1_b1)
    W2_b1 = np.array(W2_b1)
    W1_b2 = np.array(W1_b2)
    W2_b2 = np.array(W2_b2)
    Ws_b2 = np.array(Ws_b2)
    fc = np.array(fc)

    # Initial conv + ReLU
    out = x @ conv1
    out = np.maximum(0, out)

    # Block 1: identity shortcut
    identity = out
    residual = out @ W1_b1
    residual = np.maximum(0, residual)
    residual = residual @ W2_b1
    out = np.maximum(0, residual + identity)

    # Block 2: projection shortcut
    identity = out @ Ws_b2
    residual = out @ W1_b2
    residual = np.maximum(0, residual)
    residual = residual @ W2_b2
    out = np.maximum(0, residual + identity)

    # Fully connected classifier
    logits = out @ fc

    return logits