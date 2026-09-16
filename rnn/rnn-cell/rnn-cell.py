import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    # Force 1D arrays into 2D matrices, then transpose to make them column vectors
    return np.tanh(x_t@W_xh.T+h_prev@W_hh.T+b_h).astype(np.float64)