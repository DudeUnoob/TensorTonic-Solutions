import numpy as np
import torch

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    hidden = h_0.copy()

    states = []


    for t in range(X.shape[1]):

        hidden = np.tanh(X[:, t, :] @ W_xh.T + hidden @ W_hh.T + b_h )
        states.append(hidden)

    return {"hidden_states" : np.stack(states, axis=1).astype(np.float64), "final_hidden_state": hidden.astype(np.float64)}

        

    
        

    