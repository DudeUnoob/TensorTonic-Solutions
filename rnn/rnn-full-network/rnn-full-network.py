import numpy as np

def vanilla_rnn(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, W_hy: np.ndarray, b_h: np.ndarray,
                b_y: np.ndarray) -> dict:
    hidden=h_0.copy()
    outputs=[]
    for step in range(X.shape[1]):
        hidden=np.tanh(X[:,step,:]@W_xh.T+hidden@W_hh.T+b_h)
        outputs.append(hidden@W_hy.T+b_y)
    return {"outputs":np.stack(outputs,axis=1).astype(np.float64),"final_hidden_state":hidden.astype(np.float64)}
