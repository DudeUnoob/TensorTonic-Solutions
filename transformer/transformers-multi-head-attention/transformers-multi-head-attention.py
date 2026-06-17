import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    def attention(query, key, value):
        scores = query @ key.transpose(0, 1, 3, 2)
        weights = softmax(scores / np.sqrt(key.shape[-1]), axis=-1)
        return weights @ value

    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    batch_size, seq_len, d_model = Q_proj.shape
    head_dim = d_model // num_heads

    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, head_dim)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, head_dim)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, head_dim)

    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)

    heads = attention(Q_heads, K_heads, V_heads)

    heads = heads.transpose(0, 2, 1, 3)
    concat = heads.reshape(batch_size, seq_len, d_model)

    output = concat @ W_o

    return output