import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Positions: shape (seq_length, 1)
    positions = np.arange(seq_length).reshape(-1, 1)  # [0, 1, ..., seq_length-1]

    # Dimension indices: shape (1, d_model)
    dims = np.arange(d_model).reshape(1, -1)

    # Compute the denominator term 10000^(2i/d_model) but we want pairs (sin, cos)
    # For each pair (2i, 2i+1) we use the same frequency, so we use dims//2
    div_term = np.power(10000.0, (2 * (dims // 2)) / d_model)

    # Argument to sin/cos: pos / 10000^(2i/d_model)
    angle_rads = positions / div_term  # shape (seq_length, d_model)

    # Initialize encoding matrix
    pe = np.zeros((seq_length, d_model), dtype=np.float64)

    # Even indices: sin
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Odd indices: cos
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pe