import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Validation check: probabilities must sum to approximately 1.0
    if not np.isclose(sum(p), 1.0):
        raise ValueError("The probabilities in 'p' must sum to 1.")
        
    total = 0
    for i in range(len(x)):
        total += x[i] * p[i]
    return total
