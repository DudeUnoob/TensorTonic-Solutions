import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    # Write code here
    sum = 0
    for xi, yi in zip(x, y):

        sum += (xi - yi) ** 2

    return sum ** 0.5
    pass