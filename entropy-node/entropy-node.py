import numpy as np
from collections import defaultdict

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    if len(y) == 0:
        return 0.0
    connections = defaultdict(int)
    total = len(y)

    for i in y:
        connections[i] += 1

    answer = 0

    #np.log(2)

    for i in connections:
        # key: 1, 2, 3

        proportion = connections[i] / total 

        answer += (-proportion) * np.log2(proportion)

    return answer

    
