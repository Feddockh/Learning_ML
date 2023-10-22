import numpy as np

def costFunction(X, y, theta):
    
    # Calculate m (number of examples)
    m = len(y)
    
    # Calculate the hypothesized values
    h = X @ theta
    
    # Calculate the cost
    J = (1 / (2 * m)) * np.sum(np.square(h - y))
    
    return J
    