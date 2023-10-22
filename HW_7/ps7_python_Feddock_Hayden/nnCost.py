# Hayden Feddock
# 4/2/2023

import numpy as np
import predict

# Compute the cost of the nueral network (with regularization)
def nnCost(Theta1, Theta2, X, y, K, lam):

    # Compute the length of the X matrix
    m = X.shape[0]
    
    # Compute the output of the nueral network
    h_x = predict.predict(Theta1, Theta2, X)[1]
    
    '''
    Compute the regularization term
    
        Theta1.shape = [8, 5]
        Theta2.shape = [3, 9]
        
        Skip over the bias (first) term of each row (first array column) using slicing: [:, 1:]
        
        Element-wise square each of the theta values using the numpy square function (** 2)
        
        Sum all of the elements in the array using the numpy sum function
    '''
    regularized_weight = (lam / (2 * m)) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))
    
    '''
    Compute the cost (without regularized term)
        
        y.shape = [150, 3]
        h_x.shape = [150, 3]
        
        Element-wise multiplication using numpy multiply function (*)
        
        Sum all elements using the numpy sum function
        
        Take the element-wise log using the numpy log function
        
        By multiplying the y matrix (each row has a 1 in the index coresponding to a class and zeros elsewhere)
        we isolate the logged values from h_x that correspond to the class
    '''
    J_unregularized = -(1 / m) * np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x))
    
    # Return the cost with the regularized weight
    return J_unregularized + regularized_weight