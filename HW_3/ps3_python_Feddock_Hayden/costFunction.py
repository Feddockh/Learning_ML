# Write a function to calculate the cost function for logistic regression
from sigmoid import *
import numpy as np

def costFunction(theta, X_train, y_train):
       
    '''
    X = [[x_0^1, x_1^1, ..., x_n^1],
         [x_0^2, x_1^2, ..., x_n^2],
         [  .      .           .  ],
         [  .      .           .  ],
         [  .      .           .  ],
         [x_0^m, x_1^m, ..., x_n^m]]
    m x n
        
    y = [[y^1],
         [y^2],
         [ . ],
         [y^m]]
    m x 1
    
    theta = [[theta_0],
             [theta_1], 
             [   .   ],
             [theta_n]]
    n x 1
    
    '''
    m = y_train.shape[0] # Get the m dimension
    z = np.matmul(X_train, theta) # z is an m x 1 array
    h = sigmoid(z) # h is an m x 1 array
    
    # Calculate the cost function for logistic regression
    J = 1/m * (np.matmul(-y_train.T, np.log(h)) - np.matmul((1 - y_train).T, np.log(1 - h)))
    
    return J # Return the dimensionless value

