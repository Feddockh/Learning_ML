import numpy as np
from computeCost import *

# Question 2: Gradient Descent
def gradientDescent (x_train, y_train, alpha, iters):
    
    # alpha is the learning rate to use in the weight update
    # iters is the number of iterations to run gradient descent

    # x_train = [[x_0^1, x_1^1, ..., x_n^1],
    #            [x_0^2, x_1^2, ..., x_n^2],
    #            [  .      .           .  ],
    #            [  .      .           .  ],
    #            [  .      .           .  ],
    #            [x_0^m, x_1^m, ..., x_n^m]]
    
    # y_train = [[y^1],
    #            [y^2],
    #            [ . ],
    #            [ . ],
    #            [ . ],
    #            [y^m]]
    
    # Find shape of x array (m x n)
    [m, n] = np.shape(x_train)
    
    # Create randomized theta (1 x n)
    # theta = [theta_0, theta_1, ..., theta_n]
    theta = np.random.rand(1, n)

    # Create an empty h array (m x 1)
    h = np.empty([m, 1])
    
    # Create an empty cost array with a row for each iter
    cost = np.empty([iters, 1])
    
    # Iterate through the cost array
    for iter in range(iters):

        # Multiply each n features of x with n thetas and sum together the products for each row of h
        # h = [[theta_0 * x_0^1 + theta_1 * x_1^1 + ... + theta_n * x_n^1],
        #      [theta_0 * x_0^2 + theta_1 * x_1^2 + ... + theta_n * x_n^2],
        #      [                              .                          ],
        #      [                              .                          ],
        #      [                              .                          ],
        #      [theta_0 * x_0^m + theta_1 * x_1^m + ... + theta_n * x_n^m]
        for i in range(m):
            h[i] = np.sum(np.multiply(theta, x_train[i]))
        
        for j in range(n):
            
            # Find the value of the summation
            summation = 0
            for i in range(m):
                summation += (h[i] - y_train[i]) * x_train[i][j]
                
            # Update the j value of theta using theta, alpha, m, and the summation
            theta[0][j] = theta[0][j] - (alpha / m) * summation
        
        cost[iter] = computeCost(x_train, y_train, theta)
    
    return [theta[0], cost]
