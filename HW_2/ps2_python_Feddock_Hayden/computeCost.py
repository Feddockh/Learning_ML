import numpy as np
import warnings

# Function to compute the cost
def computeCost(X, y, theta):
    
    # J: cost function (scalar)
    
    #       X = [[x_0^1, x_1^1, ..., x_n^1],
    #            [x_0^2, x_1^2, ..., x_n^2],
    #            [  .      .           .  ],
    #            [  .      .           .  ],
    #            [  .      .           .  ],
    #            [x_0^m, x_1^m, ..., x_n^m]]
    
    #       y = [[y^1],
    #            [y^2],
    #            [ . ],
    #            [ . ],
    #            [ . ],
    #            [y^m]]
    
    # theta = [theta_0, theta_1, ..., theta_n]
    # Make sure that theta is an 1 x m vector
    
    # Get the number of examples m and features (n + 1)
    [m, n] = np.shape(X)
    
    # Create an empty h array (m x 1)
    h = np.empty([m, 1], dtype=np.float64)
    
    # Multiply each n features of x with n thetas and sum together the products for each row of h
        # h = [[theta_0 * x_0^1 + theta_1 * x_1^1 + ... + theta_n * x_n^1],
        #      [theta_0 * x_0^2 + theta_1 * x_1^2 + ... + theta_n * x_n^2],
        #      [                              .                          ],
        #      [                              .                          ],
        #      [                              .                          ],
        #      [theta_0 * x_0^m + theta_1 * x_1^m + ... + theta_n * x_n^m]
    for i in range(m):
        rowProd = np.multiply(theta, X[i])
        h[i] = np.sum(rowProd)
    
    # Find the value of the summation
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    summation = np.sum(np.square(h - y))
    
    # Calculate the cost J
    J = (1 / (2 * m)) * summation
    
    return J