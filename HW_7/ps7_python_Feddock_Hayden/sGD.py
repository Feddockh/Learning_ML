# Hayden Feddock
# 4/5/2023

import numpy as np
import sigmoid
import sigmoidGradient
import nnCost
import matplotlib.pyplot as plt

# Create a global variable to hold the cost data to plot later
costs = []

# Function to compute the theta weights
def sGD(input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lam, alpha, MaxEpochs):
    
    # Clear any exists cost values from the global variable
    costs.clear()
    
    '''
    Part a
    - Define Theta1 and Theta2 and randomly initialize them with a uniform distribution from -0.1 to 0.1
    - Loop over each epoch and within each epoch perform backpropagation and gradient descent on each training example sequentially
    '''
    
    '''
    Theta_ji = Weight going from node i to node j
     __                                 __
    |   Theta_10               Theta_1i   |
    |   Theta_20   .   .   .   Theta_2i   |
    |   Theta_30               Theta_3i   |
    |       .                     .       |
    |       .                     .       |
    |       .                     .       |
    |__ Theta_j0   .   .   .   Theta_ji __|
    '''

    # Define Theta1 and Theta2 and randomly initialize them with values between -0.1 and 0.1 (add additional column for bias term)
    Theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1) * 0.2 - 0.1
    Theta2 = np.random.rand(num_labels, hidden_layer_size + 1) * 0.2 - 0.1
    
    # Determine the number of training examples
    m = X_train.shape[0]
    
    # Loop over each epoch
    for epoch in range(MaxEpochs):
        
        '''
        Part b
        - Compute each training sample sequentially
        - Compute a "forward pass" to compute the activations through the network
        - Compute an error term (delta) that measures how much a node was responsible for any error in the output
        - Compute the gradient of theta 1 and theta 2
        '''
        
        # Loop over each training example
        for q in range(m):
            
            # Compute feed forward propagation for training example
            a1 = np.hstack([1, X_train[q]]) # 1x5
            z2 = a1 @ Theta1.T # 1x5 @ 5x8 = 1x8
            a2 = np.hstack([1, sigmoid.sigmoid(z2)]) # 1x9
            z3 = a2 @ Theta2.T # 1x9 @ 9x3 = 1x3
            h_x = sigmoid.sigmoid(z3) # 1x3
            
            # Compute the backpropagation algorithm
            # Compute the deltas: the error of the nodes in the layer (leave out bias term)
            delta3 = h_x - y_train[q] # 1x3 - 1x3 = 1x3
            delta2 = Theta2[:,1:].T @ delta3 * sigmoidGradient.sigmoidGradient(z2) # 8x3 @ 3x1 * 1x8 = 1x8
            
            # Compute the partial derivatives of the cost over theta 1 and theta 2 (had to use outer function, equivalent to dot product (@) but can use for 1D arrays)
            dJ_dTheta1 = np.outer(delta2, a1) # 8x1 @ 1x5 = 8x5
            dJ_dTheta2 = np.outer(delta3, a2) # 3x1 @ 1x9 = 3x9
            
            '''
            Part c
            - Implement the regularization using an additional computation to the gradients
            - Do not regularize the bias term
            '''
            
            # Regularization (do not regularize the bias term in the first column)
            dJ_dTheta1[:, 1:] += lam * Theta1[:, 1:]
            dJ_dTheta2[:, 1:] += lam * Theta2[:, 1:]
            
            '''
            Part d
            - Update the theta 1 and theta 3 weight matrices
            '''
            
            # Update the theta weights
            Theta1 -= alpha * dJ_dTheta1
            Theta2 -= alpha * dJ_dTheta2
        
        # Compute the cost and add it to the list of costs to use for the figure later
        J = nnCost.nnCost(Theta1, Theta2, X_train, y_train, num_labels, lam)
        
        # Add the cost to the global cost variable
        costs.append(J)
    
    # Return the tuned theta weights
    return Theta1, Theta2