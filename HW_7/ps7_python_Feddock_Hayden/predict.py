# Hayden Feddock
# 3/30/2023

import numpy as np
import sigmoid

# Function that performs feed-forward propagation algorithm
def predict(Theta1, Theta2, X):
    
    # Create the bias term a^1_0
    a1_0 = np.ones([X.shape[0], 1])
    
    # Add the bias term to the inputs a^1
    a1 = np.hstack([a1_0, X])
    
    # Create the hidden layer a^2 by computing the sigmoid function of the dot product of theta1 and a^1
    a2 = sigmoid.sigmoid(a1 @ Theta1.T)
    
    # Create the bias term a^2_0
    a2_0 = np.ones([a1.shape[0], 1])
    
    # Add the bias term to the inputs a^2
    a2 = np.hstack([a2_0, a2])
    
    # Create the output layer h_x by computing the sigmoid function of the dot product of theta2 and a^2
    h_x = sigmoid.sigmoid(a2 @ Theta2.T)
    
    # Predict the label for each class as the output with the highest probability (add 1 for proper class)
    p = np.argmax(h_x, axis=1) + 1
    
    # Return the predicted class label and the array of output probabilities
    return [p, h_x]