# Hayden Feddock
# 4/2/2023

import numpy as np
import sigmoid

# Calculate the gradient of the sigmoid function
def sigmoidGradient(z):
    return sigmoid.sigmoid(z) * (1 - sigmoid.sigmoid(z))