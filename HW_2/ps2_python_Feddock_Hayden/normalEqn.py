import numpy as np

def normalEqn (X_train, y_train):
    
    # Use the derived equation to calculate the values for theta
    theta = np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.matmul(np.transpose(X_train), y_train))
    
    return theta