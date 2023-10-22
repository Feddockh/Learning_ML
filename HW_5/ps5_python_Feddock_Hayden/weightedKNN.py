import numpy as np
from scipy.spatial.distance import cdist

def weightedKNN(X_train, y_train, X_test, sigma):
    '''
    X_train is an m x n feature matrix
    y_train is an m x 1 label vector
    X_test is an d x n feature matrix
    sigma is a scalar denoting the bandwidth of the Gaussian weighing function
    y_predict is an d x 1 vector containing the predicted labels for the test instances
    '''
    
    '''
    From lecture notes:
    - For each test value, we must compare it to each of the training values
    - Calculate the distance that each training sample is from the testing sample (cdist)
    - Calculate the weight that each training sample coresponds to (Weight of i-th sample = exp(-distance^2/sigma^2))
    - Add together the weights for each class, and choose the class with the greatest value
    '''
    
    # Get the shape of the testing and training data
    m_train, n_train = np.shape(X_train)
    m_test, n_test = np.shape(X_test)
    
    # Create the y_predict label vector with the same number of rows as X_test
    y_predict = np.zeros([m_test, 1])
    
    
    # X_train: m_train x n_train matrix
    # X_test: m_test x n_test matrix
    # distances: m_train x m_test matrix (ex. each row m in column 0 is the distance between X_test[0] and X_train[m])
    # Get the distances between all of the training points for each testing point
    distances = cdist(X_train, X_test, metric='euclidean')
    
    # For each distance calculate the weight using the Gaussian weighted neighbors classifier
    weights = np.exp(-np.square(distances)/(sigma**2))
    
    # Find the unique classes from the y_train label vector
    classes = np.unique(y_train)
    
    # For each class, add up all of the weights and choose the class with the largest weight
    # Store the label for the X_test value in y_predict
    
    # For each example in the X_test feature matrix
    for i in range(m_test):
        
        # Keep track of the total number of classes (hashmap)
        class_value = {}
    
        # Calculate the sum of weights for each class
        for c in classes:

            # Calculate the sum of all of the weights in column i that match class c
            total = np.sum(weights[:, i].reshape(-1,1)[y_train == c])
            
            # Store the total as the value at key c
            class_value[c] = total
        
        # Determine the key (class label) with the largest value (weight)
        y_predict[i, 0] = max(class_value, key=class_value.get)
    
    return y_predict
    
    
    
    