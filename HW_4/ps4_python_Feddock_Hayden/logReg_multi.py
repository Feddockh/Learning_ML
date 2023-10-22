import numpy as np
from sklearn.linear_model import LogisticRegression

def logReg_multi(X_train, y_train, X_test):
    
    # Determine the classes
    classes = np.unique(y_train)

    # Determine the number of classes
    numClasses = len(classes)

    # Create a list of the models
    models = []

    # For each class generate a model
    for i in range(numClasses):
        
        # Create a new training set for the label vector for the binary output
        y_train2 = np.zeros_like(y_train)
        
        # Label each value of class with a one, and the rest with a zero
        for j in range(y_train.shape[0]):
            if y_train[j] == classes[i]:
                y_train2[j] = 1
            else:
                y_train2[j] = 0
        
        # Add new model into the list of models
        models.append(LogisticRegression(random_state=0).fit(X_train, y_train2.ravel()))
        
    # Find the dimensions of the testing matrix
    d, n = X_test.shape
    
    # Create the y_predict output vector
    y_predict = np.zeros([d, 1])
    
    # Create an array of probabilites for each class
    proba = np.zeros([d, numClasses])
    for i in range(numClasses):
        
        # Generate a column with the probabilities of class[i] and add to array column
        proba[:, i] = models[i].predict_proba(X_test)[:, 1]
    
    # Once we have the full array of probabilities, now we can determine the label
    for i in range(d):
        
        # Determine the label at row i
        bestCandidate = [0] * 2
        for j in range(numClasses):
            
            # Keep track of the best candidate and update as needed
            if proba[i, j] > bestCandidate[1]:
                bestCandidate[0] = j
                bestCandidate[1] = proba[i, j]
                
        # Set the class in the predictions vector based on the best candidate's class
        y_predict[i] = classes[bestCandidate[0]]
           
    return y_predict