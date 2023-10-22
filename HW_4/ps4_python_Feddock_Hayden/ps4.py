# Hayden Feddock
# ECE 1395 - Dr. Dallal
# 2/9/2023
# HW 4

import numpy as np
import scipy.io

''' Question 1: Regularization '''

'''
Part a:
- Write a function to compute the closed-form solution to linear regression using the normal equation with regularization
'''

# Function that returns the theta values from the normal equation with regularization
def Reg_normalEqn(X_Train, y_train, lam):
    
    # Create an identity matrix the size of X
    D = np.identity(X_Train.shape[1])
    
    # Set the first value to 0 since we don't regularize theta0
    D[0, 0] = 0
    
    # Calculate the inner part of the matrix to be inversed
    inner = X_Train.T @ X_Train + lam * D
    
    # Use pseudo-inverse in case singular matrix
    if np.linalg.det(inner) == 0:
        theta = np.linalg.pinv(inner) @ X_Train.T @ y_train
        
    else:
        theta = np.linalg.inv(inner) @ X_Train.T @ y_train
        
    return theta

'''
Part b:
- Read in mat file data into feature matrix X and label vector y
- Add offset feature x0 to X
'''

# Load data from the mat file into the X feature matrix and y label vector
data = scipy.io.loadmat('input/hw4_data1.mat')
X = np.array(data['X_data'])
y = np.array(data['y'])

# Create x0 vector of ones
X0 = np.ones([X.shape[0], 1])

# Add column x0 to data matrix X
X = np.hstack((X0, X))

# Text output:
print(f"The size of feature matrix X is: {X.shape}")

'''
Part c:
- Compute average training and testing error from 20 different models trained on this data
    - Split data into 88% training and 12% testing
    - Train eight linear regression models using lam = [0 0.001 0.003 0.005 0.007 0.009 0.012 0.017]
    - Compute the training and testing error for each value of lambda
'''

from costFunction import *

# Create an array to hold the average error of training and testing datasets
trainingError = np.zeros([20, 8])
testingError = np.zeros([20, 8])

# Create the vector of lambda values
lam = np.array([0, 0.001, 0.003, 0.005, 0.007, 0.009, 0.012, 0.017])

# Find the m dimension
m = X.shape[0]

# Use loop to create 20 models
for i in range(20):
    
    # Use permutation to randomly organize the rows of the X matrix and y vector
    randomized_rows = np.random.permutation(m)
    X = X[randomized_rows, :]
    y = y[randomized_rows]
    
    # Split re-ordered data into 12% testing and 88% training data
    training_rows = int(m * 0.88)
    X_train = X[:training_rows, :]
    y_train = y[:training_rows]
    X_test = X[training_rows:, :]
    y_test = y[training_rows:]
    
    # Train a linear regression model for each value of lambda
    for j in range(8):
        
        # Compute the theta values for each value of lambda
        theta = Reg_normalEqn(X_train, y_train, lam[j])
        
        # Calculate the training error
        trainingError[i, j] = costFunction(X_train, y_train, theta)
        
        # Calculate the testing error
        testingError[i, j] = costFunction(X_test, y_test, theta)

# Calculate the average error over each column (for each value of lambda)
avgTrainingError = np.mean(trainingError, axis=0)
avgTestingError = np.mean(testingError, axis=0)

# Import the matplotlib to plot arrays
import matplotlib.pyplot as plt

# Create the plots
plt.clf()
plt.plot(lam, avgTrainingError, color='red', label='Training Error')
plt.plot(lam, avgTestingError, color='blue', label='Testing Error')
plt.legend(loc='best')

# Add labels for the axis
plt.xlabel('Lambda Values')
plt.ylabel('Error')

# Output: Scatter plot of the training data
plt.savefig('output/ps4-1-c.png')

# Test response
print("I'd suggest a value of lambda between 0.005 and 0.0125 since the difference between the testing and training error is lowest and it has the lowest testing error.")


''' Question 2: KNN '''

'''
Part a:
- Load in the second .mat data file
- Compute the average accuracy over five folds for the values of K = 1:2:15
'''

# Load in the data from the .mat file
data = scipy.io.loadmat('input/hw4_data2.mat')

# Seperate the X matrices and y vectors from the data
X1 = np.array(data['X1'])
X2 = np.array(data['X2'])
X3 = np.array(data['X3'])
X4 = np.array(data['X4'])
X5 = np.array(data['X5'])
y1 = np.array(data['y1'])
y2 = np.array(data['y2'])
y3 = np.array(data['y3'])
y4 = np.array(data['y4'])
y5 = np.array(data['y5'])

# Assemble the X matrix and y vector
X = [X1, X2, X3, X4, X5]
y = [y1, y2, y3, y4, y5]

# Find the height of the data columns
m, n = X1.shape

# Import the libraries for the nearest neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier

# Train 5 folds for cross-validation
folds = 5

# Test different K values
K = list(range(1, 16, 2))

# Create an empty list to store the average accuracy of the same length as K
accuracy = [0] * len(K)

# Test each value of k
for i in range(len(K)):
    
    # Create the KNN classifier with K[i]
    KNN = KNeighborsClassifier(n_neighbors=K[i])
    
    # Test each combination of the 5 folds
    fold_accuracy = [0] * folds
    for j in range(folds):
        
        # Create the training and testing datasets by stacking other folds underneath of each other
        X_train = np.empty([0, n])
        y_train = np.zeros([0, 1])
        for fold in range(folds):
            if fold != j:
                X_train = np.vstack([X_train, X[fold]])
                y_train = np.vstack([y_train, y[fold]])
        X_test = X[j]
        y_test = y[j]
        
        # Test the datasets after folding
        KNN.fit(X_train, y_train.ravel())
        
        # Predict the labels for the test matrix
        y_pred = KNN.predict(X_test)
        
        # Count the number of predictions that match
        matches = 0
        for row in range(m):
            matches += 1 if y_pred[row] == y_test[row] else 0
        
        # Calculate the accuracy for each fold at the K value
        fold_accuracy[j] = matches / m
        
    # Average the values of the folds accuracy at a value in K
    accuracy[i] = sum(fold_accuracy) / folds
        
# Plot the average accuracy vs the k value
import matplotlib.pyplot as plt

# Create the plots
plt.clf()
plt.plot(K, accuracy)

# Add labels for the axis
plt.xlabel('K Values')
plt.ylabel('Average Accuracy')

# Output: Scatter plot of the training data
plt.savefig('output/ps4-2-a.png')

# Text output
print("I suggest a value of 9 for K according to the graph. This value would need to be redetermined for another problem since the accuracy of that K value is specifically determined for this data.")


''' Question 3: One vs All '''

'''
Part a:
- Write a function to implement the one vs all approach using logistic regression
'''

# Function file
from logReg_multi import *
    
'''
Part b:
- Load data in from .mat file
- 
'''

# Load in the data from the .mat file
data = scipy.io.loadmat('input/hw4_data3.mat')

# Seperate the data
X_train = np.array(data['X_train'])
y_train = np.array(data['y_train'])
X_test = np.array(data['X_test'])
y_test = np.array(data['y_test'])

# Compute the predictions using the one vs all approach for the training and testing data
y_predict_train = logReg_multi(X_train, y_train, X_train)
y_predict_test = logReg_multi(X_train, y_train, X_test)

# Calculate the accuracy of our predictions for the training and testing data
train_accuracy = np.mean(y_predict_train == y_train)
test_accuracy = np.mean(y_predict_test == y_test)

# Output
print(f"Training accuracy: {train_accuracy}, Testing accuracy: {test_accuracy}")   

# Text output
print("The accuracy of the predictions of the training data are higher than the accuracy of the predictions of the testing data. Both are pretty high accuracies.")
        



