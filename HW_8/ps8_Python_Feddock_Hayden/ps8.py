# Hayden Feddock
# ECE 1395 - Dr. Dallal
# Assignment 8
# 4/13/2023

import numpy as np
import matplotlib.pyplot as plt

''' Question 1: Bagging and Handwritten-Digits Classification '''

'''
Part a
- Download the mat file and load data
- Randomly pick 25 images
- Display the random images as a 5x5 grid
- Save output figure
'''

from scipy.io import loadmat

# Load in the mat file data
data = loadmat('input/HW8_data1.mat')
X = data['X']
y = data['y']

# Randomly select 25 images
rand_img_indicies = np.random.choice(X.shape[0], 25, replace=False)

# Create a plot for a 5x5 grid of subplots
grid_size = 5
fig, axs = plt.subplots(grid_size, grid_size)

# Add each images as subplots to the plot
for i in range(grid_size):
    for j in range(grid_size):
        
        # Extract the image and reshape it as a 20x20 image
        img = X[rand_img_indicies[i * grid_size + j]].reshape((20,20), order='F')
        
        # Add the image to the grid
        axs[i, j].imshow(img, cmap='gray')
        axs[i, j].axis('off')
        
# Save the plot to output
fig.savefig('output/ps8_1_a_1.png')

'''
Part b
- Randomly split the data into 4500 training samples and 500 testing samples
'''

from sklearn.model_selection import train_test_split

# Randomly split the data into 4500 training samples and 500 testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, )

'''
Part c
- Create 5 equally sized subsets of 1000 randomly selected samples from the X_train
- Save subsets as .mat files in the input folder
'''

from scipy.io import savemat

# Create the pointer to the subsets
X_train_sub = []
y_train_sub = []

# Create and fill the subsets
for i in range(5):
    
    # Select indicies from X_train
    rand_subset_indicies = np.random.choice(X_train.shape[0], 1000, replace=True)
    
    # Fill the X and y subsets with the data at the indicies from X_train
    X_train_sub.append(X_train[rand_subset_indicies])
    y_train_sub.append(y_train[rand_subset_indicies])
    
    # Save subset to a .mat file
    savemat(f"input/X_train_subset{i+1}.mat", {'X':X_train_sub[i], 'y':y_train_sub[i]})

'''
Part d
- Train a 1 vs all SVM 10-class classifier, with RPF kernel, using subset X1
- Compute the classification error on the training set X1
- Compute the classification error on the other training subsets
- Compute the classification error on the testing set
'''

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# Create the 1 vs all svm as a 10-class classifier with an RPF kernal
svm = OneVsRestClassifier(SVC(kernel='rbf', gamma='scale'))

# Train the classifier on the first subset of training data
svm.fit(X_train_sub[0], y_train_sub[0].ravel())

# Create the pointer to the training error of each subset
train_sub_err = []

# Iterate over each subset
for i in range(5):
    
    # Compute the classification error on each of the training subsets (1 - accuracy)
    train_sub_err.append(1 - svm.score(X_train_sub[i], y_train_sub[i].ravel()))

    # Print out the classification errors for each of the training subsets
    print(f"SVM classification error for training subset {i+1}: {train_sub_err[i]}")
    
# Compute the classification error on the testing set
test_err = 1 - svm.score(X_test, y_test.ravel())
print(f"SVM classification error for testing data: {test_err}")

'''
Part e
- Train a KNN for k = 7 using subset X2
- Compute the classification error on all of the training subsets
- Compute the classification error on the testing set
'''

from sklearn.neighbors import KNeighborsClassifier

# Create the KNN classifier for k = 7
knn = KNeighborsClassifier(n_neighbors=7)

# Train the classifier on the second subset of training data
knn.fit(X_train_sub[1], y_train_sub[1].ravel())

# Create the pointer to the training error of each subset
train_sub_err = []

# Iterate over each subset
for i in range(5):
    
    # Compute the classification error on each of the training subsets (1 - accuracy)
    train_sub_err.append(1 - knn.score(X_train_sub[i], y_train_sub[i].ravel()))

    # Print out the classification errors for each of the training subsets
    print(f"KNN classification error for training subset {i+1}: {train_sub_err[i]}")
    
# Compute the classification error on the testing set
test_err = 1 - knn.score(X_test, y_test.ravel())
print(f"KNN classification error for testing data: {test_err}")

'''
Part f
- Train a logistic regression classifier using subset X3
- Compute the classification error on all of the training subsets
- Compute the classification error on the testing set
'''

from sklearn.linear_model import LogisticRegression

# Create a logistic regression classifier
lr = LogisticRegression(max_iter=500)

# Train the classifier on the third subset of training data
lr.fit(X_train_sub[2], y_train_sub[2].ravel())

# Create the pointer to the training error of each subset
train_sub_err = []

# Iterate over each subset
for i in range(5):
    
    # Compute the classification error on each of the training subsets (1 - accuracy)
    train_sub_err.append(1 - lr.score(X_train_sub[i], y_train_sub[i].ravel()))

    # Print out the classification errors for each of the training subsets
    print(f"Logistic Regression classification error for training subset {i+1}: {train_sub_err[i]}")
    
# Compute the classification error on the testing set
test_err = 1 - lr.score(X_test, y_test.ravel())
print(f"Logistic Regression classification error for testing data: {test_err}")

'''
Part g
- Train a decision tree classifier using subset X4
- Compute the classification error on all of the training subsets
- Compute the classification error on the testing set
'''

from sklearn.tree import DecisionTreeClassifier

# Create a decision tree classifier
dt = DecisionTreeClassifier()

# Train the classifier on the fourth subset of training data
dt.fit(X_train_sub[3], y_train_sub[3].ravel())

# Create the pointer to the training error of each subset
train_sub_err = []

# Iterate over each subset
for i in range(5):
    
    # Compute the classification error on each of the training subsets (1 - accuracy)
    train_sub_err.append(1 - dt.score(X_train_sub[i], y_train_sub[i].ravel()))

    # Print out the classification errors for each of the training subsets
    print(f"Decision Tree classification error for training subset {i+1}: {train_sub_err[i]}")
    
# Compute the classification error on the testing set
test_err = 1 - dt.score(X_test, y_test.ravel())
print(f"Decision Tree classification error for testing data: {test_err}")

'''
Part h
- Train a random forest classifier with 75 tree using subset X5
- Compute the classification error on all of the training subsets
- Compute the classification error on the testing set
'''

from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier with 75 trees
rf = RandomForestClassifier(n_estimators=75)

# Train the classifier on the fifth subset of training data
rf.fit(X_train_sub[4], y_train_sub[4].ravel())

# Create the pointer to the training error of each subset
train_sub_err = []

# Iterate over each subset
for i in range(5):
    
    # Compute the classification error on each of the training subsets (1 - accuracy)
    train_sub_err.append(1 - rf.score(X_train_sub[i], y_train_sub[i].ravel()))

    # Print out the classification errors for each of the training subsets
    print(f"Random forest classification error for training subset {i+1}: {train_sub_err[i]}")
    
# Compute the classification error on the testing set
test_err = 1 - rf.score(X_test, y_test.ravel())
print(f"Random forest classification error for testing data: {test_err}")

'''
Part i
- Use majority voting rule to combine the output of the five classifiers and report the error rate on the testing set
'''

# Generate the predictions for X_test using each of the 5 trained classifiers
svm_p = svm.predict(X_test)
knn_p = knn.predict(X_test)
lr_p = lr.predict(X_test)
dt_p = dt.predict(X_test)
rf_p = rf.predict(X_test)

from scipy.stats import mode

# Use majority voting to determine the output
mv_p = mode(np.array([svm_p, knn_p, lr_p, dt_p, rf_p]), axis=0, keepdims=False)[0]

from sklearn.metrics import accuracy_score

# Compute the classification error on the testing set
test_err = 1 - accuracy_score(y_test.ravel(), mv_p)
print(f"Majority voting classification error for testing data: {test_err}")

