# Hayden Feddock
# ECE 1395 - Dr. Dallal
# HW 6 - Bayesian Classifier
# 3/19/2023

import numpy as np
import scipy.io

''' Question 0: Data Preprocessing '''

# Load in the data from the .mat file
data = scipy.io.loadmat('input/hw4_data3.mat')

# Seperate the data
X_train = np.array(data['X_train'])
y_train = np.array(data['y_train'])
X_test = np.array(data['X_test'])
y_test = np.array(data['y_test'])

# Split X_train into three subsets based on the class from the y_train label vector
X_train_1 = X_train[y_train.flatten() == 1, :]
X_train_2 = X_train[y_train.flatten() == 2, :]
X_train_3 = X_train[y_train.flatten() == 3, :]

# Output: Sizes of the X_train_1, X_train_2, and X_train_3 feature matricies
print(f"Size of X_train_1: {X_train_1.shape}")
print(f"Size of X_train_2: {X_train_2.shape}")
print(f"Size of X_train_3: {X_train_3.shape}")

''' Question 1: Naive-Bayes Classifier '''

'''
Part a
- Calculate the mean and stdev for all the features of each class
'''

# Calculate the mean and standard deviation of each feature in class 1
X_mean_1 = np.mean(X_train_1, axis=0)
X_std_1 = np.std(X_train_1, axis=0)

# Calculate the mean and standard deviation of each feature in class 2
X_mean_2 = np.mean(X_train_2, axis=0)
X_std_2 = np.std(X_train_2, axis=0)

# Calculate the mean and standard deviation of each feature in class 3
X_mean_3 = np.mean(X_train_3, axis=0)
X_std_3 = np.std(X_train_3, axis=0)

# Output: Table displaying mean and standard deviation for each feature in each class
print(f"|         |     Feature 1     |     Feature 2     |     Feature 3     |     Feature 4     |")
print(f"|         |  Mean   |  Stdev  |  Mean   |  Stdev  |  Mean   |  Stdev  |  Mean   |  Stdev  |")
print(f"|---------|---------|---------|---------|---------|---------|---------|---------|---------|")
print(f"| Class 1 |{str(X_mean_1[0])[:9]}|{str(X_std_1[0])[:9]}|{str(X_mean_1[1])[:9]}|{str(X_std_1[1])[:9]}|{str(X_mean_1[2])[:9]}|{str(X_std_1[2])[:9]}|{str(X_mean_1[3])[:9]}|{str(X_std_1[3])[:9]}|")
print(f"| Class 2 |{str(X_mean_2[0])[:9]}|{str(X_std_2[0])[:9]}|{str(X_mean_2[1])[:9]}|{str(X_std_2[1])[:9]}|{str(X_mean_2[2])[:9]}|{str(X_std_2[2])[:9]}|{str(X_mean_2[3])[:9]}|{str(X_std_2[3])[:9]}|")
print(f"| Class 3 |{str(X_mean_3[0])[:9]}|{str(X_std_3[0])[:9]}|{str(X_mean_3[1])[:9]}|{str(X_std_3[1])[:9]}|{str(X_mean_3[2])[:9]}|{str(X_std_3[2])[:9]}|{str(X_mean_3[3])[:9]}|{str(X_std_3[3])[:9]}|")
print('\n')

'''
Part b
- For each feature, calculate the probability of each class p(x_j/w_1), p(x_j/w_2), and p(x_j/w_3)
- Calculate ln of the probability of each class
- Compute the log of the posterior probability of each class
- Classify the test data
- Calculate the accuracy of the classifier
'''

# Function to calculate the probability of a feature given a class
def prob(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

# Calculate the probability of each feature given each class
p_xj_given_w1 = prob(X_test, X_mean_1, X_std_1)
p_xj_given_w2 = prob(X_test, X_mean_2, X_std_2)
p_xj_given_w3 = prob(X_test, X_mean_3, X_std_3)

# Calculate the log of the probability of each class
ln_p_x_given_w1 = np.sum(np.log(p_xj_given_w1), axis=1)
ln_p_x_given_w2 = np.sum(np.log(p_xj_given_w2), axis=1)
ln_p_x_given_w3 = np.sum(np.log(p_xj_given_w3), axis=1)

# Compute the log of the posterior probability of each class
ln_p_w1 = np.log(1/3)
ln_p_w2 = np.log(1/3)
ln_p_w3 = np.log(1/3)
ln_p_w1_given_x = ln_p_x_given_w1 + ln_p_w1
ln_p_w2_given_x = ln_p_x_given_w2 + ln_p_w2
ln_p_w3_given_x = ln_p_x_given_w3 + ln_p_w3

# Classify the test data (add one to offset for class 1, 2, and 3)
predictions = np.argmax([ln_p_w1_given_x, ln_p_w2_given_x, ln_p_w3_given_x], axis=0) + 1

# Calculate the accuracy of the classifier (flatten y_test to match predictions shape)
accuracy = np.mean(predictions == y_test.flatten())
print(f"Accuracy: {accuracy}")

''' Question 2: Max Likelihood and Discriminant Function for Classification '''

'''
Part a
- Calculate the covariance matrix for each of the three classes
'''

# Calculate the covariance matrix for each class
Sigma_1 = np.cov(X_train_1, rowvar=False)
Sigma_2 = np.cov(X_train_2, rowvar=False)
Sigma_3 = np.cov(X_train_3, rowvar=False)

# Output the size of the covariance matrix for each class
print(f"Size of Sigma_1: {Sigma_1.shape}")
print(f"Size of Sigma_2: {Sigma_2.shape}")
print(f"Size of Sigma_3: {Sigma_3.shape}")

# Output a snapshot of the covariance matrix for each class
print(f"Snapshot of covariance matrix for class 1:")
print(Sigma_1)
print(f"Snapshot of covariance matrix for class 2:")
print(Sigma_2)
print(f"Snapshot of covariance matrix for class 3:")
print(Sigma_2)

'''
Part b
- Calculate the mean vector for each of the three classes
'''

# (using the mean vectors calculated in part a)

# Output the size of the mean vector for each class
print(f"Size of u1: {X_mean_1.shape}")
print(f"Size of u2: {X_mean_2.shape}")
print(f"Size of u3: {X_mean_3.shape}")

# Output a snapshot of the mean vector for each class
print(f"Snapshot of mean vector for class 1:")
print(X_mean_1)
print(f"Snapshot of mean vector for class 2:")
print(X_mean_2)
print(f"Snapshot of mean vector for class 3:")
print(X_mean_3)

'''
Part c
- Calculate the discriminant function for each class
- Classify the test data
- Calculate the accuracy of the classifier
'''

# Determine the number of features
l = X_train_1.shape[1]

# Calculate the C_i term for each class
C_1 = -1/2 * np.log(np.linalg.det(Sigma_1)) - l/2 * np.log(2 * np.pi)
C_2 = -1/2 * np.log(np.linalg.det(Sigma_2)) - l/2 * np.log(2 * np.pi)
C_3 = -1/2 * np.log(np.linalg.det(Sigma_3)) - l/2 * np.log(2 * np.pi)

# Initialize the discriminant function for each class
g_1 = np.zeros((X_test.shape[0], 1))
g_2 = np.zeros((X_test.shape[0], 1))
g_3 = np.zeros((X_test.shape[0], 1))

# Loop throguh each test data point
for i in range(X_test.shape[0]):
    
    # Select the ith test data point
    x = X_test[i, :]
    
    # Calculate the discriminant function for each class
    g_1[i] = -1/2 * (x - X_mean_1) @ np.linalg.inv(Sigma_1) @ (x - X_mean_1).T + ln_p_w1 + C_1
    g_2[i] = -1/2 * (x - X_mean_2) @ np.linalg.inv(Sigma_2) @ (x - X_mean_2).T + ln_p_w2 + C_2
    g_3[i] = -1/2 * (x - X_mean_3) @ np.linalg.inv(Sigma_3) @ (x - X_mean_3).T + ln_p_w3 + C_3

# Assign each testing entry to a class with the maximum discriminant function
predictions = np.argmax([g_1, g_2, g_3], axis=0) + 1

# Calculate the accuracy of the classifier
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy}")

# Compare the naive classifer and the MLE based classifier
print("The accuracy of the MLE based classifier is higher than the naive classifier. This can be explained by the use of the covariance matrix in the MLE classifier which is able to capture dependent features better than the naive classifier which assumes independence between features.")

