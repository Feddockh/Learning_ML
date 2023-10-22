# Hayden Feddock
# ECE 1935 - Dr. Dallal
# 3/29/2023
# Assignment 7

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

''' Questions 0 '''

# Load in the data from the .mat file
data = scipy.io.loadmat('input/HW7_Data.mat')

# Extract the data
X = np.array(data['X'])
y = np.array(data['y'])

# Verify the dimensions
print(f"Dimensions of X: {X.shape}")
print(f"Dimensions of y: {y.shape}")

''' Question 1: Forward Propagation '''

'''
Part a
- Write a function that returns the predicted the class label for every example (q)
'''

# Import the function to predict the feed-forward result
import predict

'''
Part b
- Load in pre-trained weight matricies Theta1 and Theta2
- Call the function to compute the predictions based on the feed-forward propagation
- Check the accuracy of the predictions
'''

# Load in the weights from the .mat file
weights = scipy.io.loadmat('input/HW7_weights_2.mat')

# Extract the weights
Theta1 = np.array(weights['Theta1'])
Theta2 = np.array(weights['Theta2'])

# Call the function to predict the class labels for each example
p, h_x = predict.predict(Theta1, Theta2, X)

# Check the accuracy of the data
accuracy = np.mean(p == y.flatten())
print(f"Accuracy: {accuracy}")


''' Question 2: Cost Function '''

'''
Part a
- Write a function that computes the cost of the neural network
- Recode the labels as vectors containing only values 0 or 1
'''

# Import the function to compute the cost
import nnCost

# Count the number of classes
num_classes = len(np.unique(y))

# Recode y as a vector or vectors containing only values 0 or 1
def recode(y):
    # Determine the number of rows in the y label vector
    m = y.shape[0]
    
    # Create a new y matrix to store the recoded class as a 1 in the 
    y_recoded = np.zeros([m, num_classes])
    
    for i in range(m):
        i_class = y[i]
        y_recoded[i, i_class - 1] = 1
        
    return y_recoded

y_recoded = recode(y)

'''
Part b
- Use the nnCost function to compute the cost when lambda is 0, 1, and 2
'''

# Compute the cost when lambda = 0
J_0 = nnCost.nnCost(Theta1, Theta2, X, y_recoded, num_classes, 0)
print(f"The cost when lambda = 0 is: {J_0}")

# Compute the cost when lambda = 1
J_1 = nnCost.nnCost(Theta1, Theta2, X, y_recoded, num_classes, 1)
print(f"The cost when lambda = 1 is: {J_1}")

# Compute the cost when lambda = 2
J_2 = nnCost.nnCost(Theta1, Theta2, X, y_recoded, num_classes, 2)
print(f"The cost when lambda = 2 is: {J_2}")

''' Question 3: Derivative of the Activation Function (Sigmoid Gradient) '''

'''
Part a
- Write a function to calculate the gradient of the sigmoid function
- Test the function
'''

# Import the function to compute the sigmoid gradient
import sigmoidGradient

# Test the function with input z
z = np.array([-10, 0, 10])

# Compute g_prime (values should be [0, 0.25, 0])
g_prime = sigmoidGradient.sigmoidGradient(z)
print(f"The sigmoid gradient of z = [-10, 0, 10] is: {g_prime}")


''' Question 4: Backpropagation for Gradient of Cost Function and Stochastic Gradient Descent '''

# Import the module with the functiont to compute the ideal thetas
import sGD

'''
Part d
- Find and report the alpha value used to update the weights
'''

# This alpha was found by trial and error
alpha = 0.005

# Output the value used to compute the alpha
print(f"The value used for alpha: {alpha}")

'''
Part e
- Compute the figure that shows the cost vs iteration using the training data
'''

# Compute the costs at each epoch by running the function to compute the thetas
Theta1, Theta2 = sGD.sGD(X.shape[1], 8, num_classes, X, y_recoded, 1, alpha, 50)
            
# Output a figure that shows the cost vs epoch
epochs = range(1, len(sGD.costs) + 1)
plt.plot(epochs, sGD.costs)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.savefig("output/ps7-4-e-1.png")


''' Question 5: Testing the Network '''

'''
Part a
- Random split the data from HW7_Data.mat into 85% training and 15% testing sets
- Run the code for different values of lambda and epochs and report accuracy and cost with each case
'''

# Import the library used for the training and testing data split
import sklearn.model_selection

# Randomly split the data from X and y into 85% training and 15% testing data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.15, train_size=0.85)

# Import the tabulate library to create the table
from tabulate import tabulate

### Create a table to display the parameter accuracies as a function of the epochs and lambda values ###
# Create the list with the headers
headers = ["lambda", "Training Data Accuracy @ 50 Epochs", "Testing Data Accuracy @ 50 Epochs", "Training Data Accuracy @ 100 Epochs", "Testing Data Accuracy @ 100 Epochs"]

# Create the list with the data
data = [["lambda = 0", 0, 0, 0, 0],
        ["lambda = 0.01", 0, 0, 0, 0],
        ["lambda = 0.1", 0, 0, 0, 0],
        ["lambda = 1", 0, 0, 0, 0]]

# Compute the training and testing accuracies for 50 and 100 epochs
epochs = [50, 100]
for i in range(2):
    
    # Compute the training and testing data accuracy for each lambda
    lam = [0, 0.01, 0.1, 1]
    for j in range(4):
        
        # Compute the parameters with the new lambda value
        Theta1, Theta2 = sGD.sGD(X_train.shape[1], 8, num_classes, X_train, recode(y_train), lam[j], alpha, epochs[i])
        
        # Compute the training accuracy of the parameters using the predict function
        p_train = predict.predict(Theta1, Theta2, X_train)[0]
        training_accuracy = np.mean(p_train == y_train.flatten())
        data[j][1 + 2*i] = training_accuracy
        
        # Compute the testing accuracy of the parameters using the predict function
        p_test = predict.predict(Theta1, Theta2, X_test)[0]
        testing_accuracy = np.mean(p_test == y_test.flatten())
        data[j][2 + 2*i] = testing_accuracy

# Print out the training and testing accuracies
print(tabulate(data, headers, tablefmt='grid'))
    
    
### Create a table to display the parameter costs as a function of the epochs and lambda values ###
# Create the list with the headers
headers = ["lambda", "Training Data Cost @ 50 Epochs", "Testing Data Cost @ 50 Epochs", "Training Data Cost @ 100 Epochs", "Testing Data Cost @ 100 Epochs"]

# Create the list with the data
data = [["lambda = 0", 0, 0, 0, 0],
        ["lambda = 0.01", 0, 0, 0, 0],
        ["lambda = 0.1", 0, 0, 0, 0],
        ["lambda = 1", 0, 0, 0, 0]]

# Compute the training and testing costs for 50 and 100 epochs
epochs = [50, 100]
for i in range(2):
    
    # Compute the training and testing data costs for each lambda
    lam = [0, 0.01, 0.1, 1]
    for j in range(4):
        
        # Compute the parameters with the new lambda value
        Theta1, Theta2 = sGD.sGD(X_train.shape[1], 8, num_classes, X_train, recode(y_train), lam[j], alpha, epochs[i])
        
        # Compute the training costs of the parameters using the cost function
        p_train = predict.predict(Theta1, Theta2, X_train)[0]
        training_cost = nnCost.nnCost(Theta1, Theta2, X_train, recode(y_train), num_classes, lam[j])
        data[j][1 + 2*i] = training_cost
        
        # Compute the testing costs of the parameters using the cost function
        p_test = predict.predict(Theta1, Theta2, X_test)[0]
        testing_cost = nnCost.nnCost(Theta1, Theta2, X_test, recode(y_test), num_classes, lam[j])
        data[j][2 + 2*i] = testing_cost

# Print out the training and testing costs
print(tabulate(data, headers, tablefmt='grid'))
    



