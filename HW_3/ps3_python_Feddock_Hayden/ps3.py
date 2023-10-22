# Hayden Feddock
# ECE 1395 - Dr. Dallal
# HW - 3
# 2/4/2023

''' Question 1: Logistic Regression '''

'''
Part a: 
- Define the feature matrix X where each row coresponds to a feature example, and the label vector y
- Do not forget to append 1 for each feature vector which will correspond to the bias term theta0
'''

# Import numpy to use array functions
import numpy as np

data = np.loadtxt(fname='input/hw3_data1.txt', dtype=float, delimiter=',') # Load data from a txt file
m, n = data.shape # Get the shape of the data
X = data[:, :n-1] # Store all but the last column to matrix
y = data[:, n-1] # Store the last column to the y vector

# Append 1 for each feature vector that will correspond to the bias theta0
X_0 = np.ones((m, 1)) # Column of 1s for each feature vector
X = np.hstack((X_0, X)) # Append column of 1s to X matrix
m, n = X.shape # Get the new shape of the X matrix

# Text Output:
print(f"The size of feature matrix X is: {m, n}")
print(f"The size of feature label vector is: {m, 1}")

'''
Part b:
- Plot the exam 1 scores versus the exam 2 scores
- Change the color of the point based on the output
'''

# Import the matplotlib to plot arrays
import matplotlib.pyplot as plt

# Store the values to make up the axis of the plot
Exam1Score = X[:, 1] # Store the second column to the exam 1 score vector
Exam2Score = X[:, 2] # Store the third column to the exam 2 score vector

# Loop through each row in the array and set the color of the point based on the value from the y vector
for i in range(m):
    if y[i] == 1:
        plt.scatter(Exam1Score[i], Exam2Score[i], color='green')
    else:
        plt.scatter(Exam1Score[i], Exam2Score[i], color='red')

# Add a key using "dummy points" in order to prevent the key from being repetitive
plt.scatter([], [], color='green', label='Admitted')
plt.scatter([], [], color='red', label='Not Admitted')
plt.legend(loc='best')

# Add labels for the axis
plt.xlabel('Exam 1 Scores')
plt.ylabel('Exam 2 Scores')

# Output: Scatter plot of the training data
plt.savefig('output/ps3-1-b.png')

'''
Part c:
- Randomly divide about 90% of the data into 90% training data and the other 10% into testng data
'''

# Use permutation to randomly organize the rows of the X matrix and y vector
randomized_rows = np.random.permutation(m)
X = X[randomized_rows, :]
y = y[randomized_rows]

# Split re-ordered data into 10% testing and 90% training data
training_rows = int(m * 0.9)
X_train = X[:training_rows, :]
y_train = y[:training_rows]
X_test = X[training_rows:, :]
y_test = y[training_rows:]

'''
Part d:
- Compute the sigmoid function
- Plot g(z) versus z where z = [-15:15]
'''

# Function file: sigmoid.py containing function sigmoid()
from sigmoid import *

# Create a test z vector from -15 to 15
z = np.arange(-15, 16)
gz = sigmoid(z)

# Clear the current figure
plt.clf()

# Output: Plot gz versus z
plt.plot(z, gz)
plt.xlabel('z')
plt.ylabel('gz')
plt.savefig('output/ps3-1-c.png')

# Text response: At what value does this output reach 0.9?
print("Figure ps3-1-c reaches an output of 0.9 with an input of about 2.5")

'''
Plot e:
- Implement the cost function and the gradient of the cost function for logistic regression
'''

# Import the cost and gradient functions
from costFunction import *
from gradFunction import *

# Consider the "toy" dataset
X_toy = np.array([[1, 1, 0], [1, 1, 3], [1, 3, 1], [1, 3, 4]])
y_toy = np.array([[0], [1], [0], [1]])
theta_toy = np.array([[0], [0], [0]])

# Calculate the cost J for the "toy" dataset when theta = [0, 0, 0]
cost = costFunction(theta_toy, X_toy, y_toy)

# Text output: value of the cost J for the toy dataset
print(f"The value of the cost J for the toy dataset is: {cost}")

'''
Part f:
- Optimize the cost function for logistic regression with parameters theta
- Use library for optimization solver scipy.optimize.fmin_bfgs
'''

# Import optimization solver library
from scipy.optimize import fmin_bfgs

# Create an array for the initial theta values
theta_initial = np.array([[0], [0], [0]])
theta = fmin_bfgs(costFunction, theta_initial, fprime=gradFunction, args=(X_train, y_train))

# Text output: optimal parameters of theta
print(f"The optimal parameters for theta are: {theta}")

# Find the new cost
cost = costFunction(theta, X_train, y_train)

# Text output: the value of the cost function
print(f"The cost at the optimized theta values is: {cost}")

'''
Part g:
- Using the optimal theta values, plot the decision boundary
'''

# Create the x and y values for the decision boundary line on the plot
exam1_min = X_train[:,1].min() # Find the min x value
exam1_max = X_train[:,1].max() # Find the max x value
exam1_dBound = np.linspace(exam1_min, exam1_max, 200) # Create incremented values along x axis

# Line Equation: 
#   theta[0] + theta[1] * exam1_dBound + theta[2] * exam2_dBound
# Rearrange: 
#   theta[2] * exam2_dBound = -(theta[0] + theta[1] * exam1_dBound)
exam2_dBound = -(theta[0] + theta[1] * exam1_dBound) / theta[2] # Use x axis values and thetas to calculate y values

# Clear the current figure
plt.clf()

# Re-create the plot with the training data
Exam1Score = X_train[:, 1]
Exam2Score = X_train[:, 2]
m_train = y_train.size # Use the m dimensions of the training dataset
for i in range(m_train):
    if y_train[i] == 1:
        plt.scatter(Exam1Score[i], Exam2Score[i], color='green')
    else:
        plt.scatter(Exam1Score[i], Exam2Score[i], color='red')
plt.scatter([], [], color='green', label='Admitted')
plt.scatter([], [], color='red', label='Not Admitted')
plt.legend(loc='best')
plt.xlabel('Exam 1 Scores')
plt.ylabel('Exam 2 Scores')

# Overlay the plot with the new decision boundary line
plt.plot(exam1_dBound, exam2_dBound, color='blue', label='Decision Boundary') 

# Output: save the figure to the output file
plt.savefig('output/ps3-1-f.png')

'''
Part h:
- Compute the accuracy of the logistic regression model on testing data
'''

# Create an m x 1 array of the probabilities using the sigmod function
m_test = X_test.shape[0] # Find the m dimension of the testing dataset
probabilty = np.empty(m_test) # Create an array to store the probabilities
hypothesis = sigmoid(np.matmul(X_test, theta)) # hypothesis is a m x 1 array
for i in range(m_test):
    if hypothesis[i] >= 0.5:
        probabilty[i] = 1
    else:
        probabilty[i] = 0

# Calculate the accuracy using the predictions
accuracy = np.sum(probabilty == y_test) / m_test

'''
Part i:
- Compute the admission probability of student with scores test1 = 80 and test2 = 50
'''

test1 = 80
test2 = 50
testScores = np.array([1, test1, test2]) # testScores is an 1 x n array
probabilty = sigmoid(np.matmul(testScores, theta)) # probability is a scalar

# Text output: admission probabilty
print(f"The admission probabilty is: {probabilty}")

# Determine whether or not the student will be admitted
decision = "Admitted" if probabilty >= 0.5 else "Not Admitted"

# Text output: what should be the admission decision?
print(f"The admission decision should be: {decision}")

'''
Part j:
- Bonus
'''


''' Question 2: Linear Regression '''

'''
Part a:
- Read in the training data from the csv file
- Create a linear regression model to fit a non-linear function
- Use the normal equation to solve for theta parameters
'''

# Read in data from csv file into numpy array
data = np.genfromtxt('input/hw3_data2.csv', delimiter=',')

# Store population and profit as numpy column vectors
population = np.atleast_2d(data[:, 0]).T
profit = np.atleast_2d(data[:, 1]).T

# Create the feature matrix with the first column of all ones, second column is the population, and the third column is the population squared
X = np.column_stack((np.ones([population.shape[0], 1]), population, np.square(population)))

# Create the y label vector which is the profit
y = profit

# Create a function to compute the linear regression model using the normal equation solution
# Normal solution theta = (X^T * X)^-1 * X^T * y
def linear_regression(X, y):
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    return theta

# Calculate the value of theta
theta = linear_regression(X, y)

# Text output: the learned model parameters
print(f"The learned model parameters for theta are: {theta}")

'''
Part b:
- Plot datapoints and model on the same figure
'''

# Clear the plot
plt.clf()

# Add the datapoints to the plot
plt.scatter(population, profit, color='red', label='Training Data')

# Create the line
population_axis = np.linspace(X[:,1].min(), X[:,1].max(), 100)
profit_axis = theta[0] + theta[1] * population_axis + theta[2] * np.square(population_axis)
plt.plot(population_axis, profit_axis, color='blue', label='Fitted Model')

# Save the figure to the output folder
plt.xlabel('Population (in thousands)')
plt.ylabel('Profit')
plt.legend(loc='best')

# Output: save figure
plt.savefig('output/ps3-2-b.png')