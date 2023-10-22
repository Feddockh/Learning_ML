# Hayden Feddock
# ECE 1395
# 1/25/2023
# HW2

import numpy as np
from computeCost import *
from gradientDescent import *
from normalEqn import *
import pandas as pd
import matplotlib.pyplot as plt

# Question 1: Cost Function

# Create an input array x (m x n + 1)
x = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])

# Create an output array y (m x 1)
y = np.array([[3], [5], [7], [9]])

# Create an array for the theta values (1 x n + 1)
theta = np.array([0, 0.5])

# Compute the cost with the first array of theta values
result = computeCost(x, y, theta)
print(f"Output for Cost Function Test Case 1: {result}")

# Create a new array for the theta values
theta2 = np.array([1, 1])

# Compute the cost with the second array of theta values
result2 = computeCost(x, y, theta2)
print(f"Output for Cost Function Test Case 2: {result2}")


# Question 2: Gradient Descent

# Create the alpha value
alpha = 0.001

# Set the number of iterations desired
iter = 15

# Calculate the theta and cost using the gradient descent algorithm
[theta, cost] = gradientDescent(x, y, alpha, iter)
# TODO: output theta as a n+1 x 1 vector
print(f"Theta estimate using gradient descent: {theta}")
print(f"Cost after {iter} iterations: {cost[iter-1]}")

# Question 3: Normal Equation

# Calculate the ideal values for theta using the x and y matrices
theta = normalEqn(x, y)
print(f"Theta estimate using normal equation: {theta}")
print("The difference in the estimates between the result of gradient descent \
    \n and the result of the normal equation can be explained by the alpha value. \
    \n Since the alpha value is small, and the number of iterations is low, the \
    \n gradient descent method is not getting us as accurate of an estimate. If \
    \n we increase the alpha value or the number of iterations we will get a result \
    \n closer to that of normal equation")


# Question 4: Linear regression with one variable

# Read in all of the rows and then close the file
df = pd.read_csv('input/hw2_data1.csv', names=['hp', 'price'])
hp = np.array(list(df['hp']))
price = np.array(list(df['price']))

# Plot the data and save the scatter plot
plt.scatter(hp, price)
plt.xlabel("Horse Power")
plt.ylabel("Vehicle Price")
plt.savefig("output/ps2-4-b.png")

# Create the X matrix and y vector and store each example as a row
m = len(hp)
X = np.empty([m, 2])
y = np.empty([m, 1])
for i in range(m):
    X[i][0] = 1
    X[i][1] = hp[i]
    y[i] = price[i]
print(f"Size of X feature matrix: {np.shape(X)}")
print(f"Size of Y label vector: {np.shape(y)}")

# Create an array of m random values with a uniform distribution between 0 and 1
dataDivide = np.random.rand(m, 1)

# Determine the size of the test and training arrays
testRows = round(m/10)
trainRows = m - testRows

# Create the test and training arrays
X_test = np.empty([testRows, 2])
X_train = np.empty([trainRows, 2])
y_test = np.empty([testRows, 1])
y_train = np.empty([trainRows, 1])

# Fill in the testing and training arrays
# If the random value at the index i is > .9, then add the value at that index to the test array
# If the random value at the index i is <= .9, then add the value at that index to the training array
i_test = 0
i_train = 0
for i in range(m):
    if dataDivide[i] > .9 and i_test < testRows:
        X_test[i_test] = X[i]
        y_test[i_test] = y[i]
        i_test += 1
    elif i_train < trainRows:
        X_train[i_train] = X[i]
        y_train[i_train] = y[i]
        i_train += 1
    else:
        X_test[i_test] = X[i]
        y_test[i_test] = y[i]
        i_test += 1

# Compute the gradient descent and return the theta and cost vectors
alpha = 0.3
iter = 500
theta, cost = gradientDescent(X_train, y_train, alpha, iter)

# Clear the figure and all plots
plt.clf()

# Output a plot of cost vs iteration
plt.plot(np.arange(iter), cost)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.savefig("output/ps2-4-e-1.png")

# Compute the hypothesis line
h = theta[0] + theta[1] * hp

# Clear the figure and all plots
plt.clf()

# Output a figure that shows the datapoints overlayed with the hypothesis line
plt.scatter(hp, price, color='blue', label='datapoints')
plt.plot(hp, h, color='red', label='prediction')
plt.legend()
plt.xlabel("Horse Power")
plt.ylabel("Vehicle Price")
plt.savefig("output/ps2-4-e-2.png")

# Output the computed model parameters theta
print(f"Output model parameters theta: {theta}")

# Use the obtained model parameters (theta) to make predictions on profits using testing set
y_pred = theta[0] + theta[1] * X_test[:, 1]

# Calculate the mean squared error between the 
mse = np.mean((y_test - y_pred)**2)
print(f"Mean squared error from gradient descent hypothesis: {mse}")

# Use normalEqn to learn model parameters theta and make a prediction using X_test
theta = normalEqn(X_train, y_train)

# Use the obtained model parameters (theta) to make predictions on profits using testing set
y_pred = theta[0] + theta[1] * X_test[:, 1]

# Calculate the mean squared error between the 
mse = np.mean((y_test - y_pred)**2)
print(f"Mean squared error from normal equation hypothesis: {mse}")

print("These error values are very similar")

# Study the effect of learning rate
iter = 300
alpha = np.array([0.001, 0.003, 0.03, 3])

for i in range(4):
    
    # Compute the cost and using the gradient descent for the specified alpha value for iter times
    theta, cost = gradientDescent(X_train, y_train, alpha[i], iter)

    # Clear the figure and all plots
    plt.clf()

    # Output a plot of cost vs iteration
    plt.plot(np.arange(iter), cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig(f"output/ps2-4-h-{i+1}.png")

print("As the value of alpha increases the cost drops faster over and requires less iterations")


# Problem 5: Linear regression with multiple variables

# Read in data from the text file into a numpy array
data = np.genfromtxt('input/hw2_data2.txt', delimiter=',')

# Extract the columns from the 2d array
houseSize = data[:, 0]
bedrooms = data[:, 1]
housePrice = data[:, 2]

# Compute the mean and stdev for each feature dimension
houseSize_mean = np.mean(houseSize)
houseSize_stdev = np.std(houseSize)
print(f"The mean size of a house is {houseSize_mean} (square feet), and the stdev is {houseSize_stdev}")

bedrooms_mean = np.mean(bedrooms)
bedrooms_stdev = np.std(bedrooms)
print(f"The number of bedrooms in a house is {bedrooms_mean}, and the stdev is {bedrooms_stdev}")

housePrice_mean = np.mean(housePrice)
housePrice_stdev = np.std(housePrice)
print(f"The mean price of a house is {housePrice_mean}, and the stdev is {housePrice_stdev}")

# Standardize the data by subtracting the mean and dividing by the stdev for each example of each feature
houseSize = (houseSize - houseSize_mean)/houseSize_stdev
bedrooms = (bedrooms - bedrooms_mean)/bedrooms_stdev
housePrice = (housePrice - housePrice_mean)/housePrice_stdev

# Create the 2d array out of the column vectors
m, n = np.shape(data)
X = np.ones([m, 3])
y = np.empty([m, 1])
for i in range(m):
    X[i][1] = houseSize[i]
    X[i][2] = bedrooms[i]
    y[i][0] = housePrice[i]

# Return the size of the feature matrix X and label vector y
print(f"The size of feature matrix X is: {np.shape(X)}")
print(f"The size of feature matrix y is: {np.shape(y)}")

# Compute the gradient descent solution
alpha = 0.01
iter = 750
theta, cost = gradientDescent(X, y, alpha, iter)

# Clear the figure and all plots
plt.clf()

# Output a plot of cost vs iteration
plt.plot(np.arange(iter), cost)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.savefig(f"output/ps2-5-b.png")

# Text output the computed model parameters theta
print(f"The computed value for theta is: {theta}")

# Predict the price of houses with 1080 square feet and 2 bedrooms
houseSize_pred = 1080
bedrooms_pred = 2

# Normalize the values
houseSize_pred = (houseSize_pred - houseSize_mean) / houseSize_stdev
bedrooms_pred = (bedrooms_pred - bedrooms_mean) / bedrooms_stdev

# Compute the prediction
housePrice_pred = theta[0] + theta[1] * houseSize_pred + theta[2] * bedrooms_pred

# Un-normalize the prediction
housePrice_pred = housePrice_pred * housePrice_stdev + housePrice_mean

# Print out the prediction
print(f"The predicted price of a 1080 sqft house with 2 bedrooms is: {housePrice_pred}")

