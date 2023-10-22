import numpy as np
import matplotlib.pyplot as plt
import timeit

### Section 1: Conceptual Question ###

# Propose a new regression problem that you can solve with ML and describe how you would go about solving it
    # Problem: Predict the traffic in South Oakland based on the time.
        # I would use the features (x) for the time of day.
        # I would use the labels (y) for the avg vehicle's mph under the speed limit, so that larger differences in mph will reflect higher traffic.
        # I could collect data on the speed of each car using gps (on individuals phones), or I could set up speedometers to check the vehicles speed on each street.
        # Tracking the speed of each car using gps could be an invasion of privacy and would be difficult to write the code for, and setting up lots of speedometers could get expensive and be affected by other moving objects.


### Section 2: Conceptual Question 2 ###

# Propose a new classification problem that you can solve with ML and describe how you would go about solving it
    # Problem: Predict whether I'll eat breakfast based on when I wake up.
        # I would use the features (x) for the time that I wake up.
        # I would use the labels (y) for whether or not I eat breakfast.
        # I could collect data at the time that I wake up and then whether or not I ate breakfast that morning.
        # There could be other factors that affect whether or not I eat breakfast such as rushing for an early class, or I could forget to record whether or not I ate.

### Section 3: Basic Operations ###

# a. Generate a 1,000,000 by 1 vector x of random numbers from a Gaussian (normal) distribution with a mean of 2.1 and a standard deviation of 0.7
x = np.random.default_rng().normal(2.1, 0.7, (1000000, 1))

# b. Generate a 1,000,000 by 1 vector z of random numbers from a uniform distribution between [-1.5 4]
z = np.random.default_rng().uniform(-1.5, 4, (1000000, 1))

# c. Plot the normalized histogram of vectors x and z. Store them as png images.
fig, ax = plt.subplots()
ax.hist(x, density=1)
fig.savefig("output/ps1-3-c-1.png")
# The histogram for vector x does look like a Gaussian distribution because it has a bell-like curve which is a characteristic of a Gaussian distribution.

fig2, ax2 = plt.subplots()
ax2.hist(z, density=1)
fig2.savefig("output/ps1-3-c-2.png")
# The histogram for vector z does look like a uniform distribution because it almost forms a flat line which means that there is a near equal (uniform) value across each bin.


# d. Add 1 to every value in x using a loop and calculate process time
def incrementWithLoop(x):
    dimX = np.shape(x)
    for i in range(dimX[0]):
        x[i] += 1
    
result = timeit.timeit(stmt='incrementWithLoop(x)', globals=globals(), number=1)
print(f"Array increment with loop process time: {result}")
# The processing time of incrementing every value of the vector using a loop took 3.758... seconds and appears to usually come in around 4 seconds.

# e. Add 1 to every value in x without using a loop and calculate process time
def incrementWithoutLoop(x):
    onesArray = np.ones_like(x)
    x += onesArray

result = timeit.timeit(stmt='incrementWithoutLoop(x)', globals=globals(), number=1)
print(f"Array increment without loop process time: {result}")
# The processing time of incrementing every value of the vector without using a loop took 0.00424... seconds and is much faster and therefore more time efficient than the loop method.

# f. Define vector y with values between 0 and -1 from vector z
y = np.empty_like(z)
i = 0
for val in z:
    if (val < 0 and val > -1):
        y[i] = val
        i += 1
y = np.trim_zeros(y, 'b')
dimY = np.shape(y)
print(f"Dimensions of vector y: {dimY}")
# The number of elements between 0 and -1 that I retrieved from vector z and copied to y was 182,250. Upon running this code two more times that number of retrieved elements is not exactly the same however it is similar.
# Since vector z uses a uniform distribution model and ideally the number of elements in each bin (range of values) is the same after each run, so it would make sense that the values after each run are not identical, but similar.


### Section 4: Linear Algebra ###

# a. Define matrix A without using loops.
A = np.array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])
print(f"Matrix A is:\n{A}")
# Find the min of each column in A
dim = np.shape(A)
for i in range(dim[0]):
    min = np.amin(A[:, i])
    print(f"Column {i} has min: {min}")
# Find the max of each row in A
for i in range(dim[1]):
    max = np.amax(A[i, :])
    print(f"Row {i} has max: {max}")
# Find the min value overall of A
minOverall = np.amin(A)
print(f"The overall min is: {minOverall}")
# Find the sum of each row in A
for i in range(dim[1]):
    sum = np.sum(A[i, :])
    print(f"Row {i} has sum: {sum}")
# Find the sum of all elements in A
sumOverall = np.sum(A)
print(f"The overall sum is: {sumOverall}")
# Compute matrix B which is the matrix A with an element-wise square
B = np.square(A)
print(f"Matrix B is:\n{B}")

# b. Solve the system of linear equations
A = np.array([[2, 1, 3], [2, 6, 8], [3, 5, 15]])
b = np.array([[1], [2], [5]])
sol = np.linalg.solve(A, b)
x = sol[0]
print(f"The value of x is: {x}")
y = sol[1]
print(f"The value of y is: {y}")
z = sol[2]
print(f"The value of z is: {z}")

# c. Analytically compute the L1 and L2 norms for vectors x1 and x2
x1 = np.array([0.5, 0, -1.5])
# L1 norm for vector x1
    # Step 1: Take the absolute value of the first element
        # |0.5| = 0.5
    # Step 2: Take the absolute value of the second element and add it to the first element
        # 0.5 + |0.0| = 0.5
    # Step 3: Take the absolute value of the third element and add it to the total
        # 0.5 + |-1.5| = 2
# Check L1 norm
L1 = np.linalg.norm(x1, 1)
print(f"L1 norm for vector x1: {L1}")
# L2 norm for vector x1
    # Step 1: Square the value of the first element
        # (0.5)^2 = 0.25
    # Step 2: Square the value of the second element and add it to the first element
        # 0.25 + (0.0)^2 = 0.25
    # Step 3: Square the value of the third element and add it to the total
        # 0.25 + (-1.5)^2 = 2.5
    # Step 4: Square root the total
        # (2.5)^(1/2) = 1.581...
# Check L2 norm
L2 = np.linalg.norm(x1, 2)
print(f"L2 norm for vector x1: {L2}")

x2 = np.array([1, -1, 0])
# L1 norm for vector x2
    # Step 1: Take the absolute value of the first element
        # |1| = 1
    # Step 2: Take the absolute value of the second element and add it to the first element
        # 1 + |-1| = 2
    # Step 3: Take the absolute value of the third element and add it to the total
        # 2 + |0| = 2
# Check L1 norm
L1 = np.linalg.norm(x2, 1)
print(f"L1 norm for vector x2: {L1}")
# L2 norm for vector x2
    # Step 1: Square the value of the first element
        # (1)^2 = 1
    # Step 2: Square the value of the second element and add it to the first element
        # 1 + (-1)^2 = 2
    # Step 3: Square the value of the third element and add it to the total
        # 2 + (0)^2 = 2
    # Step 4: Square root the total
        # (2)^(1/2) = 1.414...
# Check L2 norm
L2 = np.linalg.norm(x2, 2)
print(f"L2 norm for vector x2: {L2}")


### Section 5: Write a Function ###

def sum_sq_col(A):
    # Square the elements in matrix A
    squaredA = np.square(A)
    # Store the dimensions
    dim = np.shape(A)
    # Create the vector B with the same number of cols as array A rows
    B = np.zeros(dim[0])
    # Sum the squared elements for each row
    for i in range(dim[0]):
        B[i] = np.sum(squaredA[i, :])
    # convert from row vector to column vector
    B = np.atleast_2d(B)
    B = np.transpose(B)
    return B
  
# Test function with matrix 1
    # Input:
        #[[5, 2, 3],
        # [7, 4, 2],
        # [1, 0, 8]]
m1 = np.array([[5, 2, 3], [7, 4, 2], [1, 0, 8]])
print(f"Matrix m1 is:\n{m1}")
    # Output:
        #[[38],
        # [69],
        # [65]]
b1 = sum_sq_col(m1)
print(f"Column of sums of squared rows b1 is:\n{b1}")
  
# Test function with matrix 2
    # Input:
        #[[1, 8, 2, 4],
        # [3, 9, 0, 1],
        # [4, 9, 7, 0]]
m2 = np.array([[1, 8, 2, 4], [3, 9, 0, 1], [4, 9, 7, 0]])
print(f"Matrix m2 is:\n{m2}")
    # Output:
        #[[85],
        # [91],
        # [146]]
b2 = sum_sq_col(m2)
print(f"Column of sums of squared rows b2 is:\n{b2}")