'''
Hayden Feddock
ECE 1395 - Dr. Dallal
HW 5
2/19/2023
'''

import numpy as np
import scipy.io

''' Question 1: Weighted NN '''

'''
Part a
- Write a function that uses all of the neighbors to make a prediction, but weighs them according to their distance to the test sample
'''

from weightedKNN import *

'''
Part b
- Read in data
- Call function with sigma values
'''

# Load in the data from the .mat file
data = scipy.io.loadmat('input/hw4_data3.mat')

# Seperate the data
X_train = np.array(data['X_train'])
y_train = np.array(data['y_train'])
X_test = np.array(data['X_test'])
y_test = np.array(data['y_test'])

# Define values for sigma
sigmas = [0.01, 0.05, 0.2, 1.5, 3.2]

# Call function with the sigma values to make the predictions
y_predict = []
for sigma in sigmas:
    y_predict.append(weightedKNN(X_train, y_train, X_test, sigma))


# Count the number of predictions that match for each sigma value
matches = [0] * len(sigmas)
for i in range(len(sigmas)):
    for row in range(len(y_predict[0])):
        matches[i] += 1 if y_predict[i][row] == y_test[row] else 0

# Calculate the accuracy for each sigma value
accuracy = [match / len(y_predict[0]) for match in matches]

# List the accuracy vs sigma as a table
print(f"| Sigma | Accuracy |")
print(f"|{'-'*7}|{'-'*10}|")
for i in range(len(sigmas)):
    print(f"| {sigmas[i]:<5} | {accuracy[i]:<8} |")

print("The sigma value shows that at there is a range for sigma at which the testing accuracy will be greatest and above or below this range will result in a lower accuracy")

''' Question 2 '''

'''
Part 2.0: Preprocessing
- Create directories to hold the training and testing images
- Randomly select 8 images from each subject folder to move to the training folder
- The other 2 images from each subject folder should be saved under the testing folder
'''

import os
import random
import shutil

# Path to the training and testing image directories
training_folder_path = os.path.join("input", "train")
testing_folder_path = os.path.join("input", "test")

# Check if the directories already exist, in which case delete them to start fresh
if os.path.exists(training_folder_path):
    shutil.rmtree(training_folder_path)
if os.path.exists(testing_folder_path):
    shutil.rmtree(testing_folder_path)

# Create the training and testing foldes (even if they exist so that we remake them)
os.makedirs(os.path.join("input", "train"))
os.makedirs(os.path.join("input", "test"))

# Keep track of the training and testing image numbers
training_image_number = 1
testing_image_number = 1

# Iterate through each subject folder
for i in range(1, 41):
    
    # Create a random range with 8 values from 1 to 10 to designate the training images
    training_images = random.sample(range(1,11), 8)
    
    # Iterate through each image in the subject folder and copy them to either the training or testing folder
    for j in range(1, 11):
        
        # If the j value is within the random set of training image indexes, then copy it to the training folder
        if j in training_images:
            
            # Path to the training image (pgm)
            pgm_path = os.path.join("input", "s" + str(i), str(j) + ".pgm")
            
            # Copy the image to the new filepath
            shutil.copy(pgm_path, training_folder_path)
            
            # Create the new filename for the pgm image and increment the training image number
            new_pgm_path = os.path.join(training_folder_path, "PersonID_" + str(training_image_number) + ".pgm")
            training_image_number += 1
            
            # Rename the pgm file in the new directory
            os.rename(os.path.join(training_folder_path, str(j) + ".pgm"), new_pgm_path)
            
        # Otherwise copy the image to the testing folder
        else:
            
            # Path to the testing image (pgm)
            pgm_path = os.path.join("input", "s" + str(i), str(j) + ".pgm")
            
            # Copy the image to the new filepath
            shutil.copy(pgm_path, testing_folder_path)
            
            # Create the new filename for the pgm image and increment the testing image number
            new_pgm_path = os.path.join(testing_folder_path, "PersonID_" + str(testing_image_number) + ".pgm")
            testing_image_number += 1
            
            # Rename the pgm file in the new directory
            os.rename(os.path.join(testing_folder_path, str(j) + ".pgm"), new_pgm_path)
        
        
# Import the class to convert a pgm to a png
from PIL import Image, ImageDraw

# Randomly select a pgm image file from the training folder
id = "PersonID_" + str(random.randint(1, 230))
pgm_file = id + ".pgm"
pgm_file_path = os.path.join("input", "train", pgm_file)
png_file = "ps5-2-0.png"
png_file_path = os.path.join("output", png_file)

# Open the image file, add text to identify the ID, and save it as a png file to the output folder
with Image.open(pgm_file_path) as im:
    draw = ImageDraw.Draw(im)
    draw.text((0, 0), id)
    im.save(png_file_path)
    
'''
Part 2.1: PCA Analysis
- Compute the mean image and solve for the eigenfaces
'''

''' 
Part a
- Construct matrix T which is a 10304 x 320 matrix with each training pgm image stored as a column vector
'''

# Read the pgm file and produce a column vector as output
def pgm2col (pgm_file_path):
    
    # Open the PGM file using PIL
    img = Image.open(pgm_file_path)
    
    # Convert the image to an array
    A = np.array(img)
    
    # Increase the dimension to convert from array to column vector
    return A.flatten('F').reshape(-1, 1)
    

# Create the empty matrix T
T = None

# Stack on the new column vector to the array for each image in the training dataset
for i in range(1, 320):
    
    # Determine the next file to add to the matrix
    pgm_file_path = os.path.join("input", "train", "PersonID_" + str(i) + ".pgm")
    
    # Add the file to the matrix as a column vector
    if T is None:
        T = pgm2col(pgm_file_path)
    else:
        new_col = pgm2col(pgm_file_path)
        T = np.hstack((T, new_col))
        
# Create an image from the T matrix
img = Image.fromarray(T, mode='L')

# Save the image file as a png in the output folder
img.save(os.path.join("output", "ps5-1-a.png"))

'''
Part b
- Compute the average face vector m (Take the average across each row of T to get the average pixel)
- Resize m to be 112 x 92 and display the resultant image
'''

# Compute the mean across each row
m = np.mean(T, axis=1).reshape(-1, 1)

# Resize column vector to be 112 x 92
# Need to convert means to ints and reshape using Fortran method
avg_face = m.astype(np.uint8).reshape((112,92), order='F')

# Create an image from the m matrix 
img = Image.fromarray(avg_face)

# Save the image file as a png in the output folder
img.save(os.path.join("output", "ps5-2-1-b.png"))

# Describe the result
print("The averaged values that have been re-assmebled into a face resemble a blurry and fadded generic facial structure")

'''
Part c
- Find the centered data matrix A = T - m
- Find the data covariance matrix C = AA^T
'''

# The centered data matrix A is the T matrix subtracted by m
A = T - m

# The data covariance matrix C is matrix A * A^T
C = np.dot(A, A.T)
        
# Create an image from the covariance C matrix (have to store as type uint8)
img = Image.fromarray(C.astype(np.uint8), mode='L')

# Save the image file as a png in the output folder
img.save(os.path.join("output", "ps5-1-c.png"))

'''
Part d
- Use the eig function to compute the eigenvalues of A.T * A
- Determine the minimum number of eigenvalues needed to capture 95% of the variance
'''

# Compute the eigenvalues of the covariance matrix (for some reason you cannot use C, you must use A.T * A)
evals = np.linalg.eig(np.dot(A.T, A))[0]

# Sort the eigenvalues in descending order
sorted_evals = np.sort(evals)[::-1]

# Compute the total variance by adding up all of the eigenvalues
total_variance = np.sum(sorted_evals)

# Compute the percent variance at with using the cumulative sum and divide by the total variance
v = np.cumsum(sorted_evals) / total_variance

# Find the first value (min eigenvalues) in the v array of variance for each additional k that captures 95% variance
k = np.argmax(v >= 0.95) + 1
print(f"The minimum number of eigenvalues needed to capture 95% variance is: {k}")

# Create a plot of v versus k
import matplotlib.pyplot as plt
plt.plot(range(1, len(v) + 1), v)
plt.xlabel('Number of Eigenvalues (k)')
plt.ylabel('% Variance Captured (v(k))')
plt.title('v(k) vs k')
plt.savefig(os.path.join("output", "ps5-2-1-d.png"))

'''
Part e
- Retrieve the k dominant eigenvectors corresponding to the heights K eigenvalues from covariance matrix C
- Resize and save the first 9 eigenvectors as subimages of an image
'''

# Compute the k eigenvectors from the covariance matrix and store to basis matrix U
import scipy.sparse.linalg as sp
U = np.real(sp.eigs(C, k)[1])
print(f"The dimensions of matrix U are: {np.shape(U)}")

# Create the figure and axes for the subplots
fig, axes = plt.subplots(nrows=3,ncols=3,figsize=(8,8))

# Iterate through adding each of the 9 images
for i in range(9):
    
    # Extract part of the i-th eigenvector
    ev = U[:, i]
    
    # Reshape it to display as an image in the subplot
    # Error: Do not try to cast to uint8 (makes black images)
    ev_img = ev.reshape((112,92), order='F')
    
    # Plot the eigenvector in the subplot
    axes[i//3,i%3].imshow(ev_img, cmap='gray')
    axes[i//3,i%3].axis('off')
    
# Save the figure
plt.tight_layout()
plt.savefig(os.path.join("output", "ps5-2-1-e.png"))
print("The eigenfaces appear as very blurry non-distinctive images of peoples faces")

'''
Part 2.2: Feature Extraction for Face Recognition
'''

'''
Part a:
- Compute matrix W_training and use it to hold the calculated w value from each training image
'''

# Create an array to hold the w values coresponding to the training images
W_training = np.zeros((320, k))

# Create y label vector which will keep track of which subject corresponds to which row in W_training
y_train = np.zeros(320)

# Start at subject 1
subject = 0

# Iterate over each image in the training folder
for i in range(320):

    # Create the file path to the ith training image
    file_path = os.path.join("input", "train", "PersonID_" + str(i + 1) + ".pgm")
    
    # Load in the ith image
    img = Image.open(file_path)
    
    # Convert the image into a column vector and subtract the mean vector
    A = np.array(img).flatten() - m.flatten()
    
    # Project the image onto the reduced eigenface space
    w = np.dot(U.T, A)
    
    # Add the reduced representation to the feature matrix
    W_training[i, :] = w.real
    
    # Increment the subject at multiples of 8
    if i % 8 == 0:
        subject += 1
    
    # Get the corresponding subject label and add it to the labels vector
    y_train[i] = subject

'''
Part b
- Project all of the images in the testing folder to the new reduced testing space
'''
    
# Construct a matrix for all of the w values for the testing images
W_testing = np.zeros((80, k))

# Create a label vector y test
y_test = np.zeros(80)

# Start at subject 1
subject = 0
    
# Iterate over each image in the testing folder
for i in range(80):

    # Create the file path to the ith training image
    file_path = os.path.join("input", "test", "PersonID_" + str(i + 1) + ".pgm")
    
    # Load in the ith image
    img = Image.open(file_path)
    
    # Convert the image into a column vector and subtract the mean vector
    A = np.array(img).flatten() - m.flatten()
    
    # Project the image onto the reduced eigenface space
    w = np.dot(U.T, A)
    
    # Add the reduced representation to the feature matrix
    W_testing[i, :] = w.real
    
    # Increment the subject at multiples of 2
    if i % 2 == 0:
        subject += 1
    
    # Get the corresponding subject label and add it to the labels vector
    y_test[i] = subject
    
# Output the dimensions of W_training and W_testing
print(f"Dimensions of W_training: {np.shape(W_training)}")
print(f"Dimensions of W_testing: {np.shape(W_testing)}")
    
'''
Part 2.3: Face Recognition
'''

'''
Part a
- Train a KNN classifier using the W_training matrix
- Use K = 1, 3, 5, 7, 9, and 11
- Compute the accuracy of the classifier
'''

from sklearn.neighbors import KNeighborsClassifier

# Train KNN Classifier for each k value with W_training and y_train
k_vals = [1, 3, 5, 7, 9, 11]

# Keep track of the accuracy from each of the classifiers
accuracies = []

# Create a classifier for each value of k
for k in k_vals:
    
    # Create the classifier using the k value
    KNN = KNeighborsClassifier(n_neighbors=k)
    
    # Train the classifier
    KNN.fit(W_training, y_train)
    
    # Create the prediction vector using the testing data
    y_predict = KNN.predict(W_testing)
    
    # Compute the accuracy
    accuracy = np.sum(y_predict == y_test) / len(y_test)
    
    # Append the accuracy of the classifier to the list of accuracies
    accuracies.append(accuracy)
    
# List the k vs accuracy as a table
print(f"| K  | Accuracy |")
print(f"|{'-'*4}|{'-'*10}|")
for i in range(len(k_vals)):
    print(f"| {k_vals[i]:<2} | {accuracies[i]:<8} |")
    
print("As the number of K values used in the classifier increases, the accuracy decreases")

'''
Part b
- Train 6 SVM classifiers using 6 different combinations of multi-class classification paradigms
'''

from sklearn.svm import SVC
import time

# Define the SVM classifiers
SVM_Classifiers = [
    
    # One-vs-one SVM with a linear kernel
    SVC(kernel='linear', decision_function_shape='ovo'),
    
    # One-vs-one SVM with a polynomial kernel
    SVC(kernel='poly', degree=3, decision_function_shape='ovo'),
    
    # One-vs-one SVM with a Gaussian kernel
    SVC(kernel='rbf', decision_function_shape='ovo'),
    
    # One-vs-all SVM with a linear kernel
    SVC(kernel='linear', decision_function_shape='ovr'),
    
    # One-vs-all SVM with a polynomial kernel
    SVC(kernel='poly', degree=3, decision_function_shape='ovr'),
    
    # One-vs-all SVM with a Gaussian kernel
    SVC(kernel='rbf', decision_function_shape='ovr')
]

# Create lists to hold the training accuracies and times for the SVM classifiers
training_accuracies = []
training_times = []

# Train the SVM classifiers
for SVM in SVM_Classifiers:
    
    # Record the time before training
    start_time = time.time()
    
    # Train the model
    SVM.fit(W_training, y_train)
    
    # Record the time after training
    end_time = time.time()
    
    # Append the training time for the SVM
    training_times.append(end_time - start_time)
    
    # Calculate the predicted labels
    y_predict = SVM.predict(W_training)
    
    # Compute the accuracy of the SVM
    accuracy = np.mean(y_predict == y_train)
    
    # Append the accuracy of the SVM
    training_accuracies.append(accuracy)
    
# Print out a table of the training accuracy for each classifier
print(f"|           Training Accuracies            |")
print(f"|------------------------------------------|")
print(f"|            |  One-vs-One  |  One-vs-All  |")
print(f"|{'-'*12}|{'-'*14}|{'-'*14}|")
print(f"| Linear     |{training_accuracies[0]:<14}|{training_accuracies[3]:<14}|")
print(f"| Polynomial |{training_accuracies[1]:<14}|{training_accuracies[4]:<14}|")
print(f"| Gaussian   |{training_accuracies[2]:<14}|{training_accuracies[5]:<14}|")
print('\n')

# Print out a table of the training accuracy for each classifier
print(f"|              Training Times              |")
print(f"|------------------------------------------|")
print(f"|            |  One-vs-One  |  One-vs-All  |")
print(f"|{'-'*12}|{'-'*14}|{'-'*14}|")
print(f"| Linear     |{training_times[0]:.12f}|{training_times[3]:.12f}|")
print(f"| Polynomial |{training_times[1]:.12f}|{training_times[4]:.12f}|")
print(f"| Gaussian   |{training_times[2]:.12f}|{training_times[5]:.12f}|")
print('\n')

# Compute the testing accuracies for each SVM
testing_accuracies = []
for SVM in SVM_Classifiers:
    
    # Calculate the predicted labels
    y_predict = SVM.predict(W_testing)
    
    # Compute the accuracy of the SVM
    accuracy = np.mean(y_predict == y_test)
    
    # Append the accuracy of the SVM
    testing_accuracies.append(accuracy)

# Print out a table of the training accuracy for each classifier
print(f"|           Testing Accuracies             |")
print(f"|------------------------------------------|")
print(f"|            |  One-vs-One  |  One-vs-All  |")
print(f"|{'-'*12}|{'-'*14}|{'-'*14}|")
print(f"| Linear     |{testing_accuracies[0]:<14}|{testing_accuracies[3]:<14}|")
print(f"| Polynomial |{testing_accuracies[1]:<14}|{testing_accuracies[4]:<14}|")
print(f"| Gaussian   |{testing_accuracies[2]:<14}|{testing_accuracies[5]:<14}|")
print('\n')

print("On average the linear kernals take the shortest amount of time to train and the Gaussian kernels take the longest. The accuracy between using one vs one and one vs all is identical and the training time is also nearly identical. The testing accuracy was below the training accuracy on average, but interestingly the accuracy using the Gaussian kernal showed no difference.")    

print("The best fit for this dataset is definitely the linear algorithm since it takes the shortest amount of time to train and has the highest accuracy. The one vs one and one vs all appear to have nearly identical accuracies but the one vs one seems to take a little longer to train than the one vs all (at least for the linear kernel).")