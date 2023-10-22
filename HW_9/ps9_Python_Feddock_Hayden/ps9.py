# Hayden Feddock
# ECE 1395 - Dr. Dallal
# Assignment 9
# 4/18/2023

import numpy as np

''' Question 1: K-means clustering and image segmentation '''

'''
Part a
- Write a function to implement a basic version of k-means
'''

from scipy.spatial.distance import cdist

def kmeans_single(X, K, iters):
    
    '''
    Inputs:
    X: m x n data matrix where m is the number of samples and n is the features
    K: number of clusters to output
    iters: number of iterations to run K-means for
    '''

    # Determine shape of X data matrix with m samples and n feature dimensions
    m, n = X.shape
    
    # Determine the min and max of each feature dimension (axis=0 is vertical)
    mins = X.min(axis=0) # [:n]
    maxes = X.max(axis=0) # [:n]
    
    # Determine the range in values
    num_range = maxes - mins # [:n]
    
    # Randomly initialize the cluster centers between the min and max of each feature dimension
    means = mins + num_range * np.random.rand(K, n) # [:n] + [:n] * [K:n]
    
    # Update K-means for each iteration
    for iter in range(iters):
        
        # Compute the distance between each point and the cluster center
        distances = cdist(X, means) # [m:K] - returns the distance between the point X and each of the means
        
        # Update the ids to classify the sample by the closest cluster center (axis=1 is horizontal)
        ids = np.argmin(distances, axis=1) # [m:]
    
        # Recompute the means for each K
        for i in range(K):
            
            # Collect all of the points that are classified as cluster i
            cluster_i = X[ids == i] # [(m[ids==i]):n]
            
            # Find the mean point from all the points in the cluster
            means[i,:] = np.mean(cluster_i, axis=0) # [K:n]
    
    # Compute SSD
    distances = cdist(X, means)
    ssd = np.sum(np.min(distances,axis=1)**2)
    
    '''
    Outputs:
    ids: m x 1 vector containing the cluster id of each sample
    means: K x n matrix containing the centers/means of each cluster
    ssd: sum of the squared distances between points and their assigned means, summed over all clusters
    '''
    
    return ids, means, ssd

'''
# For testing K-means function:

import matplotlib.pyplot as plt

# Testing with random 2D dataset
X = np.random.rand(100, 2)*10 # [100:2]

# Compute K-means clustering using function
K = 3
ids, means, ssd = kmeans_single(X, K, 100)

# Plot data using colors to distinguish clusters
plt.scatter(X[:,0], X[:,1], c=ids)

# Plot cluster centers on top
plt.scatter(means[:,0], means[:,1], c=np.array([0, 1, 2]), marker='x')
    
# Save image
plt.savefig('question1_demo.png')
'''

'''
Part b
- Implement a wrapper function to do R random initializations and return the lowest SSD
'''

def kmeans_multiple(X, K, iters, R):
    
    # Compute the first k-means function
    ids, means, ssd = kmeans_single(X, K, iters)
    
    # Compute the next R-1 kmeans and replace if the ssd is lower
    for i in range(R-1):
        
        # Compute the kmeans again
        ids_temp, means_temp, ssd_temp = kmeans_single(X, K, iters)
        
        # Replace the current results with the new ones is the new ssd is lower
        if (ssd_temp < ssd):
            ids = ids_temp
            means = means_temp
            ssd = ssd_temp
    
    return ids, means, ssd


'''
Part c
- Read in all of the input images using the imread function
- Convert images to double format
- Downsample images using imresize
- Use reshape to convert the 3D image matrix into a 2D 
- Use the kmeans multiple function to perform clustering
- Replace each pixel with the average RGB values for the cluster
- Convert the image to the uint8 format and save
- Implement the above in a wrapper function that takes an image and then produces an image
'''

from skimage.util import img_as_float, img_as_ubyte
from skimage.transform import resize

def Segment_kmeans(im_in, K, iters, R):
    
    # Convert the image to the double format using the img_as_float function
    im = img_as_float(im_in)
    
    # Downsample images using the resize function
    im = resize(im, [100, 100])
    
    # Convert the images from 3D to 2D using the reshape function
    H, W, _ = im.shape
    X = np.reshape(im, [H*W, 3])
    
    # Compute the clustering using the kmeans multiple function
    ids, means, ssd = kmeans_multiple(X, K, iters, R)
    
    # Replace each pixel with the average RGB values for the cluster
    for i in range(H*W):
        X[i,:] = means[ids[i]]
        
    # Convert the image to uint8 format
    im_out = img_as_ubyte(X)
        
    # Reshape the image
    im_out = np.reshape(im_out, [H, W, 3])
    
    return im_out

from skimage.io import imread, imsave

# Read in the images from the input folder
im_1 = imread(f'input/im1.jpg')
im_2 = imread(f'input/im2.jpg')
im_3 = imread(f'input/im3.png')

# Values to use for K
K = [3, 5, 7]

# Values to use for iters
iters = [7, 13, 20]

# Values to use for R
R = [5, 15, 30]
    
# Compute the first segmented image
im_out_1 = Segment_kmeans(im_1, K[0], iters[0], R[0])
imsave(f'output/im1.jpg', im_out_1)

# Compute the first segmented image
im_out_2 = Segment_kmeans(im_2, K[1], iters[1], R[1])
imsave(f'output/im2.jpg', im_out_2)

# Compute the first segmented image
im_out_3 = Segment_kmeans(im_3, K[2], iters[2], R[2])
imsave(f'output/im3.jpg', im_out_3)


