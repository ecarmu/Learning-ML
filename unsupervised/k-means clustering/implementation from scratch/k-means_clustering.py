import numpy as np
import matplotlib.pyplot as plt
from utils import *
from public_tests import *


def find_closest_centroids(X, centroids):

    # MY WAY OF ACHIEVING
    '''
    idx = np.zeros(X.shape[0], dtype = int)

    for i in range(X.shape[0]):
        training_example = X[i]

        #distortions = np.square(np.abs(np.subtract(training_example - centroids)))
        distances = []

        for j in range(centroids.shape[0]):
            distance = np.sqrt((np.square(np.abs(training_example[0] - centroids[j][0])) + np.square(np.abs(training_example[1] - centroids[j][1]))))
            distances.append(distance)
        
        idx[i] = np.argmin(distances),
    '''


    # MORE OPTIMIZED WAY TO ACHIEVE IT
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
    # Find the index of the closest centroid for each point
    idx = np.argmin(distances, axis=1)

    return idx




def compute_centroids(X, idx, K):

    # Useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    
    # MY WAY OF DOING
    '''
    centroLenghts = np.zeros(K)
    
    for i in range (m):
        centroids[idx[i]] += X[i]
        centroLenghts[idx[i]] += 1
    
    for j in range (K):
        centroids[j] /= centroLenghts[j]
    '''
    
    # OPTIMIZED WAY
    centroids = np.array([X[idx == k].mean(axis=0) for k in range(K)])    
    #centroids = np.array([X[np.where(idx == k)].mean(axis=0) for k in range(K)])   ->    intuitively, I would prefer this, but the one above is more efficient

    # "idx == k" method
    '''
      [ True , False , False , True , False] --> idx == k
      [  0   ,   1   ,   2   ,  0   ,   2  ] -->  X 
      [  0            ,         0          ] -->  X [ idx == k ] --> X [ [T,F,F,T,F] ]
      If, the index has "true", the value (from that index) is taken; if the index has "false" the  value is not taken
      Returns a boolean array, which has a boolean value corresponding to every indices
    '''

    # "np.where(idx == k)" method
    '''
      [ True , False , False , True , False] --> idx == k
      [  0            ,         3          ] -->  np.where(condition)
      [  0            ,         0          ] -->  X [ np.where(condition) ] --> X [ [0,3] ] 
      np.where(condition), iterates over all indices... if the iterated index is true, it holds the integer value of that index

      np.where ( array || boolean operation || number)
    '''

    
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        
        #Output progress
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    plt.show() 
    return centroids, idx



def kMeans_init_centroids(X, K):
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids

# Load an example dataset
X = load_data()


# Set number of centroids and max number of iterations
K = 3
max_iters = 10

for i in range(10):
    # Set initial centroids by picking random examples from the dataset
    initial_centroids = kMeans_init_centroids(X, K)

    # Run K-Means
    centroids, idx = run_kMeans(X, initial_centroids, max_iters, plot_progress=True)

