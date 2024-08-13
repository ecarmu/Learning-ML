import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *
from public_tests import *

#Load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()



def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    
    nm, nu = Y.shape  # number of movies and users --> Y (4778, 443)
    f_no = X.shape[1] # number of features
    
    J = 0

    # MY LONG METHOD 
    '''
    for j in range(nu):
        for i in range(nm):
            pred = b[b.shape[0] - 1][j]
            for f in range(f_no):
                pred += W[j][f] * X[i][f]
            J += R[i, j] * ((pred - Y[i][j]) ** 2)
    
    reg_term = 0
    for j in range(nu):
        for f in range(f_no):
            reg_term += (W[j][f])**2
    
    J += reg_term * lambda_
    
    reg_term = 0

    for i in range(nm):
        for f in range(f_no):
            reg_term += (X[i][f])**2
    
    J += reg_term * lambda_
    '''

    # MORE OPTIMIZED WAY
    '''
    for j in range(nu):
        pred = 0
        for i in range(nm):
            # for a single user w(j), compute the whole preditions (with x(1,...,nm))
            J += R[i,j] * np.square(np.dot(W[j,:],X[i,:]) + b[0,j] - Y[i,j])  # also, X[i,:].T works as well
        # then proceed to next user w(j+1)
    
    J += (lambda_) * (np.sum(np.square(W)) + np.sum(np.square(X))) 
    '''

    # MOST OPTIMIZED / VECTORIZED WAY
    J = 2 * (0.5 * np.sum(np.square(np.dot(X, W.T) + b - Y) * R) + (lambda_ / 2) * (np.sum(np.square(X)) + np.sum(np.square(W))))


    return J / 2


def cofi_grad_func(X, W, b, Y, R, lambda_):

    
    
    pred = np.dot(X, W.T) + b  # (num_items, num_users)
    error = R * (pred - Y)     # (num_items, num_users)

    # Compute gradients
    dJ_dW = np.dot(error.T, X) + lambda_ * W  # (num_users, num_features)
    dJ_dX = np.dot(error, W) + lambda_ * X    # (num_items, num_features)
    dJ_db = np.sum(error, axis=0)             # (num_users,)


    return (dJ_dW, dJ_dX, dJ_db)


def cofi_grad_desc_func(X, W, b, Y, R, alpha, lambda_, num_iters, clip_value=1.5):

    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        dJ_dW, dJ_dX, dJ_db = cofi_grad_func(X=X, W=W, b=b, Y=Y,R=R,lambda_=lambda_)

        # Gradient Clipping
        dJ_dW = np.clip(dJ_dW, -clip_value, clip_value)
        dJ_dX = np.clip(dJ_dX, -clip_value, clip_value)
        dJ_db = np.clip(dJ_db, -clip_value, clip_value)

        W -= alpha * dJ_dW
        X -= alpha * dJ_dX
        b -= alpha * dJ_db

        J_history[i] = cofi_cost_func(X=X, W=W, b=b, Y=Y, R=R, lambda_=lambda_)

        # Log periodically.
        if i % 20 == 0:
            print(f"Training loss at iteration {i}: {J_history[i]:0.1f}")
    
    return X, W, b, J_history


movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings

# Check the file small_movie_list.csv for id of each movie in our dataset
# For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700] = 5 

#Or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609] = 2;

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
my_ratings[246]  = 5   # Shrek (2001)
my_ratings[2716] = 3   # Inception
my_ratings[1150] = 5   # Incredibles, The (2004)
my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]


# Reload ratings and add new ratings
Y, R = load_ratings_small()
Y    = np.c_[my_ratings, Y]
R    = np.c_[(my_ratings != 0).astype(int), R]


# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)

num_movies, num_users = Y.shape
num_features = 100

# Create variables with normally distributed random values
W = np.random.normal(loc=0.0, scale=1.0, size=(num_users, num_features)).astype(np.float64)
X = np.random.normal(loc=0.0, scale=1.0, size=(num_movies, num_features)).astype(np.float64)
b = np.random.normal(loc=0.0, scale=1.0, size=(1, num_users)).astype(np.float64)

# Calculate the mean and standard deviation

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)

X, W, b, J_History = cofi_grad_desc_func(X=X, W=W, Y=Ynorm, b=b, R=R, alpha=0.01, lambda_= 1, num_iters=500)


# Make a prediction using trained weights and biases
p = np.matmul(X, W.T) + b

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

#filtered_predictions = my_predictions[(my_predictions >= 5.0) | (my_predictions <= 1.0)]
filtered_predictions = my_predictions[(my_predictions >= 5.0) | (my_predictions <= 0.0)]
print(len(filtered_predictions))

# Format the filtered predictions as strings
width = 10
formatted_elements = [f'{x: {width}.2f}' for x in filtered_predictions]

# Print formatted string with line breaks
print('[')
for i in range(0, len(formatted_elements), 5):  # Adjust number of elements per line
    print(' ' * 2 + ' '.join(formatted_elements[i:i+5]))
print(']')
# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')


for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')

import pandas as pd

# Create the filter
filter = (movieList_df["number of ratings"] > 20)

# Assign predictions to the DataFrame
movieList_df["pred"] = my_predictions

# Reindex columns
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])

# Filter the DataFrame
filtered_df = movieList_df[filter]

# Sort and select top 300 rows
sorted_filtered_df = filtered_df.sort_values("mean rating", ascending=False).iloc[:300]

# Print the sorted and filtered DataFrame
print(sorted_filtered_df)

