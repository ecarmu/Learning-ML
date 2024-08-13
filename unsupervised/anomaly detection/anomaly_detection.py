import numpy as np
import matplotlib.pyplot as plt
from utils import *
from public_tests import *

# Load the dataset
X_train, X_val, y_val = load_data()

# While your servers were operating, you collected examples of how they were behaving, and thus have an unlabeled dataset { x(1) ... x(n)}.
# Because of that, we don't have y_train but we have y_val


def estimate_gaussian(X):

    m, n = X.shape
    mu = np.zeros(n)
    var = np.zeros(n)

    # LONG WAY TO UNDERSTAND HOW DOES THIS FUNCTION WORK
    '''
    """ DON'T FORGET... WE GO LIKE "FEATURE BY FEATURE"... NOT LIKE "SAMPLE BY SAMPLE """
    """ WE CALCULATE "MEAN & VARIANCE" FOR EACH FEATURE """
    
    [
        [1,2],
        [3,4],
        [5,6]
    ]

    """ We should first iterate over 1,3,5 and take mean for that specific feature """

    for i in range(n):
       
        sum = 0

        for j in range(m):
            sum += X[j][i]
        
        mu[i] = sum / m
        sum = 0

        for j in range(m):
            sum += (X[j][i] - mu[i] ) ** 2
        
        var[i] = sum / m

    '''
    
    # OPTIMIZED WAY 
    mu = np.sum(X, axis=0) / m
    var = np.sum(np.square(X - mu), axis=0) / m

    return mu, var

def select_threshold(y_val, p_val):
    # LONG WAY

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000

    # LONG WAY
    '''
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(y_val.shape[0]):
            # predict it positive (focus on epsilon not choice itself, choice = epsilon) -> P
            if p_val[i] < epsilon:
                # it is actually positive (true prediction) -> TP
                if(y_val[i]):
                    tp+=1
                # it is actually negative (false prediction) -> FP
                else:
                    fp+=1
            # predict it negative -> N
            else:
                # it is actually positive (false prediction) -> FN
                if(y_val[i] == 1):
                    fn+=1
        if (tp + fp == 0 or tp+fn==0):
            continue
        prec = tp / (tp+fp)
        rec  = tp / (tp+fn)
            
        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    '''

    
    # DOESN'T WORK BUT DON'T KNOW THE REASON
    '''
    epsilon_values = np.arange(min(p_val), max(p_val), step_size)

    for epsilon in epsilon_values:
        tp = 0 
        fp = 0
        fn = 0
        F1 = 0
        
        p_vall = p_val
        positive_idx = np.where(p_val < epsilon)
        negative_idx = np.where(p_val >= epsilon)
    
        p_vall[positive_idx] = 1
        p_vall[negative_idx] = 0

    
        tp_arr = p_vall[positive_idx] == y_val[positive_idx]
        fp_arr = p_vall[positive_idx] != y_val[positive_idx]
        fn_arr = p_vall[negative_idx] != y_val[negative_idx]

        tp = sum(tp_arr)
        fp = sum(fp_arr)
        fn = sum(fn_arr)

        if (tp + fp == 0 or tp+fn==0):
            continue
        prec = tp / (tp+fp)
        rec  = tp / (tp+fn)
            
        F1 = (2 * prec * rec) / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    '''

    # MOST OPTIMIZED WAY
    epsilon_values = np.arange(min(p_val), max(p_val), step_size)

    for epsilon in epsilon_values:
        predictions = p_val < epsilon

        # Calculate true positives, false positives, false negatives
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        
        F1 = (2 * prec * rec) / (prec + rec)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

        
    return best_epsilon, best_F1        





mu, var = estimate_gaussian(X_train)
p = multivariate_gaussian(X_train, mu, var)
p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)
    
# UNIT TEST
estimate_gaussian_test(estimate_gaussian)
select_threshold_test(select_threshold)

# Find the outliers in the training set 
outliers = p < epsilon

# Visualize the fit
visualize_fit(X_train, mu, var)

# Draw a red circle around those outliers
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize= 10,markerfacecolor='none', markeredgewidth=2)
plt.show()
