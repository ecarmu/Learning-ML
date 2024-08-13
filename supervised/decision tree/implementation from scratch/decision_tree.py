import numpy as np
import matplotlib.pyplot as plt
from public_tests import *

#%matplotlib inline
import matplotlib.pyplot as plt


X_train = np.array([[1,1,1], [1,0,1], [1,0,0], [1,0,0], [1,1,1], [0,1,1], [0,0,0], [1,0,1], [0,1,0], [1,0,0]])
y_train = np.array([   1   ,    1   ,    0   ,    0   ,    1   ,    0   ,    0   ,    1   ,    1   ,    0  ])
root_indices =     [   0   ,    1   ,    2   ,    3   ,    4   ,    5   ,    6   ,    7   ,    8   ,    9  ]

MAX_DEPTH = 2
LEFT_BRANCH = 'Left'
RIGHT_BRANCH = 'Right'

def compute_entropy(target):

    if(len(target) == 0):
        return 0;

    countOfOnes = np.sum(target == 1)
    p1 = countOfOnes / len(target)
    p2 = 1 - p1

    if(p1 == 0 or p1==1):
        entropy = 0
    else:
        entropy = -1*p1*np.log2(p1) - p2*np.log2(p2)

    return entropy


def split_dataset(X, node_indices, feature):
    node_indices_array = np.array(node_indices)

    # Extract the feature column for the given indices
    feature_values = X[node_indices_array, feature]

    # Find indices where the feature value is 1
    left_indices = node_indices_array[feature_values == 1].tolist()
    # Find indices where the feature value is 0
    right_indices = node_indices_array[feature_values == 0].tolist()

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):

    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0

    X_node_entropy  = compute_entropy(y_node)
    X_left_entropy  = compute_entropy(y_left)
    X_right_entropy = compute_entropy(y_right)

    weight_X_left  = len(X_left)  / len(X_node)
    weight_X_right = len(X_right) / len(X_node)

    information_gain = X_node_entropy - (weight_X_left * X_left_entropy + weight_X_right * X_right_entropy)

    return information_gain

def get_best_split(X, y, node_indices):   
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1

    # Checking for pureness
    if(np.all(y == y[0])):
        return best_feature

    # Compute information gains for all features using list comprehension
    information_gains = np.array([compute_information_gain(X=X, y=y, node_indices=node_indices, feature=i) for i in range(num_features)])
    
    # Find the best feature (the one with the highest information gain)
    best_feature = np.argmax(information_gains)

    return best_feature

# Not graded
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 
    if current_depth >= max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    best_feature = get_best_split(X, y, node_indices)
    
    tree.append((current_depth, branch_name, best_feature, node_indices))
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)

    build_tree_recursive(X, y, left_indices, LEFT_BRANCH, max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, RIGHT_BRANCH, max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)

# UNIT TESTS

compute_entropy_test(compute_entropy)
split_dataset_test(split_dataset)
compute_information_gain_test(compute_information_gain)
get_best_split_test(get_best_split)