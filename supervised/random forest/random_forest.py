import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

'''
# pandas array for data - visualization
data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
print(dataset)
'''

# numpy arrays for data - computation
data = load_breast_cancer()
dataset = data['data']  # This is already a NumPy array
feature_names = data['feature_names']  # This is a NumPy array of feature names


from sklearn.model_selection import train_test_split
X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


def random_sampling(X, y):
    # Generate random indices
    random_indices = np.random.choice(len(X_train), size=X_train.shape[0], replace=True)
    # Select rows based on random indices
    X_sampling = X[random_indices]
    y_sampling = y[random_indices] 

    return X_sampling, y_sampling


# training random forest
def build_random_forest(X_train, y_train, n_estimators=100):
    random_forest = [DecisionTreeClassifier().fit(*random_sampling(X_train, y_train)) for _ in range(n_estimators)]
    return random_forest

def predict(random_forest, X_test):
    
    # Predict with all trees
    all_predictions = np.array([tree.predict(X_test) for tree in random_forest]).T
    
    # Compute the majority vote
    #predictions = np.array(np.bincount(all_predictions[i]).argmax() for i in range(all_predictions.shape[0]))
    
    predictions = np.array([np.bincount(all_predictions[i]).argmax() for i in range(all_predictions.shape[0])])
    
    return predictions

forest = build_random_forest(X_train=X_train, y_train=y_train)
predictions_forest = predict(random_forest=forest, X_test=X_test)

# testing the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions_forest))


clf = DecisionTreeClassifier(ccp_alpha=0.01)
clf.fit(X_train, y_train)
predictions_tree = clf.predict(X_test)
print(accuracy_score(y_test, predictions_tree))
'''

'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)
print(accuracy_score(y_test, predictions_rf))
