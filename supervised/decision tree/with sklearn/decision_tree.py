import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])

from sklearn.model_selection import train_test_split
X = dataset.copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)





# training the model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(ccp_alpha=0.01)
# clf.train(X_train, y_train)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)



# testing the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

from sklearn.metrics import precision_score
print(precision_score(y_test, predictions))

from sklearn.metrics import recall_score
print(recall_score(y_test, predictions))