from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Plot the sample data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Sample Data for Clustering')
plt.show()





# Important part
from sklearn.cluster import KMeans
kMeans = KMeans(n_clusters=4)
kMeans.fit(X=X_train, y=y_train)
predictions = kMeans.predict(X=X_test)
# Important part



# Plotting
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Predictions')

centers = kMeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')

plt.legend()
plt.show()




# How to find the best amount of clusters
costs = []

for i in range (1,11):
    kMeans = KMeans(n_clusters=i)
    kMeans.fit(X=X_train, y=y_train)
    costs.append(kMeans.inertia_)
# How to find the best amount of clusters



# Plotting data
plt.plot(range(1,11), costs, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Costs')
plt.show()

kMeans = KMeans(n_clusters=3)
kMeans.fit(X=X_train, y=y_train)
predictions = kMeans.predict(X=X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Predictions')

centers = kMeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centers')

plt.legend()
plt.show()