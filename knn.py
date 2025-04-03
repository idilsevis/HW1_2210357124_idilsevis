# knn.py

import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize the k-NN classifier.
        
        Parameters:
        - k: Number of neighbors to use
        - distance_metric: 'euclidean' or 'manhattan'
        """
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        """
        Store the training data.
        
        Parameters:
        - X_train: Training features (numpy array)
        - y_train: Training labels (numpy array)
        """
        self.X_train = X_train
        self.y_train = y_train

    def _compute_distance(self, x1, x2):
        """
        Compute distance between two points using the selected metric.
        
        Parameters:
        - x1, x2: Data points (numpy arrays)
        
        Returns:
        - Distance (float)
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance metric")

    def predict(self, X_test):
        """
        Predict class labels for test data.
        
        Parameters:
        - X_test: Test features (numpy array)
        
        Returns:
        - Predictions (numpy array)
        """
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Predict the class for a single test instance.
        
        Parameters:
        - x: Single data point (numpy array)
        
        Returns:
        - Predicted class label
        """
        # Compute distances to all training points
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, X_test, y_test):
        """
        Calculate the classification accuracy.
        
        Parameters:
        - X_test: Test features
        - y_test: True test labels
        
        Returns:
        - Accuracy (float)
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
