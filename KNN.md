
# K-Nearest Neighbors (KNN)

## Overview

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm using Python and the `scikit-learn` library. KNN is a simple, non-parametric, and instance-based learning algorithm that can be used for both classification and regression tasks. It is based on the principle that similar instances will exist in close proximity.

## Features

- Implementation of KNN for classification tasks.
- Use of the Iris dataset for demonstration.
- Evaluation of model performance with accuracy metrics.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.6+
- `scikit-learn` library
- `numpy` library
- `pandas` library (optional, for data manipulation)

You can install the required libraries using pip:

```sh
pip install scikit-learn numpy pandas
```

## Dataset

The Iris dataset is used in this project. It consists of 150 samples from three species of Iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Four features are measured for each sample: sepal length, sepal width, petal length, and petal width.

## Implementation

### Step-by-Step Instructions

1. **Import necessary libraries**:

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score
   ```
2. **Load dataset**:

   ```python
   # Load the Iris dataset
   iris = load_iris()
   X = iris.data
   y = iris.target
   ```
3. **Split the dataset into training and testing sets**:

   ```python
   # Split the dataset into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
4. **Create and train the KNN model**:

   ```python
   # Create the KNN classifier
   knn_clf = KNeighborsClassifier(n_neighbors=3)

   # Train the model
   knn_clf.fit(X_train, y_train)
   ```
5. **Make predictions and evaluate the model**:

   ```python
   # Make predictions on the test set
   y_pred = knn_clf.predict(X_test)

   # Calculate the accuracy of the model
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {accuracy * 100:.2f}%")
   ```
6. **Complete Code Example**:

   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   # Load the Iris dataset
   iris = load_iris()
   X = iris.data
   y = iris.target

   # Split the dataset into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Create the KNN classifier
   knn_clf = KNeighborsClassifier(n_neighbors=3)

   # Train the model
   knn_clf.fit(X_train, y_train)

   # Make predictions on the test set
   y_pred = knn_clf.predict(X_test)

   # Calculate the accuracy of the model
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {accuracy * 100:.2f}%")
   ```

## Conclusion

This project provides a basic implementation of a K-Nearest Neighbors classifier using the `scikit-learn` library. By following the steps outlined above, you can apply the KNN algorithm to various classification tasks and evaluate its performance.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Introduction to K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
