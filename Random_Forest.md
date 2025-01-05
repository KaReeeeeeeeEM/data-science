# Random Forest

## Overview

This project demonstrates the implementation of a Random Forest algorithm using Python and the `scikit-learn` library. Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is particularly useful for handling overfitting and improving accuracy.

## Features

- Implementation of Random Forest for classification tasks.
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
   from sklearn.ensemble import RandomForestClassifier
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
4. **Create and train the Random Forest model**:

   ```python
   # Create the Random Forest classifier
   rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

   # Train the model
   rf_clf.fit(X_train, y_train)
   ```
5. **Make predictions and evaluate the model**:

   ```python
   # Make predictions on the test set
   y_pred = rf_clf.predict(X_test)

   # Calculate the accuracy of the model
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {accuracy * 100:.2f}%")
   ```
6. **Complete Code Example**:

   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import accuracy_score

   # Load the Iris dataset
   iris = load_iris()
   X = iris.data
   y = iris.target

   # Split the dataset into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Create the Random Forest classifier
   rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

   # Train the model
   rf_clf.fit(X_train, y_train)

   # Make predictions on the test set
   y_pred = rf_clf.predict(X_test)

   # Calculate the accuracy of the model
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Accuracy: {accuracy * 100:.2f}%")
   ```

## Conclusion

This project provides a basic implementation of a Random Forest classifier using the `scikit-learn` library. By following the steps outlined above, you can apply the Random Forest algorithm to various classification tasks and evaluate its performance.

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Introduction to Random Forests](https://en.wikipedia.org/wiki/Random_forest)
