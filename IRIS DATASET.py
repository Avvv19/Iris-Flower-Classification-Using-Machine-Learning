# Import necessary libraries
import pandas as pd  # Used to handle and manipulate datasets.
import numpy as np  # Used for numerical operations.
from sklearn.model_selection import train_test_split, RandomizedSearchCV, \
    cross_val_score  # For splitting data, hyperparameter tuning, and cross-validation.
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier for training.
from sklearn.linear_model import LogisticRegression  # Logistic Regression Classifier.
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors Classifier.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For model evaluation.
import seaborn as sns  # For plotting and visualization.
import matplotlib.pyplot as plt  # For plotting graphs.
from scipy.stats import randint  # For using random values during RandomizedSearchCV.

# Load the Iris dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'  # URL where the dataset is stored.
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']  # Names of the columns.
data = pd.read_csv(url, header=None, names=columns)  # Read the dataset into a pandas DataFrame.


# Print the column names of the dataset
print(f"Columns available in the dataset: {data.columns.tolist()}")

# Print the total number of columns in the dataset
print(f"Total number of columns: {len(data.columns)}")

# Adding a new column with 150 values
data['new_column'] = range(1, 151)  # Create a range of values from 1 to 150 (the number of rows)
print(data.head())  # Show the first few rows to verify

# Adding a new column with the same value for all rows
data['new_column'] = 10 # All values in this new column will be 10
print(data.head())  # Show the first few rows to verify

# Adding a new column based on a condition (e.g., if sepal_length > 5, assign 1, else assign 0)
data['new_column'] = np.where(data['sepal_length'] > 5, 1, 0)
print(data.head())  # Show the first few rows to verify

# Print the total number of columns in the dataset
print(f"Total number of columns: {len(data.columns)}")


# Now, we separate our features (X) and the target variable (y).
X = data.drop('class', axis=1)  # We remove the 'class' column to keep features (X).
y = data['class']  # 'class' column is our target variable, which we want to predict.

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# This step splits the dataset into training (80%) and testing (20%) sets. Random state ensures reproducibility.
# If you want a better split, try experimenting with different `test_size` values (e.g., 0.25, 0.3).

# Initialize classifiers for comparison: Random Forest, Logistic Regression, and KNN.
rf_classifier = RandomForestClassifier(random_state=32)  # Random Forest model.
log_reg_classifier = LogisticRegression(max_iter=210, random_state=32)  # Logistic Regression model with 210 iterations.
knn_classifier = KNeighborsClassifier()  # KNN model with default parameters.

# Fit all models and evaluate their performance.
models = [rf_classifier, log_reg_classifier, knn_classifier]  # List of classifiers to compare.
model_names = ["Random Forest", "Logistic Regression", "KNN"]  # Corresponding model names.

# Dictionary to store the results of each model for comparison.
results = {}

# Loop through each model, fit, predict, and evaluate.
for model, name in zip(models, model_names):
    model.fit(X_train, y_train)  # Train the model using the training data.

    # Predict the target variable using the test data.
    y_pred = model.predict(X_test)

    # Evaluate model performance using accuracy, confusion matrix, and classification report.
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy.
    cm = confusion_matrix(y_test, y_pred)  # Confusion matrix to see how well the model predicted each class.
    report = classification_report(y_test, y_pred)  # Classification report for precision, recall, f1-score.

    # Store results for later comparison.
    results[name] = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report
    }

    # Print the results for each model.
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")

    # Plot confusion matrix using Seaborn for better visual understanding.
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()

# Hyperparameter tuning using RandomizedSearchCV for Random Forest model
param_dist = {
    'n_estimators': randint(50, 200),  # Range for number of trees in the forest.
    'max_depth': randint(3, 15),  # Maximum depth of the trees.
    'min_samples_split': randint(2, 20),  # Minimum number of samples required to split an internal node.
    'min_samples_leaf': randint(1, 20)  # Minimum number of samples required to be at a leaf node.
}

# RandomizedSearchCV performs hyperparameter tuning with random values.
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=5,
                                   random_state=42)
# n_iter=10 means 10 random combinations will be tested.
# cv=5 means 5-fold cross-validation will be used during the search.

# Fit RandomizedSearchCV with the training data to find best hyperparameters.
random_search.fit(X_train, y_train)

# Best hyperparameters found from RandomizedSearchCV.
print(f"Best parameters from RandomizedSearchCV: {random_search.best_params_}")
best_rf_model = random_search.best_estimator_  # Get the model with the best parameters.

# Evaluate the best model using test data.
y_pred_rf_tuned = best_rf_model.predict(X_test)
accuracy_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)  # Calculate accuracy of the tuned model.
print(f"Accuracy of the tuned Random Forest model: {accuracy_rf_tuned:.2f}")

# Cross-validation to evaluate Random Forest performance across different splits of data.
cv_scores_rf = cross_val_score(best_rf_model, X, y, cv=5)  # Perform 5-fold cross-validation.
print(f"Cross-validation scores for Random Forest: {cv_scores_rf}")
print(f"Average cross-validation score: {cv_scores_rf.mean():.2f}")

