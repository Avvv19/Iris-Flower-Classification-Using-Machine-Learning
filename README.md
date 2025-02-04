# Iris-Flower-Classification-Using-Machine-Learning
This repository contains a machine learning project where different classification algorithms are applied to the Iris flower dataset to predict the species of Iris flowers based on their physical attributes. 

Project Overview:
This project is about using machine learning to classify different species of Iris flowers based on their measurements (sepal length, sepal width, petal length, and petal width). I used three popular machine learning models to compare their performance and tune them for better accuracy. The models used in this project are Random Forest, Logistic Regression, and K-Nearest Neighbors (KNN). Additionally, I have used RandomizedSearchCV to fine-tune the Random Forest model's hyperparameters and applied cross-validation to check its robustness.


Loading the Dataset: First, I loaded the Iris dataset from a URL into a DataFrame using pandas. The dataset has 150 samples with 4 features each and a target variable (the flower species).

data = pd.read_csv(url, header=None, names=columns)

Data Preprocessing: I checked the dataset’s structure and added new columns for experimentation. I added columns with fixed values (like 10) and based on conditions (e.g., if sepal_length > 5, then 1, else 0).

Model Building: I built three machine learning models:

Random Forest Classifier
Logistic Regression
K-Nearest Neighbors (KNN)
These models were trained and evaluated using metrics like accuracy, confusion matrix, and classification report.

rf_classifier = RandomForestClassifier(random_state=32)
log_reg_classifier = LogisticRegression(max_iter=210, random_state=32)
knn_classifier = KNeighborsClassifier()

Model Comparison: I trained and tested each model, then compared their performances. I also visualized the confusion matrix using a heatmap for better understanding.
Hyperparameter Tuning with RandomizedSearchCV: After building the models, I used RandomizedSearchCV to tune the hyperparameters of the Random Forest model for better accuracy. I tested a range of parameters like the number of trees, tree depth, and more.

random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=5, random_state=42)

RandomizedSearchCV gives us the best combination of parameters by randomly searching the parameter space.
Cross-Validation: To ensure that the model doesn't overfit, I applied cross-validation (5-fold) to evaluate the Random Forest model's performance across different data splits. This provides a more reliable estimate of its accuracy.

cv_scores_rf = cross_val_score(best_rf_model, X, y, cv=5)


What I Learned from This Project:
Data Preprocessing: I learned how important data preprocessing is. Cleaning data, handling missing values, and feature engineering play a huge role in model performance.
Model Selection: I tried out three different machine learning models, each having its strengths and weaknesses. It’s crucial to compare multiple models to find the best one.
Hyperparameter Tuning: Tuning the parameters of a model can make a big difference in its performance. RandomizedSearchCV helps us find the best parameters efficiently.
Cross-Validation: Cross-validation helps to avoid overfitting and gives us a better idea of how well the model will perform on unseen data.


Ways to Improve:
Feature Engineering: I can explore additional techniques like feature scaling or creating new features to improve the models further.
Ensemble Methods: I could combine different models to improve performance (e.g., using voting classifiers or stacking).
Deep Learning: For more complex datasets, I might explore neural networks, but for this dataset, traditional ML models work well.
GridSearchCV: Instead of RandomizedSearchCV, I could use GridSearchCV for hyperparameter tuning if I want to try every possible combination.


Conclusion:
In this project, I built and compared multiple machine learning models to classify Iris flowers. I learned a lot about model training, hyperparameter tuning, and cross-validation. With further experimentation and more data, these models can be improved for even better accuracy.
