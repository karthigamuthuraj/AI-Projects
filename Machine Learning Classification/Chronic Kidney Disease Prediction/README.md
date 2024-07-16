# Chronic Kidney Disease Prediction Project

## Problem Statement
The hospital management has tasked us with creating a predictive model to diagnose Chronic Kidney Disease (CKD) based on several patient parameters. This model aims to assist healthcare professionals in early detection and treatment planning.

## Dataset Information
- **Dataset Name:** CKD.csv
- **Total Rows:** [Insert Number]
- **Total Columns:** 25

## Pre-processing Methods
- **Handling Missing Values:** Imputed missing values using mean or median as appropriate.
- **Converting Categorical Data:** Converted categorical variables using one-hot encoding to numerical format.
- **Feature Selection:** Selected relevant features based on correlation analysis and domain knowledge.
- **Data Split:** Split the data into training (80%) and test (20%) sets for model evaluation.

## Model Development
Several machine learning algorithms were evaluated to predict CKD:
- **Random Forest**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Gaussian Naive Bayes**
- **Multinomial Naive Bayes**
- **Bernoulli Naive Bayes**
- **Logistic Regression**
- **Passive Aggressive Classifier**

## Model Evaluation and Results
Each algorithm was assessed using 5-fold cross-validation on the training set. The table below summarizes the evaluation metrics for each model:

```plaintext
+---------------------+--------------------------------------------+------------+-------------+---------+-----------+----------+---------------------+
|     Algorithm       |               Best Params                  | Accuracy   | Precision   | Recall  | F1-Score  | ROC AUC  |   Confusion Matrix   |
+=====================+============================================+============+=============+=========+===========+==========+=====================+
|    RandomForest     | {'classifier__max_depth': 10,               |   0.95     |    0.95     |  0.95   |   0.95    |   0.98   | [[30  2]            |
|                     |  'classifier__n_estimators': 100}           |            |             |         |           |          |  [ 2 26]]           |
+---------------------+--------------------------------------------+------------+-------------+---------+-----------+----------+---------------------+
|         SVC         | {'classifier__C': 1,                        |   0.92     |    0.92     |  0.92   |   0.92    |   0.96   | [[29  3]            |
|                     |  'classifier__kernel': 'rbf'}               |            |             |         |           |          |  [ 2 26]]           |
+---------------------+--------------------------------------------+------------+-------------+---------+-----------+----------+---------------------+
|     KNeighbors      | {'classifier__metric': 'euclidean',         |   0.88     |    0.89     |  0.88   |   0.88    |   N/A    | [[28  4]            |
|                     |  'classifier__n_neighbors': 7}              |            |             |         |           |          |  [ 3 25]]           |
+---------------------+--------------------------------------------+------------+-------------+---------+-----------+----------+---------------------+
|     GaussianNB      | {}                                         |   0.85     |    0.87     |  0.85   |   0.85    |   0.91   | [[28  4]            |
|                     |                                            |            |             |         |           |          |  [ 4 24]]           |
+---------------------+--------------------------------------------+------------+-------------+---------+-----------+----------+---------------------+
...

Best model saved: RandomForest
```

## Final Model Selection
Based on the evaluation metrics, the **Random Forest** model was selected as the final model for predicting CKD. It achieved the highest accuracy (95%) and F1-score (0.95) among all models tested. Its robust performance across multiple metrics and its ability to handle complex interactions in the data make it well-suited for this predictive task.

## 11. Model Deployment with Streamlit
- **Save the Trained Model**: Saved the best model using pickle.
- **Create a Streamlit App**: Created a Streamlit app to take user inputs and display the prediction.
- **Run the Streamlit App**: Deployed the app locally by running the Streamlit server.
  ![image](images/app.png)