import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Step 1: Install the Dataset Library
pip install ucimlrepo

# Step 2: Load the Heart Disease Dataset
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Step 3: Data Preprocessing

# Handle missing values (impute with the mean for simplicity)
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42, stratify=y)

# Reshape y_train and y_test using .ravel()
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Create a pipeline for scaling and the Naïve Bayes model
pipeline = Pipeline([
    ('scalar', StandardScaler()),
    ('model', GaussianNB())
])

# Step 4: Train the Naïve Bayes Model
pipeline.fit(X_train, y_train)

# Step 5: Make Predictions on the Test Set
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test) # Probability of the positive class

# Step 6: Evaluate the Model

# Calculate the AUC score

# Calculate the AUC score using 'ovo' (One-vs-One) strategy
auc_ovo = roc_auc_score(y_test, y_prob, multi_class='ovo')
print(f'test AUC (one-vs-one): {auc_ovo:.4f}')

# Calculate the AUC score using 'ovr' (One-vs-Rest) strategy
auc_ovr  = roc_auc_score(y_test, y_prob, multi_class='ovr')
print(f'test AUC (one-vs-rest): {auc_ovr:.4f}')

# Print the classification report
print('\nClassification Report:')
print(classification_report(y_test, y_pred))
