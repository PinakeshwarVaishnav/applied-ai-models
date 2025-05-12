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

# Train the model
pipeline.fit(X_train, y_train)
