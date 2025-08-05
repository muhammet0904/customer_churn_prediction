
# Customer Churn Prediction Project
# This script performs customer churn prediction using Logistic Regression and Random Forest Classifier.
# It includes data loading, merging, preprocessing, feature engineering, model training, and evaluation.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load telecom demographics and usage datasets
telecom_demographics = pd.read_csv("telecom_demographics.csv")
telecom_usage = pd.read_csv("telecom_usage.csv")

# Explore missing values and unique categories in both datasets
for col in telecom_demographics.columns:
    if telecom_demographics[col].dtype == "object":
        print(f"{col} unique values: ", telecom_demographics[col].unique())
        print(f"{col} missing value counts: ", telecom_demographics[col].isna().sum(), "\n")
    else:
        print(f"{col} missing value counts: ", telecom_demographics[col].isna().sum(), "\n")

for col in telecom_usage.columns:
    if telecom_usage[col].dtype == "object":
        print(f"{col} unique values: ", telecom_usage[col].unique())
        print(f"{col} missing value counts: ", telecom_usage[col].isna().sum(), "\n")
    else:
        print(f"{col} missing value counts: ", telecom_usage[col].isna().sum(), "\n")

# Merge datasets on customer_id
churn_df = telecom_demographics.merge(telecom_usage, on="customer_id")
print(churn_df.head())

# Convert categorical columns to category dtype
categorical_columns = ["telecom_partner", "gender", "state", "city"]
for col in categorical_columns:
    churn_df[col] = churn_df[col].astype("category")

# Print churn rate
churn_rate = churn_df["churn"].value_counts() / len(churn_df["churn"])
print("churn rate:\n", churn_rate)

# Convert categorical variables to dummy variables
churn_df = pd.get_dummies(churn_df, columns=["telecom_partner", "gender", "state", "city", "registration_event"])

# Feature scaling
scaler = StandardScaler()
features = churn_df.drop(["customer_id", "churn"], axis=1)
scaled_features = scaler.fit_transform(features)
target = churn_df["churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# Logistic Regression model
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

# Evaluate Logistic Regression model
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logreg_pred))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, logreg_pred))

# Random Forest Classifier model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)

# Evaluate Random Forest model
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predict))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predict))
