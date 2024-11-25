#!/usr/bin/env python
# coding: utf-8

# In[1]:


#use xgboost to see if its better than linear regression model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score, make_scorer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Load the data
df = pd.read_csv("./apple_quality.csv")

# Preprocessing
df.columns = df.columns.str.lower()
df = df.dropna()
df.drop('a_id', axis=1, inplace=True)
df['acidity'] = df['acidity'].astype('float')

quality_mapping = {"good": 1, "bad": 0}
df['quality'] = df['quality'].map(quality_mapping)

# Define features and target variable
X = df.drop(columns=['quality'])
y = df['quality']

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

# Set up XGBoost with a pipeline
xgboost_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale features
    ('xgb', XGBClassifier(
        random_state=1,
        use_label_encoder=False,
        eval_metric='logloss'  # Avoids a warning
    ))
])

# Perform cross-validation on the training set
scorer = make_scorer(f1_score)  # Use F1-score as the evaluation metric
cv_scores = cross_val_score(xgboost_pipeline, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1)

print(f"Cross-Validation F1-Scores: {cv_scores}")
print(f"Mean Cross-Validation F1-Score: {np.mean(cv_scores):.3f}")

# Train the model and evaluate on validation and test sets
xgboost_pipeline.fit(X_train, y_train)

# Validation Evaluation
y_val_pred = xgboost_pipeline.predict(X_val)
y_val_pred_proba = xgboost_pipeline.predict_proba(X_val)[:, 1]
print("\nValidation Results:")
print("F1-Score:", f1_score(y_val, y_val_pred))
print("AUC-ROC:", roc_auc_score(y_val, y_val_pred_proba))
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# Test Evaluation
y_test_pred = xgboost_pipeline.predict(X_test)
y_test_pred_proba = xgboost_pipeline.predict_proba(X_test)[:, 1]
print("\nTest Results:")
print("F1-Score:", f1_score(y_test, y_test_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_test_pred_proba))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'xgb__n_estimators': [50, 100, 200],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__max_depth': [3, 5, 7],
    'xgb__subsample': [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(xgboost_pipeline, param_grid, cv=5, scoring=scorer, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)

# Evaluate the best model from GridSearchCV
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nTest Results for Best Model:")
print("F1-Score:", f1_score(y_test, y_test_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_test_pred_proba))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# In[2]:


#pickle this xgboost model with the best parameters
import pickle

# Save the pipeline (preprocessing + trained model) using pickle
with open('xgboost_apple_quality_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("The best tuned pipeline has been saved as 'xgboost_apple_quality_model.pkl'")

# In[ ]:



