#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import joblib

# Load data
data = pd.read_csv("GRAPE_QUALITY.csv")
new_data = data.drop(columns=['sample_id', 'harvest_date'])

# Split features and target variables
X = new_data.drop(columns=['quality_category', 'quality_score'])
y_category = new_data['quality_category']
y_score = new_data['quality_score']

# Split data into train/validation/test sets (60/20/20)
X_train, X_temp, y_category_train, y_category_temp, y_score_train, y_score_temp = train_test_split(
    X, y_category, y_score, test_size=0.4, random_state=25
)
X_val, X_test, y_category_val, y_category_test, y_score_val, y_score_test = train_test_split(
    X_temp, y_category_temp, y_score_temp, test_size=0.5, random_state=25
)

# Identify categorical and numerical columns
cat_columns = [col for col in X_train.columns if X_train[col].dtype == 'object']
num_columns = [col for col in X_train.columns if X_train[col].dtype in ['int', 'float']]

# Preprocessing pipelines
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, cat_columns),
    ('num', numerical_transformer, num_columns)
])

# Define parameter grids
classification_params = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7]
}

regression_params = {
    'model__n_estimators': [50, 100, 150],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 5, 10]
}

# Classifier pipeline and hyperparameter tuning
classifier_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=25))
])

grid_search_classifier = GridSearchCV(
    estimator=classifier_pipeline,
    param_grid=classification_params,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_classifier.fit(X_train, y_category_train)
best_classifier = grid_search_classifier.best_estimator_
print("Best Classifier Parameters:", grid_search_classifier.best_params_)

# Regressor pipeline and hyperparameter tuning
regression_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(random_state=25))
])

grid_search_regressor = GridSearchCV(
    estimator=regression_pipeline,
    param_grid=regression_params,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search_regressor.fit(X_train, y_score_train)
best_regressor = grid_search_regressor.best_estimator_
print("Best Regressor Parameters:", grid_search_regressor.best_params_)

# Combined model
class CombinedGradientBoosting:
    def __init__(self, classification_model, regression_model):
        self.classification = classification_model
        self.regression = regression_model

    def fit(self, X, y):
        self.classification.fit(X, y['quality_category'])
        self.regression.fit(X, y['quality_score'])
        return self

    def predict(self, X):
        classification_preds = self.classification.predict(X)
        regression_preds = self.regression.predict(X)
        return pd.DataFrame({
            'quality_category': classification_preds,
            'quality_score': regression_preds
        }, index=X.index)

# Multi-output model combining both
optimized_model = CombinedGradientBoosting(
    classification_model=best_classifier,
    regression_model=best_regressor
)

# Fit the combined model
optimized_model.fit(X_train, pd.concat([y_category_train, y_score_train], axis=1))

# Save the pipeline
joblib.dump(optimized_model, 'optimized_combined_model.pkl')
print("Optimized model saved successfully!")

