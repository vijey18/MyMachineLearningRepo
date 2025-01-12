import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Define the custom CombinedGradientBoosting class
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

# Load the saved pipeline
pipeline = joblib.load('optimized_combined_model.pkl')

# Streamlit app
st.title("Grape Quality Prediction")
st.write(
    "This app predicts the grape quality category (e.g., High, Medium, Low) "
    "and the quality score based on the input features."
)

# Define sliders for input variables
sugar_content_brix = st.slider("Sugar Content (Brix)", min_value=15.0, max_value=30.0, value=22.5, step=0.1)
acidity_ph = st.slider("Acidity (pH)", min_value=2.5, max_value=4.5, value=3.5, step=0.1)
cluster_weight_g = st.slider("Cluster Weight (g)", min_value=50.0, max_value=300.0, value=120.0, step=1.0)
berry_size_mm = st.slider("Berry Size (mm)", min_value=5.0, max_value=25.0, value=15.0, step=0.1)
sun_exposure_hours = st.slider("Sun Exposure (hours)", min_value=0, max_value=12, value=8, step=1)
soil_moisture_percent = st.slider("Soil Moisture (%)", min_value=10.0, max_value=40.0, value=25.0, step=0.1)
rainfall_mm = st.slider("Rainfall (mm)", min_value=50.0, max_value=300.0, value=200.0, step=1.0)
variety = st.selectbox("Variety", ["Cabernet Sauvignon", "Merlot", "Zinfandel"])
region = st.selectbox("Region", ["Napa Valley", "Sonoma", "Central Coast"])

# Predict button
if st.button("Predict"):
    # Create a dataframe for the input
    input_data = pd.DataFrame({
        'sugar_content_brix': [sugar_content_brix],
        'acidity_ph': [acidity_ph],
        'cluster_weight_g': [cluster_weight_g],
        'berry_size_mm': [berry_size_mm],
        'sun_exposure_hours': [sun_exposure_hours],
        'soil_moisture_percent': [soil_moisture_percent],
        'rainfall_mm': [rainfall_mm],
        'variety': [variety],
        'region': [region]
    })

    # Generate predictions
    predictions = pipeline.predict(input_data)

    # Display predictions
    st.subheader("Predicted Results")
    st.write(f"**Quality Category:** {predictions['quality_category'][0]}")
    st.write(f"**Quality Score:** {predictions['quality_score'][0]:.2f}")
