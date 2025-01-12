# Grape Quality and Category prediction

This project predicts grape quality score and grape quality category  based on various features such as sugar content, berry size, sun exposure, rainfall and acidity among other factors. Accurate grape quality/category prediction can assist stakeholders in making informed decisions and avoid paying high prices for low Quality grapes.

## Problem Description

Grape quality_score/quality_category fluctuate due to multiple factors, making it challenging for buyers to ascertain the best deals. This project addresses the problem by developing a machine learning model that predicts grape quality/category, enabling buyers to plan and anticipate costs. 

## Dataset

The dataset used in this project is derived from [Kaggle's Grape Quality Dataset](https://www.kaggle.com/datasets/mrmars1010/grape-quality). It contains information about Grape Quality such as:

- Identification: Sample ID and variety
- Geographic Origin: Region
- Quality Metrics: Quality score, category, sugar content, acidity
- Physical Characteristics: Cluster weight, berry size
- Environmental Factors: Harvest date, sun exposure, soil moisture, rainfall

## Exploratory Data Analysis (EDA)

An extensive EDA has been performed to understand the dataset:

- **Data Overview**: Analyzed the range of values for each feature, checked for missing values, and assessed data types.
- **Target Variable Analysis**: Examined the distribution of Grape quality/category to identify patterns and outliers.
- **correlation matrix**: Investigated correlations between features and the target variable to determine their impact on Grape quality/category.
- **Feature Importance**: Utilized statistical methods to evaluate the significance of each feature in predicting Grape quality/category.

## Model Training

Multiple models were trained and evaluated:

1. **Linear Regression**: Served as a baseline model.
2. **Decision Tree Regressor**: Captured non-linear relationships.
3. **Random Forest Regressor**: Improved performance through ensemble learning.
4. **Custom Combined Gradient Boosting Regressor**: Enhanced accuracy by focusing on errors of previous models.

Since there is a close correlation between grape quality_score and grape quality_category, I have taken the Decision to put together a custom gradient boosting 
algorithm that uses Regression for predicting the grape quality_score and classifier for predicting the grape quality_category.

Hyperparameter tuning was performed on this custom combined Gradient Boosting model that had the best performance among them to optimize performance.

## Exporting Notebook to Script

The logic for training the model has been exported to a separate Python script (`train_pipeline.py`) to facilitate reproducibility and deployment.

## Reproducibility

To reproduce the results:

1. **Data Access**: The dataset is available in the repository under the name GRAPE_QUALITY.csv.
2. **Execution**: Run the Jupyter notebooks for EDA(Not necessary)  (`notebook.ipynb`) and  the training script (`train_pipeline.py`) without errors.

## Environment Setup

To set up the project, follow these steps.

1. Clone the repository at your desired path, then open the folder:
   ```bash
      git clone https://github.com/yasamangs/Flight-Price-Prediction.git
      cd Flight-Price-Prediction
   ```
2. Create a Conda environment and activate it:
   ```bash
      conda create --name grape-prediction-env python=3.9
      conda activate grape-prediction-env
   ```
3. Install the required dependencies:
   ```bash
      pip install -r requirements.txt
   ```

## Running the Model Training Script
To train the model using the provided script, execute:
   ```bash
      python script/train_pipeline.py
   ```

## Containerization

A Dockerfile is provided to containerize the application.

1. Build the Docker Image
   - With Docker installed on your system, build the image using:
   ```bash
   docker build -t streamlit-grape-quality-prediction .
   ```
2. Start the Prediction Service Container: 
   ```bash
   docker run -p 8501:8501 streamlit-grape-quality-prediction
   ```


## Testing the Prediction Service

The streamlit app has sliders each with a min and maximum value. Do adjust the sliders as per your needs and click the predict button.
It will output a 'grape_quality' index which is coming from the Regression model of the Custom Gradient boost and 'grape_category' index
which is coming from the Classification model of the Custom Gradient boost.
   ``

# License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

# Acknowledgments
- **Kaggle** for providing the dataset.  
- **Scikit-learn** for machine learning tools.  
- **Streamlit** for the web framework.    

