# MyMidTermProject

This project contains a Flask application for predicting apple quality using an XGBoost model. The application is Dockerized for easy deployment.

## Features
- REST API with Flask.
- Dockerized for cross-platform usage.
- XGBoost for prediction. Prediction was also tried using Linear regression but the F1-Score and AUC-ROC was observed to be better on the Xgboost
-  The Data analysis on the input data is in the data folder.This contains a public domain dataset with 4000 rows and was picked up from Kaggle.
-  The data analysis and initial modelling are in the file named DataAnalysis.ipynb
-  Linear Regression details are in the LinearRegression notebook.
-  XgBoost details are in the XgBoost notebook.

## How to Use
1. Clone the repository named `MyMidTermProject`
2. Navigate to the `MyMidTermProject` directory.
3. Build the Docker Image: Run the following command in the project directory: docker build -t flask-xgboost-apple-app .
4. Run the Docker Container: Start the container with: docker run -p 5000:5000 flask-xgboost-apple-app
5. Make sure that the port 5000 is forwarded if you are using VS Code or any other similar IDE.
6. Send a POST request to the /predict endpoint using curl, Postman, or Python.
Example with curl:
curl -X POST -H "Content-Type: application/json" -d '{"features": [-3.970049,-2.512336,5.346330,-1.012009,1.844900,0.329840,-0.491590]}' http://127.0.0.1:5000/predict


## File Structure
- `app.py`: Flask application code.
- `Dockerfile`: Docker configuration for containerization.
- `xgboost_apple_quality_model.pkl`: Pretrained XGBoost model.

  ## What could have been done
- I wanted to try and implement the front end predict endpoint using  streamlit but could not implement due to my work schedule.
  
