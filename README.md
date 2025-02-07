# Mental Health Prediction Project
# Demo: https://github.com/JuliusR8ll/Argo
## Overview
This project aims to predict anxiety severity based on various mental health assessment data using machine learning techniques. The project consists of three main components: a Python script for prediction, a dataset containing mental health data, and a Jupyter notebook for exploratory data analysis and model training.

## Files

### 1. `predict_mental_health.py`
This Python script is responsible for generating predictions of anxiety severity based on user input. It utilizes a pre-trained Random Forest model to make predictions and generates a PDF report with suggestions based on the model's output.

#### Key Features:
- Loads environment variables for API keys.
- Configures the Google Gemini API for generating suggestions.
- Defines functions for generating PDFs and calling the Gemini API.
- Implements a function to predict anxiety severity based on user input.

#### Usage:
To run the script, ensure that the required libraries are installed and the environment variables are set up correctly. Execute the script in a Python environment.

### 2. `depression_anxiety_data.csv`
This CSV file contains the dataset used for training the machine learning model. It includes various features related to mental health assessments, such as age, gender, BMI, PHQ scores, and severity levels of depression and anxiety.

#### Key Columns:
- `age`: Age of the individual.
- `gender`: Gender of the individual (male/female).
- `bmi`: Body Mass Index.
- `phq_score`: PHQ score indicating depression severity.
- `gad_score`: GAD score indicating anxiety severity.
- `anxiety_severity`: Categorical variable indicating the severity of anxiety.

### 3. `metal_health_model.ipynb`
This Jupyter notebook is used for exploratory data analysis (EDA), data preprocessing, and training the machine learning model. It includes visualizations and model evaluation metrics.

#### Key Sections:
- **Data Loading**: Loads the dataset from the CSV file.
- **Data Cleaning**: Handles missing values and encodes categorical variables.
- **Feature Scaling**: Standardizes numerical features for better model performance.
- **Model Training**: Trains a Random Forest classifier on the processed data.
- **Model Evaluation**: Evaluates the model's performance using confusion matrices and classification reports.
- **SHAP Analysis**: Performs SHAP analysis to interpret the model's predictions.

#### Usage:
Open the notebook in a Jupyter environment, run the cells sequentially, and follow the instructions to train the model and make predictions.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, scikit-learn, joblib, shap, matplotlib, seaborn, fpdf, google-generativeai, dotenv

## Installation
To set up the project, clone the repository and install the required libraries using pip:
