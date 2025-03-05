# Mental Health Prediction Project

## Overview
This project aims to predict anxiety severity based on various mental health assessment data using machine learning techniques. The project consists of three main components: a Python script for prediction, a dataset containing mental health data, and a Jupyter notebook for exploratory data analysis and model training.

# Mental Health Prediction Project

## Overview
This project aims to predict anxiety severity based on various mental health assessment data using machine learning techniques. The project consists of several components, including a Python script for prediction, a dataset containing mental health data, and a Jupyter notebook for exploratory data analysis and model training.

## Features
- Predict anxiety severity based on user input.
- Generate personalized PDF reports with suggestions.
- Calculate PHQ and GAD scores.
- Calculate Body Mass Index (BMI).
- User-friendly web interface for data input.

## Dataset
The dataset used for training the machine learning model can be found at:
[Depression and Anxiety Data](https://www.kaggle.com/datasets/shahzadahmad0402/depression-and-anxiety-data)

## Installation
To set up the project, follow these steps:

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the Gemini API key:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to create a new API key.
   - Copy the API key and create a `.env` file in the root directory.
   - Add your API key to the `.env` file:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## Usage
1. **Run the application:**
   ```bash
   python predict_mental_health.py (make sure you have saved .pkl by running mental_health.ipynb script)
   ```

2. **Access the web application:**
   Open your web browser and go to `http://127.0.0.1:5000/`.

3. **Input your data:**
   Fill out the form with your age, gender, BMI, PHQ score, anxiety severity, Epworth score, and GAD score.

4. **Submit the form:**
   Click the "Predict" button to receive your mental health prediction and suggestions.

5. **Download suggestions:**
   After receiving the suggestions, you can download them as a PDF.

## Key Components
- **predict_mental_health.py**: The main Python script responsible for handling predictions and generating reports.
- **EDA.py**: Contains exploratory data analysis insights about the dataset.
- **model.py**: Contains model training and evaluation logic.
- **utlis.py**: Handles API calls and PDF generation.
- **templates/**: Contains HTML templates for user input forms.

## Additional Features
- **PHQ Score Calculation**: A separate page to calculate the PHQ score based on user responses.
- **GAD Score Calculation**: A separate page to calculate the GAD score based on user responses.
- **BMI Calculator**: A separate page to calculate Body Mass Index (BMI) based on user input.

