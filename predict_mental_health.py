import numpy as np
import joblib
import os
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_pdf(suggestions):
    """Generate a PDF with the suggestions from the Gemini API."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Suggestions ", ln=True, align='C')

    # Replace unsupported characters
    suggestions = suggestions.replace('\u2014', '-')  # Replace em dash with a regular dash
    suggestions = suggestions.replace('\u2013', '-')  # Replace en dash with a regular dash
    suggestions = suggestions.replace('\u2018', "'")  # Replace left single quote with a regular quote
    suggestions = suggestions.replace('\u2019', "'")  # Replace right single quote with a regular quote
    suggestions = suggestions.replace('\u201c', '"')  # Replace left double quote with a regular quote
    suggestions = suggestions.replace('\u201d', '"')  # Replace right double quote with a regular quote
    # Add more replacements as needed for other unsupported characters

    # Directly use the suggestions string
    pdf.multi_cell(0, 10, txt=suggestions)  # No need to join if suggestions is already a string

    # Save the PDF to a file
    pdf_file_name = "suggestions.pdf"
    pdf.output(pdf_file_name)
    print(f"PDF generated: {pdf_file_name}")

def call_gemini_api(data):
    prompt = f"""
Given the following mental health assessment data, generate a detailed yet empathetic natural language explanation of the predicted severity level. 

### Task:
1. Provide a **clear interpretation** of the results based on the given data.
2. Suggest **coping mechanisms** tailored to the individual's condition.
3. Recommend **potential next steps**, including professional consultation if necessary.

### Provided Information:
- **Age:** {data['age']}
- **Gender:** {data['gender']}
- **BMI:** {data['bmi']}
- **PHQ Score:** {data['phq_score']}
- **Depression Severity:** {data['anxiety_severity']}
- **Epworth Score (Daytime Sleepiness Level):** {data['epworth_score']}
- **GAD Score (Generalized Anxiety Disorder Level):** {data['gad_score']}
- **Predicted Severity:** {data['predicted_severity']}

### Expected Response:
- A **concise summary** of the mental health findings.
- **Personalized coping strategies** for managing symptoms.
- **Actionable next steps** for improving well-being, including professional support if needed.
"""

    response = model.generate_content(prompt)
    
    suggestions = response.text  # This should be a single string
    
    generate_pdf(suggestions)  # Pass the string directly
    return suggestions

def predict_anxiety_severity():
    depression_model = joblib.load('depression_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    # Get user input
    try:
        age = float(input("Enter age: "))
        gender = input("Enter gender (male/female): ")
        bmi = float(input("Enter BMI: "))
        phq_score = float(input("Enter PHQ Score: "))
        anxiety_severity = input("Enter Anxiety Severity: ")
        epworth_score = float(input("Enter epworth Score: "))
        gad_score = float(input("Enter gad Score: "))
    except ValueError:
        print("Error: Please enter valid numerical values for age, BMI, and PHQ Score.")
        return
    
    
    
    input_data = np.array([[age, gender, bmi, phq_score, anxiety_severity,epworth_score,gad_score]], dtype=float)
    
    prediction = depression_model.predict(input_data)
    
    predicted_severity = label_encoder.inverse_transform(prediction)
    
    print(f"Predicted Anxiety Severity: {predicted_severity[0]}")

    api_input = {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "phq_score": phq_score,
        "anxiety_severity": anxiety_severity,
        "epworth_score": epworth_score,
        "gad_score": gad_score,
        "predicted_severity": predicted_severity
    }

    # Call Gemini API
    response = call_gemini_api(api_input)
    print("Suggestions:", response)


predict_anxiety_severity()
