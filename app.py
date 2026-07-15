from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Load model if exists
try:
    with open('models/credit_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    model_accuracy = model_data.get('accuracy', 0.85)
    model_name = model_data.get('model_name', 'Random Forest')
    print("✅ Model loaded!")
except:
    print("⚠️ Using rule-based logic")
    model = None
    scaler = None
    model_accuracy = 0.85
    model_name = 'Rule-Based'

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form.get('name', 'Applicant')
            email = request.form.get('email', '')
            phone = request.form.get('phone', '')
            age = float(request.form.get('age', 30))
            income = float(request.form.get('income', 500000))
            employment = request.form.get('employment', 'salaried')
            credit_score = float(request.form.get('credit_score', 700))
            debt_ratio = float(request.form.get('debt_ratio', 30))
            education = request.form.get('education', 'graduate')
            marital = request.form.get('marital', 'single')
            residence = request.form.get('residence', 'rented')
            car = request.form.get('car', 'no')
            children = float(request.form.get('children', 0))
            
            # Calculate approval using rule-based logic
            approved, confidence = calculate_approval(
                age, income, credit_score, debt_ratio, 
                employment, education, marital, residence, car
            )
            
            result = {
                'name': name,
                'email': email,
                'phone': phone,
                'approved': approved,
                'confidence': confidence,
                'model': model_name,
                'accuracy': f"{model_accuracy:.1%}",
                'time': datetime.now().strftime("%d-%m-%Y %H:%M")
            }
            
        except Exception as e:
            print(f"Error: {e}")
            result = {'error': str(e)}
    
    return render_template('index.html', result=result)

def calculate_approval(age, income, credit_score, debt_ratio, employment, education, marital, residence, car):
    """Calculate approval and confidence score"""
    
    score = 0
    reasons = []
    
    # 1. Credit Score (Most Important - Max 40 points)
    if credit_score >= 750:
        score += 40
        reasons.append("Excellent credit score")
    elif credit_score >= 700:
        score += 30
        reasons.append("Good credit score")
    elif credit_score >= 650:
        score += 20
        reasons.append("Average credit score")
    elif credit_score >= 600:
        score += 10
        reasons.append("Below average credit score")
    else:
        score += 0
        reasons.append("Low credit score")
    
    # 2. Income (Max 25 points)
    if income >= 1000000:
        score += 25
        reasons.append("High income")
    elif income >= 700000:
        score += 20
        reasons.append("Good income")
    elif income >= 400000:
        score += 12
        reasons.append("Moderate income")
    elif income >= 250000:
        score += 5
        reasons.append("Low income")
    else:
        score += 0
        reasons.append("Very low income")
    
    # 3. Debt Ratio (Max 20 points)
    if debt_ratio <= 15:
        score += 20
        reasons.append("Low debt")
    elif debt_ratio <= 30:
        score += 15
        reasons.append("Manageable debt")
    elif debt_ratio <= 45:
        score += 8
        reasons.append("High debt")
    else:
        score += 0
        reasons.append("Very high debt")
    
    # 4. Employment (Max 10 points)
    if employment in ['government', 'salaried']:
        score += 10
        reasons.append("Stable employment")
    elif employment == 'business':
        score += 8
        reasons.append("Business owner")
    elif employment == 'self_employed':
        score += 5
        reasons.append("Self employed")
    else:
        score += 0
        reasons.append("Unstable employment")
    
    # 5. Education (Max 5 points)
    if education in ['post_graduate', 'professional']:
        score += 5
    elif education == 'graduate':
        score += 3
    
    # 6. Age (Max 5 points)
    if 28 <= age <= 50:
        score += 5
        reasons.append("Good age")
    elif 22 <= age <= 55:
        score += 3
    else:
        score += 0
    
    # 7. Marital Status (Max 5 points)
    if marital == 'married':
        score += 5
        reasons.append("Married")
    elif marital == 'single':
        score += 2
    
    # 8. Residence (Max 5 points)
    if residence == 'owned':
        score += 5
        reasons.append("Owns house")
    elif residence == 'with_parents':
        score += 3
    else:
        score += 1
    
    # 9. Car (Max 2 points)
    if car == 'yes':
        score += 2
        reasons.append("Owns car")
    
    # Decision
    approved = score >= 55
    
    # Calculate confidence (50-95%)
    confidence = 50 + (score / 100) * 45
    confidence = min(95, max(50, confidence))
    
    # Adjust confidence based on score
    if score >= 80:
        confidence = 90
    elif score >= 70:
        confidence = 80
    elif score >= 60:
        confidence = 70
    elif score >= 50:
        confidence = 60
    else:
        confidence = 50
    
    return approved, confidence

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)