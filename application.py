# Import the Required Libraries 
import flask
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
import numpy as np
import joblib

# Initialize the Instance 
application = Flask(__name__)

# Load the Model 
model = load_model("models/model.h5")

# Load the Scaler
scaler = joblib.load("models/scaler.pkl")

# Route the Home Page 
@application.route('/')
def home():
    return render_template('home.html')

# Navigate from Home Page to Result Page 
@application.route('/predict', methods=['POST'])
def result():
    # Extract and convert form inputs
    income = float(request.form.get('Income'))
    credit_score = float(request.form.get('Credit_Score'))
    loan_amount = float(request.form.get('Loan_Amount'))
    years_employed = float(request.form.get('Years_Employed'))
    points = float(request.form.get('Points'))

    # Prepare input and scale
    input_data = np.array([[income, credit_score, loan_amount, years_employed, points]])
    input_scaled = scaler.transform(input_data)

    # Predict the Model
    prediction = model.predict(input_scaled)
    probability = (1 - float(prediction[0][0])) * 100

    # Rule Based Logic
    if (
        credit_score >= 750
        and income >= 50000
        and years_employed >= 2
        and points >= 60
        and loan_amount <= income * 0.4
    ):
        status = "Loan Approved"

    elif (
        credit_score >= 650
        and income >= 35000
        and years_employed >= 1
        and loan_amount <= income * 0.6
    ):
        status = "Loan Possibly Approved"

    elif (
        credit_score >= 550
        and income >= 25000
        and loan_amount <= income * 0.8
    ):
        status = "Loan Possibly Not Approved"

    else:
        status = "Loan Not Approved"

    # Combine both logic + probability
    if probability >= 75 and "Approved" in status:
        final_result = f"{status} (High confidence)"
    elif 50 <= probability < 75:
        final_result = f"{status} (Moderate confidence)"
    else:
        final_result = f"{status} (Low confidence)"

    # Send the computed prediction and input details to the results page for display
    return render_template(
        'results.html',
        prediction=final_result,
        probability=f"{probability:.2f}%",
        income=income,
        credit_score=credit_score,
        loan_amount=loan_amount,
        years_employed=years_employed,
        points=points
    )

# Code Runner 
if __name__ == '__main__':
    application.run(debug=True)