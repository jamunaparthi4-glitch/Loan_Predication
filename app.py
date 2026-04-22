from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = int(request.form["Gender"])
        married = int(request.form["Married"])
        income = float(request.form["ApplicantIncome"])
        loan = float(request.form["LoanAmount"])
        credit = int(request.form["Credit_History"])

        features = np.array([[gender, married, income, loan, credit]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)

        result = "Loan Approved " if prediction[0] == 1 else "Loan Rejected "

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)