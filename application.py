from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application
# Import the model and scaler
ridge_model = joblib.load('models/ridge_model.joblib')
scaler = joblib.load('models/scaler.joblib')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH  = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        data_scaled = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        results = ridge_model.predict(data_scaled)
        return render_template('home.html', results=round(results[0], 2))
    else:
        return render_template('home.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0")