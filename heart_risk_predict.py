
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(
    age: float,
    sex: float,
    cp: float,
    trestbps: float,
    chol: float,
    fbs: float,
    restecg: float,
    thalach: float,
    exang: float,
    oldpeak: float,
    slope: float,
    ca: float,
    thal: float
):
    features = np.array([[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]])

    prediction = model.predict(features)[0]

    label = "High Risk" if prediction == 1 else "Low Risk"

    return {"prediction": label}