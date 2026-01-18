from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import numpy as np
import os

app = FastAPI(title="Diabetes Prediction API")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load Model ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "diabetes_predictor_model.pkl")
model = joblib.load(MODEL_PATH)

# ---------- Serve Frontend ----------
@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

# ---------- Prediction ----------
@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array([[  
            data["Pregnancies"],
            data["Glucose"],
            data["BloodPressure"],
            data["SkinThickness"],
            data["Insulin"],
            data["BMI"],
            data["DiabetesPedigreeFunction"],
            data["Age"]
        ]])
        prediction = model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "result": "Diabetic" if prediction == 1 else "Not Diabetic"
        }

    except Exception as e:
        return {"error": str(e)}
