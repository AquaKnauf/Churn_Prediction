from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import json
import requests
import os

app = FastAPI()
MODEL_URL = "https://huggingface.co/AquaKnauf/FastAPI_Churn_Prediction/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

download_model()
# Load model and feature names
model = joblib.load(MODEL_PATH)


with open("models/feature_names.json") as f:
    expected_columns = json.load(f)

# Simple encoders (match training encoding)
label_maps = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "MultipleLines": {"No phone service": 0, "No": 1, "Yes": 2},
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2},
    "OnlineSecurity": {"No": 0, "Yes": 1, "No internet service": 2},
    "OnlineBackup": {"No": 0, "Yes": 1, "No internet service": 2},
    "DeviceProtection": {"No": 0, "Yes": 1, "No internet service": 2},
    "TechSupport": {"No": 0, "Yes": 1, "No internet service": 2},
    "StreamingTV": {"No": 0, "Yes": 1, "No internet service": 2},
    "StreamingMovies": {"No": 0, "Yes": 1, "No internet service": 2},
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "PaymentMethod": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
}

@app.post("/predict")
def predict(data: dict):
    try:
        # Encode categorical fields
        encoded = {}
        for col in expected_columns:
            val = data.get(col)
            if col in label_maps:
                encoded[col] = label_maps[col].get(val)
                if encoded[col] is None:
                    raise HTTPException(status_code=400, detail=f"Invalid value for {col}: {val}")
            else:
                encoded[col] = val

        df = pd.DataFrame([encoded], columns=expected_columns)
        pred = model.predict(df)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
