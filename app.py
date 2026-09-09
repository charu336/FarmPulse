from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("farm_risk_model.pkl")


@app.get("/")
def home():
    return {"message": "FarmPulse Risk Assessment API is running"}


@app.post("/predict")
def predict_risk(data: dict):

    # Convert incoming data into a DataFrame
    farm_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(farm_data)

    return {
        "risk_level": prediction[0]
    }