from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Customer Churn Prediction API")

MODEL_PATH = os.path.join("models", "churn_xgb_pipeline.pkl")
model = joblib.load(MODEL_PATH)

# Define input schema matching raw CSV features
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"status": "Active", "model": "XGBoost Churn Predictor v1"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert Pydantic object to Pandas DataFrame
    input_df = pd.DataFrame([customer.model_dump()])
    
    # Get probability for class 1 (Churn = Yes)
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)
    
    # Determine risk category
    if probability > 0.6:
        risk_level = "High"
    elif probability > 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk_level
    }