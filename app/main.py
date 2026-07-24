from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import shap

app = FastAPI(title="Customer Churn Prediction API")

MODEL_PATH = os.path.join("models", "churn_xgb_pipeline.pkl")
pipeline = joblib.load(MODEL_PATH)

# Extract transformer and model steps from the saved pipeline
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['classifier']

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
    input_df = pd.DataFrame([customer.model_dump()])
    
    # 1. Prediction Probability
    probability = float(pipeline.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)
    
    # 2. Risk Categorization
    if probability > 0.6:
        risk_level = "High"
    elif probability > 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"
        
    # 3. Calculate SHAP values for feature importance
    processed_input = preprocessor.transform(input_df)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(processed_input)[0]
    
    # Get feature names post-one-hot encoding
    cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    all_feature_names = list(num_cols) + list(cat_cols)
    
    # Pair feature names with SHAP impact scores
    impact_dict = dict(zip(all_feature_names, shap_vals))
    sorted_drivers = sorted(impact_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    top_factors = [
        {"feature": feat, "impact": round(float(val), 4)} 
        for feat, val in sorted_drivers
    ]

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "top_risk_drivers": top_factors
    }