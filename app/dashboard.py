import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn IQ", layout="wide")

st.title("📊 Customer Churn Risk Analyzer")
st.markdown("Predict customer churn probability and analyze risk profiles in real-time.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

with col2:
    st.subheader("📡 Service Details")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("💳 Billing & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

st.divider()

if st.button("🚀 Analyze Risk Profile", use_container_width=True):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": tech_support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        res_data = response.json()
        
        prob = res_data['churn_probability'] * 100
        risk = res_data['risk_level']
        
        st.metric(label="Predicted Churn Risk Score", value=f"{prob:.1f}%")
        
        if risk == 'High':
            st.error("⚠️ **High Risk of Churning** — Immediate retention intervention or discount offer recommended.")
        elif risk == 'Medium':
            st.warning("⚡ **Moderate Churn Risk** — Consider proactive customer engagement.")
        else:
            st.success("✅ **Low Churn Risk** — High retention likelihood.")
            
        st.divider()
        st.subheader("🔍 Top Risk Drivers (SHAP Analysis)")
        drivers = res_data.get("top_risk_drivers", [])
        
        if drivers:
            for item in drivers:
                direction = "Increases Risk" if item['impact'] > 0 else "Decreases Risk"
                st.write(f"• **{item['feature']}**: Impact Score = `{item['impact']}` ({direction})")
        else:
            st.info("No SHAP risk drivers returned from the backend.")
            
    except Exception as e:
        st.error(f"Could not connect to FastAPI server. Make sure FastAPI is running on port 8000!\nError: {e}")