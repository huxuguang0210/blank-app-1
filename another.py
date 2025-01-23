import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample

# Load trained SVM model
model = joblib.load("svm_model.joblib")

# Define function to calculate 95% Confidence Interval
def calculate_confidence_interval(predictions, confidence=0.95):
    lower = np.percentile(predictions, (1 - confidence) / 2 * 100)
    upper = np.percentile(predictions, (1 + confidence) / 2 * 100)
    return lower, upper

# Function to make predictions
def predict_single(data):
    probability = model.predict_proba([data])[0, 1]  # Assuming binary classification
    return probability

def predict_batch(df):
    probabilities = model.predict_proba(df)[:, 1]
    return probabilities

# Streamlit UI
st.set_page_config(page_title="Pregnancy Outcome Prediction", page_icon="🤰", layout="wide")
st.title("🤰 Pregnancy Outcome Prediction Tool")
st.markdown(
    """
    Welcome to the **Pregnancy Outcome Prediction Tool**. 
    This tool predicts the probability of pregnancy outcome based on clinical input variables. 
    Please fill in the required inputs or upload a CSV file for batch predictions.
    """
)

# Sidebar input form
st.sidebar.header("Input Variables")
st.sidebar.markdown("Fill in the following details:")
surgical_method = st.sidebar.selectbox("Surgical Method", options=[0, 1], format_func=lambda x: "Open Surgery" if x == 0 else "Laparoscopic Surgery")
surgical_procedure = st.sidebar.selectbox("Surgical Procedure", options=[1, 2, 3], format_func=lambda x: {1: "Tumor Resection", 2: "Unilateral Adnexectomy", 3: "Unilateral + Contralateral Tumor Resection"}[x])
tumor_rupture = st.sidebar.selectbox("Tumor Rupture", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
comprehensive_staging = st.sidebar.selectbox("Comprehensive Staging", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
omentum_resection = st.sidebar.selectbox("Omentum Resection", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
lymphadenectomy = st.sidebar.selectbox("Lymphadenectomy", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
staging = st.sidebar.selectbox("Staging", options=[0, 1, 2, 3, 4], format_func=lambda x: {0: "Stage IA", 1: "Stage IB", 2: "Stage IC", 3: "Stage II", 4: "Stage III"}[x])
unilateral_bilateral = st.sidebar.selectbox("Unilateral/Bilateral", options=[0, 1], format_func=lambda x: "Unilateral" if x == 0 else "Bilateral")
tumor_diameter = st.sidebar.selectbox("Tumor Diameter", options=[0, 1], format_func=lambda x: "<7 cm" if x == 0 else "≥7 cm")

# Single prediction
st.subheader("🔍 Single Case Prediction")
if st.button("Predict Single Case"):
    input_data = [
        surgical_method, surgical_procedure, tumor_rupture, comprehensive_staging,
        omentum_resection, lymphadenectomy, staging, unilateral_bilateral, tumor_diameter
    ]
    prob = predict_single(input_data)
    st.success(f"### Predicted Pregnancy Result Probability: {prob:.2f}")

    # Bootstrap for Confidence Interval
    bootstrap_preds = [predict_single(resample(input_data)) for _ in range(1000)]
    lower, upper = calculate_confidence_interval(bootstrap_preds)
    st.info(f"### 95% Confidence Interval: [{lower:.2f}, {upper:.2f}]")

# Batch prediction
st.subheader("📁 Batch Prediction")
st.markdown("Upload a CSV file for batch prediction. Make sure the file matches the input format.")
uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    probabilities = predict_batch(batch_data)

    # Add confidence intervals to the batch
    bootstrap_preds_batch = [predict_batch(resample(batch_data)) for _ in range(1000)]
    lower_ci, upper_ci = zip(*[calculate_confidence_interval(bootstrap_preds) for bootstrap_preds in bootstrap_preds_batch])
    
    batch_data["Predicted Probability"] = probabilities
    batch_data["95% CI Lower"] = lower_ci
    batch_data["95% CI Upper"] = upper_ci

    st.write("### Prediction Results")
    st.dataframe(batch_data)

    # Download button
    csv = batch_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions with Confidence Intervals",
        data=csv,
        file_name="predictions_with_confidence_intervals.csv",
        mime="text/csv"
    )

# Footer
st.markdown("""
---
© Shengjing Hospital of China Medical University
""", unsafe_allow_html=True)
