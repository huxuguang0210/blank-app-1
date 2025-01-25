import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained SVM model
model = joblib.load("svm_model.joblib")

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
surgical_method = st.sidebar.selectbox("Surgical Route", options=[0, 1], format_func=lambda x: "Laparotomy" if x == 0 else "Laparoscope")
surgical_procedure = st.sidebar.selectbox("Surgical Procedure", options=[1, 2, 3], format_func=lambda x: {1: "Unilateral Cystectomy", 2: "Unilateral Salpingo-oophorectomy", 3: "Unilateral Salpingo-oophorectomy and Contralateral Cystectomy"}[x])
tumor_rupture = st.sidebar.selectbox("Tumor Rupture", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
comprehensive_staging = st.sidebar.selectbox("Comprehensive Staging", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
omentum_resection = st.sidebar.selectbox("Omentum Resection", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
lymphadenectomy = st.sidebar.selectbox("Lymphadenectomy", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
staging = st.sidebar.selectbox("Stage", options=[0, 1, 2, 3, 4], format_func=lambda x: {0: "Stage IA", 1: "Stage IB", 2: "Stage IC", 3: "Stage II", 4: "Stage III"}[x])
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
    
    # Show predicted probability as a list
    st.write("### Prediction Results:")
    st.write("- **Predicted Pregnancy Result Probability**: {:.2f}".format(prob))

    # Show additional information (example)
    if prob >= 0.5:
        st.success("The predicted outcome is more likely to be positive.")
        st.image("positive_outcome_image.jpg", caption="Positive Outcome", width=400)
    else:
        st.warning("The predicted outcome is more likely to be negative.")
        st.image("negative_outcome_image.jpg", caption="Negative Outcome", width=400)

# Batch prediction
st.subheader("📁 Batch Prediction")
st.markdown("Upload a CSV file for batch prediction. Make sure the file matches the input format.")
uploaded_file = st.file_uploader("Upload CSV File", type="csv")
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    probabilities = predict_batch(batch_data)
    
    batch_data["Predicted Probability"] = probabilities

    st.write("### Batch Prediction Results:")
    for i, row in batch_data.iterrows():
        st.write(f"**Case {i+1}:**")
        st.write(f"- **Predicted Probability**: {row['Predicted Probability']:.2f}")
        
        # Example of showing images based on prediction
        if row["Predicted Probability"] >= 0.5:
            st.image("positive_outcome_image.jpg", caption=f"Positive Outcome for Case {i+1}", width=400)
        else:
            st.image("negative_outcome_image.jpg", caption=f"Negative Outcome for Case {i+1}", width=400)

    st.write("### Prediction Results Data:")
    st.dataframe(batch_data)

    # Download button
    csv = batch_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

# Footer
st.markdown(""" 
--- 
© Shengjing Hospital of China Medical University 
""", unsafe_allow_html=True)
