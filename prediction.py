import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Load the trained SVM model
model = joblib.load('svm_model.joblib')  # Ensure the model file is in the same directory or provide the correct path

# Title and description
st.title("SVM-Based Fertility Outcome Prediction")
st.markdown("""
This application predicts fertility outcomes based on the following input variables. Select the appropriate option from the dropdown menus below:
""")

# Input form for single prediction
st.header("Single Input Prediction")
with st.form("single_input_form"):
    surgical_method = st.selectbox("Surgical Method", ["0=0", "1=1"], format_func=lambda x: x.split('=')[1])
    surgical_procedure = st.selectbox(
        "Surgical Procedure",
        ["1=Tumor Resection", "2=Unilateral Adnexectomy", "3=Unilateral + Contralateral Tumor Resection"],
        format_func=lambda x: x.split('=')[1]
    )
    tumor_rupture = st.selectbox("Tumor Rupture", ["0=No", "1=Yes"], format_func=lambda x: x.split('=')[1])
    comprehensive_staging = st.selectbox(
        "Comprehensive Staging (Ascites + Omentum + Peritoneal Biopsy)",
        ["0=No", "1=Yes"],
        format_func=lambda x: x.split('=')[1]
    )
    omentum_resection = st.selectbox("Omentum Resection", ["0=No", "1=Yes"], format_func=lambda x: x.split('=')[1])
    lymphadenectomy = st.selectbox("Lymphadenectomy", ["0=No", "1=Yes"], format_func=lambda x: x.split('=')[1])
    staging = st.selectbox(
        "Staging",
        ["0=Stage IA", "1=Stage IB", "2=Stage IC", "3=Stage II", "4=Stage III"],
        format_func=lambda x: x.split('=')[1]
    )
    unilateral_bilateral = st.selectbox("Unilateral/Bilateral", ["0=Unilateral", "1=Bilateral"], format_func=lambda x: x.split('=')[1])
    tumor_diameter = st.selectbox("Tumor Diameter", ["0=Diameter < 7 cm", "1=Diameter ≥ 7 cm"], format_func=lambda x: x.split('=')[1])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Extract numerical values from the selected options
    input_data = np.array([[
        int(surgical_method.split('=')[0]),
        int(surgical_procedure.split('=')[0]),
        int(tumor_rupture.split('=')[0]),
        int(comprehensive_staging.split('=')[0]),
        int(omentum_resection.split('=')[0]),
        int(lymphadenectomy.split('=')[0]),
        int(staging.split('=')[0]),
        int(unilateral_bilateral.split('=')[0]),
        int(tumor_diameter.split('=')[0])
    ]])

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # Display results
    st.subheader("Prediction Result")
    result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"**Fertility Outcome:** {result}")
    st.write(f"**Probability:** {probability[0][prediction[0]]:.2f}")

    # Display feature contribution (using permutation importance if available)
    st.subheader("Variable Contribution")
    try:
        importance = permutation_importance(model, input_data, [prediction[0]], scoring='accuracy')
        importance_df = pd.DataFrame({
            'Variable': [
                "Surgical Method", "Surgical Procedure", "Tumor Rupture",
                "Comprehensive Staging", "Omentum Resection", "Lymphadenectomy",
                "Staging", "Unilateral/Bilateral", "Tumor Diameter"
            ],
            'Importance': importance.importances_mean
        }).sort_values(by='Importance', ascending=False)

        st.write(importance_df)

        # Plot feature importance
        fig, ax = plt.subplots()
        ax.barh(importance_df['Variable'], importance_df['Importance'])
        ax.set_xlabel("Importance")
        ax.set_title("Variable Contribution to Prediction")
        st.pyplot(fig)
    except Exception as e:
        st.write("Unable to calculate variable contribution:", e)

# File upload for batch prediction
st.header("Batch Input Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Ensure required columns exist
    expected_columns = [
        "Surgical Method", "Surgical Procedure", "Tumor Rupture",
        "Comprehensive Staging", "Omentum Resection", "Lymphadenectomy",
        "Staging", "Unilateral/Bilateral", "Tumor Diameter"
    ]

    if all(col in data.columns for col in expected_columns):
        # Predict for batch data
        predictions = model.predict(data[expected_columns])
        probabilities = model.predict_proba(data[expected_columns])

        # Add results to the DataFrame
        data['Fertility Outcome'] = ["Yes" if p == 1 else "No" for p in predictions]
        data['Probability'] = [prob[p] for prob, p in zip(probabilities, predictions)]

        st.subheader("Prediction Results")
        st.write(data)

        # Plot lollipop chart
        st.subheader("Lollipop Chart of Predictions")
        fig, ax = plt.subplots()
        outcomes = data['Fertility Outcome'].value_counts()
        ax.stem(outcomes.index, outcomes.values, basefmt=" ", use_line_collection=True)
        ax.set_ylabel("Count")
        ax.set_title("Fertility Outcomes")
        st.pyplot(fig)
    else:
        st.error(f"The uploaded file must contain the following columns: {', '.join(expected_columns)}")

# Footer with copyright
st.markdown("""
---
**© 2025 Your Organization Name**
""")
