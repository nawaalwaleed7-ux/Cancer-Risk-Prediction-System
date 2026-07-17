import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Preprocessor
# -----------------------------
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/logistic_regression.pkl")

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Cancer Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.success("Best Model: Logistic Regression")
st.sidebar.info("Accuracy: 80.83%")

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Cancer Risk Prediction System")

st.markdown("""
This application predicts the **Cancer Risk Level** of a patient using Machine Learning.

### Algorithms Trained
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

**Best Performing Model:** Logistic Regression
""")

st.write("### Enter Patient Information")

# -----------------------------
# User Inputs
# -----------------------------

age = st.number_input("Age", 18, 100, 40)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    10.0,
    50.0,
    25.0
)

smoking_history = st.selectbox(
    "Smoking History",
    ["Never", "Former", "Current"]
)

alcohol_use = st.selectbox(
    "Alcohol Use",
    ["Never", "Occasional", "Heavy"]
)

family_history_cancer = st.selectbox(
    "Family History of Cancer",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

physical_activity_level = st.selectbox(
    "Physical Activity Level",
    ["Low", "Moderate", "High"]
)

diet_quality = st.selectbox(
    "Diet Quality",
    ["Poor", "Average", "Good"]
)

systolic_bp = st.number_input(
    "Systolic Blood Pressure",
    80,
    220,
    120
)

cholesterol_ldl = st.number_input(
    "LDL Cholesterol",
    50,
    250,
    120
)

glucose_fasting = st.number_input(
    "Fasting Glucose",
    50,
    300,
    90
)

wbc_count = st.number_input(
    "WBC Count",
    2000,
    15000,
    7000
)

hemoglobin = st.number_input(
    "Hemoglobin",
    8.0,
    20.0,
    14.0
)

psa_level = st.number_input(
    "PSA Level",
    0.0,
    20.0,
    2.0
)

ca125_level = st.number_input(
    "CA125 Level",
    0.0,
    100.0,
    20.0
)

brca_mutation_flag = st.selectbox(
    "BRCA Mutation",
    [0, 1],
    format_func=lambda x: "Positive" if x == 1 else "Negative"
)

tp53_mutation_flag = st.selectbox(
    "TP53 Mutation",
    [0, 1],
    format_func=lambda x: "Positive" if x == 1 else "Negative"
)

egfr_expression = st.number_input(
    "EGFR Expression",
    0.0,
    10.0,
    2.0
)

telomere_length_score = st.number_input(
    "Telomere Length Score",
    0.0,
    10.0,
    5.0
)

inflammation_index = st.number_input(
    "Inflammation Index",
    0.0,
    100.0,
    20.0
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Cancer Risk"):

    patient = pd.DataFrame({

        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "smoking_history": [smoking_history],
        "alcohol_use": [alcohol_use],
        "family_history_cancer": [family_history_cancer],
        "physical_activity_level": [physical_activity_level],
        "diet_quality": [diet_quality],
        "systolic_bp": [systolic_bp],
        "cholesterol_ldl": [cholesterol_ldl],
        "glucose_fasting": [glucose_fasting],
        "wbc_count": [wbc_count],
        "hemoglobin": [hemoglobin],
        "psa_level": [psa_level],
        "ca125_level": [ca125_level],
        "brca_mutation_flag": [brca_mutation_flag],
        "tp53_mutation_flag": [tp53_mutation_flag],
        "egfr_expression": [egfr_expression],
        "telomere_length_score": [telomere_length_score],
        "inflammation_index": [inflammation_index]

    })

    patient_processed = preprocessor.transform(patient)

    prediction = model.predict(patient_processed)

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction[0] == "Low":

        st.success("🟢 Cancer Risk: LOW")

        st.write(
            "The patient appears to have a **low cancer risk** based on the provided information."
        )

    elif prediction[0] == "Moderate":

        st.warning("🟡 Cancer Risk: MODERATE")

        st.write(
            "The patient has a **moderate cancer risk**. Regular medical check-ups are recommended."
        )

    else:

        st.error("🔴 Cancer Risk: HIGH")

        st.write(
            "The patient has a **high cancer risk**. Please consult a healthcare professional for further evaluation."
        )