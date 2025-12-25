import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Profile Score Prediction", layout="centered")

model = pickle.load(open("profile_score_model (1).pkl", "rb"))


st.title("📊 Profile Score Prediction")
st.markdown("Predict applicant profile score using Machine Learning")

with st.form("profile_form"):
    st.subheader("🧑 Personal Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married"])
    with col2:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        employment = st.selectbox("Employment Type", ["Salaried", "Self-employed"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.subheader("💰 Financial Information")
    income = st.number_input("Income", min_value=0.0)
    credit = st.number_input("Credit Score", min_value=1.0, max_value=900.0)

    colb1, colb2 = st.columns(2)
    with colb1:
        submit = st.form_submit_button("🔮 Predict")
    with colb2:
        reset = st.form_submit_button("🔄 Reset")

if reset:
    st.experimental_rerun()

if submit:
    if income <= 0:
        st.error("Income must be greater than 0")
    else:
        gender = 1 if gender == "Male" else 0
        marital = 1 if marital == "Married" else 0
        education = 1 if education == "Graduate" else 0
        employment = 1 if employment == "Salaried" else 0
        property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
        property_area = property_map[property_area]

        dependents = 0
        loan_amount = 0
        loan_term = 0
        existing_loan = 0
        savings = 0

        input_data = np.array([[
            age, gender, income, credit, marital,
            education, property_area, employment,
            dependents, loan_amount, loan_term,
            existing_loan, savings
        ]])

        result = model.predict(input_data)
        st.success(f"Predicted Profile Score: {result[0]:.2f}")

        st.subheader("📌 Feature Importance")
        feature_names = [
            "Age", "Gender", "Income", "Credit Score", "Marital Status",
            "Education", "Property Area", "Employment Type",
            "Dependents", "Loan Amount", "Loan Term",
            "Existing Loan", "Savings"
        ]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))
