import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Bank Churn Dashboard",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("models/churn_model.pkl")

scaler = joblib.load("models/scaler.pkl")

# ---------------- TITLE ---------------- #

st.title("🏦 AI-Powered Bank Customer Churn Prediction System")

st.markdown("""
This intelligent banking analytics system predicts customer churn probability and generates risk scores to support proactive retention strategies.
""")

st.markdown("---")

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Customer Input Features")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)

age = st.sidebar.slider("Age", 18, 92, 35)

tenure = st.sidebar.slider("Tenure", 0, 10, 5)

balance = st.sidebar.number_input("Balance", value=50000.0)

num_products = st.sidebar.slider("Number of Products", 1, 4, 2)

has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])

is_active = st.sidebar.selectbox("Is Active Member", [0, 1])

salary = st.sidebar.number_input("Estimated Salary", value=100000.0)

gender_male = st.sidebar.selectbox("Gender Male", [0, 1])

geo_germany = st.sidebar.selectbox("Germany", [0, 1])

geo_spain = st.sidebar.selectbox("Spain", [0, 1])

# ---------------- FEATURE ENGINEERING ---------------- #

balance_salary_ratio = balance / (salary + 1)

tenure_age_ratio = tenure / (age + 1)

products_per_balance = num_products / (balance + 1)

# ---------------- INPUT DATAFRAME ---------------- #

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [salary],
    "Geography_Germany": [geo_germany],
    "Geography_Spain": [geo_spain],
    "Gender_Male": [gender_male],
    "BalanceSalaryRatio": [balance_salary_ratio],
    "TenureAgeRatio": [tenure_age_ratio],
    "ProductsPerBalance": [products_per_balance]
})

# ---------------- CUSTOMER PROFILE SECTION ---------------- #

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Customer Profile")

    st.write(f"**Credit Score:** {credit_score}")
    st.write(f"**Age:** {age}")
    st.write(f"**Balance:** ${balance:,.2f}")
    st.write(f"**Products:** {num_products}")

with col2:
    st.subheader("📊 Engagement Metrics")

    st.write(f"**Active Member:** {is_active}")
    st.write(f"**Has Credit Card:** {has_card}")
    st.write(f"**Estimated Salary:** ${salary:,.2f}")
    st.write(f"**Tenure:** {tenure} years")

# ---------------- PREDICTION ---------------- #

if st.button("Predict Churn Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # ---------------- RISK CATEGORY ---------------- #

    if probability < 0.3:
        risk = "🟢 Low Risk"

    elif probability < 0.6:
        risk = "🟠 Medium Risk"

    else:
        risk = "🔴 High Risk"

    st.markdown("---")

    st.subheader("📈 Prediction Results")

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

    with metric2:
        st.metric(
            label="Risk Category",
            value=risk
        )

    # ---------------- PREDICTION MESSAGE ---------------- #

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")

    else:
        st.success("✅ Customer is likely to stay.")

    # ---------------- FEATURE IMPORTANCE ---------------- #

    st.markdown("---")

    st.subheader("🔍 Top Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        ax=ax
    )

    st.pyplot(fig)

# ---------------- INPUT SUMMARY ---------------- #

st.markdown("---")

st.subheader("🧾 Customer Input Summary")

st.write(input_data)
