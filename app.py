
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("customer_churn_model.pkl")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "### Predict customer churn using Machine Learning"
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("👤 Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure Months", 0, 72, 12)
monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# -----------------------------
# MAIN PAGE
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("📅 Tenure", f"{tenure} Months")
col2.metric("💳 Monthly Charges", f"${monthly_charges:.2f}")
col3.metric("📄 Contract", contract)

st.divider()

st.subheader("🤖 Churn Prediction")

if model_loaded:

    if st.button("🔮 Predict Churn", use_container_width=True):

        input_data = pd.DataFrame([{
            "Gender": gender,
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": internet,
            "Online Security": online_security,
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": tech_support,
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": contract,
            "Paperless Billing": "Yes",
            "Payment Method": payment,
            "Monthly Charges": monthly_charges,
            "Total Charges": tenure * monthly_charges
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == "Yes":
            st.error("⚠️ High Risk: Customer is likely to Churn")
        else:
            st.success("✅ Low Risk: Customer is likely to stay")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        fig, ax = plt.subplots(figsize=(8, 1.5))

        ax.barh(
            ["Churn Risk"],
            [probability * 100]
        )

        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")

        st.pyplot(fig)

else:
    st.error(
        "Model file not found. Save the trained model as customer_churn_model.pkl first."
    )

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.caption(
    "Customer Churn Prediction System | Machine Learning Project"
)
