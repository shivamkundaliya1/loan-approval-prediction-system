"""
app.py
------
Streamlit web app for the Loan Approval Prediction System.
Provides a modern, form-based UI where a user can enter applicant
details and get an instant loan approval prediction with confidence.

Run this from the PROJECT ROOT folder (one level above src/):
    streamlit run src/app.py

Note: The model must already be trained before running this app.
Run "python train.py" inside the src/ folder first.
"""

import os

import streamlit as st

from preprocessing import (
    load_and_explore_data,
    remove_duplicates,
    prepare_features_and_target,
    handle_missing_values,
    encode_categorical,
)
from predict import load_model_and_scaler, predict_new_customer

# ---------------------------------------------------------------------------
# Paths
# app.py lives inside src/, so we go one level up to reach the project root
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "loan_approval.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "loan_approval_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "loan_approval_scaler.pkl")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(180deg, #f5f7fa 0%, #eef2f7 100%); }

        .main-header {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 60%, #818cf8 100%);
            padding: 2.2rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.8rem;
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
        }
        .main-header h1 { color: white; font-size: 2rem; margin: 0; font-weight: 700; }
        .main-header p { color: rgba(255,255,255,0.9); margin-top: 0.4rem; font-size: 1rem; }

        div[data-testid="stForm"] {
            background: white;
            padding: 1.8rem 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            border: 1px solid #eef0f4;
        }

        .section-label {
            font-weight: 600;
            font-size: 0.95rem;
            color: #4f46e5;
            margin-bottom: 0.4rem;
            margin-top: 0.6rem;
        }

        .stButton > button, button[kind="formSubmit"] {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            width: 100%;
            transition: transform 0.15s ease;
        }
        .stButton > button:hover, button[kind="formSubmit"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
        }

        .result-approved {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 1px solid #6ee7b7;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin-top: 1.5rem;
        }
        .result-rejected {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 1px solid #fca5a5;
            border-radius: 16px;
            padding: 1.5rem 2rem;
            text-align: center;
            margin-top: 1.5rem;
        }
        .result-approved h2 { color: #059669; margin: 0.3rem 0 0 0; }
        .result-rejected h2 { color: #dc2626; margin: 0.3rem 0 0 0; }
        .result-icon { font-size: 2.8rem; }

        .confidence-label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.6rem;
        }

        .app-footer {
            text-align: center;
            margin-top: 2.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid #e5e7eb;
            font-size: 0.8rem;
            color: #9ca3af;
        }
        .app-footer a { color: #4f46e5; text-decoration: none; }

        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "This app predicts whether a loan application will be "
        "**Approved** or **Rejected** using a tuned Logistic "
        "Regression model trained on historical loan data."
    )
    st.markdown("---")
    st.markdown("### ⚙️ How it works")
    st.write(
        "1. Fill in the applicant's details\n"
        "2. Click **Predict**\n"
        "3. Get an instant loan decision with confidence score"
    )
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[GitHub Repository](https://github.com/shivamkundaliya1)")
    st.markdown("[LinkedIn](https://linkedin.com/in/shivamkundaliya)")
    st.markdown("---")
    st.caption("Built by Shivam Kundaliya with Python, scikit-learn & Streamlit")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🏦 Loan Approval Prediction</h1>
        <p>Fill in the applicant details below to get an instant prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_everything():
    """Load the trained model, scaler, and rebuild the exact training feature columns."""
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

    df = load_and_explore_data(DATA_PATH)
    df = remove_duplicates(df)
    X, y = prepare_features_and_target(df)
    X = handle_missing_values(X)
    X = encode_categorical(X)

    return model, scaler, X.columns


# ---------------------------------------------------------------------------
# Check model exists before proceeding
# ---------------------------------------------------------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.error(
        "⚠️ Model not found! Please train the model first.\n\n"
        "Open a terminal, go into the `src` folder, and run:\n\n"
        "`python train.py`"
    )
    st.stop()

model, scaler, feature_columns = load_everything()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("loan_form"):
    st.markdown('<p class="section-label">👤 Personal Details</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        married = st.selectbox("Married", ["Yes", "No"])
    with col3:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

    col4, col5 = st.columns(2)
    with col4:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with col5:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    st.markdown('<p class="section-label">💰 Financial Details</p>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=500)
    with col7:
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=500)

    col8, col9 = st.columns(2)
    with col8:
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, step=10)
    with col9:
        loan_term = st.number_input("Loan Term (in days)", min_value=0, value=360, step=30)

    st.markdown('<p class="section-label">🏠 Other Details</p>', unsafe_allow_html=True)
    col10, col11 = st.columns(2)
    with col10:
        credit_history = st.selectbox(
            "Credit History", [1.0, 0.0],
            format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)"
        )
    with col11:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Loan Status")

# ---------------------------------------------------------------------------
# Run prediction on submit
# ---------------------------------------------------------------------------
if submitted:
    new_customer_data = {
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_history": credit_history,
        "property_area": property_area,
    }

    with st.spinner("Analyzing application..."):
        prediction = predict_new_customer(new_customer_data, model, scaler, feature_columns)

        # Get prediction confidence if the model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            import pandas as pd

            df_input = pd.DataFrame([new_customer_data])
            df_input = pd.get_dummies(
                df_input,
                columns=["gender", "married", "dependents", "education", "self_employed", "property_area"],
                drop_first=True, dtype=int
            )
            df_input = df_input.reindex(columns=feature_columns, fill_value=0)
            df_scaled = scaler.transform(df_input)
            proba = model.predict_proba(df_scaled)[0]
            class_index = list(model.classes_).index(prediction)
            confidence = proba[class_index] * 100

    if prediction == "Approved":
        st.markdown(
            f"""
            <div class="result-approved">
                <div class="result-icon">✅</div>
                <h2>Loan Approved</h2>
                <p>Based on the details provided, this application is likely to be approved.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()
    else:
        st.markdown(
            f"""
            <div class="result-rejected">
                <div class="result-icon">❌</div>
                <h2>Loan Rejected</h2>
                <p>Based on the details provided, this application is likely to be rejected.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if confidence is not None:
        st.markdown(f'<p class="confidence-label">Model confidence: {confidence:.1f}%</p>', unsafe_allow_html=True)
        st.progress(int(confidence))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Built with Python &amp; Streamlit by Shivam Kundaliya ·
        <a href="https://github.com/shivamkundaliya1" target="_blank">GitHub</a> ·
        <a href="https://linkedin.com/in/shivamkundaliya" target="_blank">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)