"""
predict.py
----------
Loads the saved model and scaler, then predicts loan approval
status for a new customer's data.
"""

import joblib
import pandas as pd


def load_model_and_scaler(model_path="../loan_approval_model.pkl", scaler_path="../loan_approval_scaler.pkl"):
    """Load the trained model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_new_customer(customer_data, model, scaler, feature_columns):
    """
    Predict loan approval status for a new customer.

    Args:
        customer_data: dict with raw customer details
                        e.g. {"gender": "Male", "married": "Yes", ...}
        model: trained classifier (loaded from .pkl)
        scaler: fitted StandardScaler (loaded from .pkl)
        feature_columns: the exact encoded column order used during training

    Returns:
        Predicted loan status ("Approved" or "Rejected")
    """
    new_customer = pd.DataFrame([customer_data])

    # One-hot encode the same way as training data
    new_customer = pd.get_dummies(
        new_customer,
        columns=[
            "gender",
            "married",
            "dependents",
            "education",
            "self_employed",
            "property_area"
        ],
        drop_first=True,
        dtype=int
    )

    # Align columns with training data (fill missing dummy columns with 0)
    new_customer = new_customer.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale using the same scaler used during training
    new_customer_scaled = scaler.transform(new_customer)

    # Predict
    prediction = model.predict(new_customer_scaled)

    print("\nLoan Prediction:-")
    print(prediction[0])

    return prediction[0]


if __name__ == "__main__":
    # Quick manual test — run this file directly to test a sample prediction
    # NOTE: Run train.py first, so the model + scaler .pkl files exist.

    from preprocessing import (
        load_and_explore_data,
        remove_duplicates,
        prepare_features_and_target,
        handle_missing_values,
        encode_categorical,
    )

    model, scaler = load_model_and_scaler()

    # Rebuild the exact same feature columns that were used during training
    df = load_and_explore_data()
    df = remove_duplicates(df)
    X, y = prepare_features_and_target(df)
    X = handle_missing_values(X)
    X = encode_categorical(X)
    feature_columns = X.columns

    # Sample new customer
    new_customer_data = {
        "gender": "Male",
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 5000,
        "coapplicant_income": 2000,
        "loan_amount": 300,
        "loan_term": 360,
        "credit_history": 1.0,
        "property_area": "Urban"
    }

    predict_new_customer(new_customer_data, model, scaler, feature_columns)