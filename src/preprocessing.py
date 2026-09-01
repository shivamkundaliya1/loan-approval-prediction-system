"""
preprocessing.py
-----------------
Loads the raw dataset, cleans it, and prepares it for model training:
loading, EDA prints, duplicate removal, missing value handling,
categorical encoding, train-test split, and feature scaling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_explore_data(path="../data/loan_approval.csv"):
    """Load the dataset and print basic exploratory information."""
    df = pd.read_csv(path)

    print("Dataset loaded successfully:")

    print("\nHead of Dataset:")
    print(df.head())

    print("\nShape Of Dataset:-")
    print(df.shape)

    print("\nDataset Information:-")
    print(df.info())

    print("\nDescribe:-")
    print(df.describe())

    print("\nColumn Names:-")
    print(df.columns)

    print("\nUnique Values:-")
    print(df.nunique())

    print("\nLoan_status Count:-")
    print(df["loan_status"].value_counts())

    print("\nLoan_Status Percentage:-")
    print(df["loan_status"].value_counts(normalize=True) * 100)

    print("\nNumerical columns:")
    print(df.select_dtypes(include="number").columns)

    print("\nCategorical columns:")
    print(df.select_dtypes(include="object").columns)

    print("\nMissing Values Analysis:-")
    print(df.isnull().sum())

    print("\nDuplicate Rows Count:-")
    print(df.duplicated().sum())

    return df


def remove_duplicates(df):
    """Drop duplicate rows from the dataframe."""
    df = df.drop_duplicates()
    print("\nShape After Removing Duplicates:-")
    print(df.shape)
    return df


def analyze_columns(df):
    """Print unique values for categorical columns and stats for numerical columns."""
    print("\nUnique Values in Categorical Columns:-")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n{col}:")
        print(df[col].unique())

    print("\nImportant Values in Numerical Columns:-")
    for num_col in df.select_dtypes(include="number").columns:
        print(f"\n{num_col}:")
        print(df[num_col].agg(["min", "max", "mean", "median"]))


def analyze_outliers(df):
    """Detect outliers in numerical columns using the IQR method."""
    print("\nOutlier Analysis:")
    for num_col in df.select_dtypes(include="number").columns:
        Q1 = df[num_col].quantile(0.25)
        Q3 = df[num_col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[num_col] < lower) | (df[num_col] > upper)]

        print(f"\n{num_col}:")
        print("Q1:-", Q1)
        print("Q3:-", Q3)
        print("IQR:", IQR)
        print("Lower Bound:", lower)
        print("Upper Bound:", upper)
        print("Outlier Count:", len(outliers))


def analyze_correlation(df):
    """Print skewness and correlation matrix for numerical columns."""
    print("\nSkewness Analysis:-")
    for num_col in df.select_dtypes(include="number").columns:
        print(f"\n{num_col}:")
        print(df[num_col].skew())

    print("\nCorrelation matrix:-")
    print(df.select_dtypes(include="number").corr())

    print("\nCorrelation With Loan Amount:-")
    print(
        df.select_dtypes(include="number")
        .corr()["loan_amount"]
        .sort_values(ascending=False)
    )


def prepare_features_and_target(df):
    """Split the dataframe into feature matrix (X) and target vector (y)."""
    X = df[[
        "gender",
        "married",
        "dependents",
        "education",
        "self_employed",
        "applicant_income",
        "coapplicant_income",
        "loan_amount",
        "loan_term",
        "credit_history",
        "property_area"
    ]]

    y = df["loan_status"]

    print("\nFeatures (X):")
    print(X.head())

    print("\nTarget (y):")
    print(y.head())

    print("\nX Shape:", X.shape)
    print("y Shape:", y.shape)

    print("\nMissing Values in Features:-")
    print(X.isnull().sum())

    print("\nMissing Values in Target:-")
    print(y.isnull().sum())

    return X, y


def handle_missing_values(X):
    """Fill missing values: mode for categorical columns, median for numerical columns."""
    X = X.copy()

    cat_cols = ["gender", "married", "dependents", "self_employed"]
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    num_col = ["loan_amount", "loan_term", "credit_history"]
    for col in num_col:
        X[col] = X[col].fillna(X[col].median())

    print(X.dtypes)
    return X


def encode_categorical(X):
    """One-hot encode categorical columns (drop_first=True to avoid multicollinearity)."""
    X = pd.get_dummies(
        X,
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

    print("\nEncoded Data:")
    print(X.head())

    print("\nX Shape:")
    print(X.shape)

    return X


def split_and_scale(X, y):
    """Split into train/test sets and scale the features."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print("\nX_Train Shape:-", X_train.shape)
    print("X_Test Shape:-", X_test.shape)
    print("Y_Train Shape:-", Y_train.shape)
    print("Y_Test Shape:-", Y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nScaled X_Train Shape:-", X_train_scaled.shape)
    print("Scaled X_Test Shape:-", X_test_scaled.shape)

    return X_train, X_test, X_train_scaled, X_test_scaled, Y_train, Y_test, scaler