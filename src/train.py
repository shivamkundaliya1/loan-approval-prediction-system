"""
train.py
--------
Trains multiple classification models on the preprocessed data,
evaluates each one, performs hyperparameter tuning on Logistic
Regression, and saves the final tuned model + scaler to disk.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)

from preprocessing import (
    load_and_explore_data,
    remove_duplicates,
    analyze_columns,
    analyze_outliers,
    analyze_correlation,
    prepare_features_and_target,
    handle_missing_values,
    encode_categorical,
    split_and_scale,
)


def train_logistic_regression(X_train_scaled, Y_train, X_test_scaled, Y_test):
    """Train a baseline Logistic Regression model and print its metrics."""
    print("\nLogistic Regression:-")

    model = LogisticRegression()
    model.fit(X_train_scaled, Y_train)

    print("\nModel Training Completed!!!")

    y_pred = model.predict(X_test_scaled)
    print("\nPrediction:-")
    print(y_pred[:11])

    print("\nMetrics:-")
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, pos_label="Approved")
    recall = recall_score(Y_test, y_pred, pos_label="Approved")
    f1 = f1_score(Y_test, y_pred, pos_label="Approved")
    confusion = confusion_matrix(Y_test, y_pred)

    print("\nAccuracy:-", accuracy)
    print("Precision:-", precision)
    print("Recall:-", recall)
    print("F1 Score:-", f1)
    print("\nConfusion Matrix:-")
    print(confusion)

    return model


def train_decision_tree(X_train, Y_train, X_test, Y_test):
    """Train a Decision Tree classifier and print its metrics."""
    print("\nDecision Tree:-")

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, Y_train)
    dt_pred = dt_model.predict(X_test)

    print("Decision Tree training Completed!!")

    dt_accuracy = accuracy_score(Y_test, dt_pred)
    dt_precision = precision_score(Y_test, dt_pred, pos_label="Approved")
    dt_recall = recall_score(Y_test, dt_pred, pos_label="Approved")
    dt_f1 = f1_score(Y_test, dt_pred, pos_label="Approved")
    dt_confusion = confusion_matrix(Y_test, dt_pred)

    print("\nAccuracy:-", dt_accuracy)
    print("Precision:-", dt_precision)
    print("Recall:-", dt_recall)
    print("F1 Score:-", dt_f1)
    print("\nConfusion Matrix:-")
    print(dt_confusion)

    return dt_model


def train_random_forest(X_train, Y_train, X_test, Y_test):
    """Train a Random Forest classifier and print its metrics."""
    print("\nRandom Forest:-")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    rf_pred = rf_model.predict(X_test)

    print("Random Forest Training Completed!!")

    rf_accuracy = accuracy_score(Y_test, rf_pred)
    rf_precision = precision_score(Y_test, rf_pred, pos_label="Approved")
    rf_recall = recall_score(Y_test, rf_pred, pos_label="Approved")
    rf_f1 = f1_score(Y_test, rf_pred, pos_label="Approved")
    rf_confusion = confusion_matrix(Y_test, rf_pred)

    print("\nAccuracy:-", rf_accuracy)
    print("Precision:-", rf_precision)
    print("Recall:-", rf_recall)
    print("F1 Score:-", rf_f1)
    print("\nConfusion Matrix:-")
    print(rf_confusion)

    return rf_model


def train_knn(X_train_scaled, Y_train, X_test_scaled, Y_test):
    """Train a K-Nearest Neighbors classifier and print its metrics."""
    print("\nKNN:-")

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, Y_train)
    knn_pred = knn_model.predict(X_test_scaled)

    print("KNN Training Completed!!")

    knn_accuracy = accuracy_score(Y_test, knn_pred)
    knn_precision = precision_score(Y_test, knn_pred, pos_label="Approved")
    knn_recall = recall_score(Y_test, knn_pred, pos_label="Approved")
    knn_f1 = f1_score(Y_test, knn_pred, pos_label="Approved")
    knn_confusion = confusion_matrix(Y_test, knn_pred)

    print("\nAccuracy:-", knn_accuracy)
    print("Precision:-", knn_precision)
    print("Recall:-", knn_recall)
    print("F1 Score:-", knn_f1)
    print("\nConfusion Matrix:-")
    print(knn_confusion)

    return knn_model


def train_svm(X_train_scaled, Y_train, X_test_scaled, Y_test):
    """Train a Support Vector Machine classifier and print its metrics."""
    print("\nSVM:-")

    svm_model = SVC()
    svm_model.fit(X_train_scaled, Y_train)
    svm_pred = svm_model.predict(X_test_scaled)

    print("SVM Training Completed!!")

    svm_accuracy = accuracy_score(Y_test, svm_pred)
    svm_precision = precision_score(Y_test, svm_pred, pos_label="Approved")
    svm_recall = recall_score(Y_test, svm_pred, pos_label="Approved")
    svm_f1 = f1_score(Y_test, svm_pred, pos_label="Approved")
    svm_confusion = confusion_matrix(Y_test, svm_pred)

    print("\nSVM Metrics:-")
    print("Accuracy:-", svm_accuracy)
    print("Precision:-", svm_precision)
    print("Recall:-", svm_recall)
    print("F1 Score:-", svm_f1)
    print("\nConfusion Matrix:-")
    print(svm_confusion)

    return svm_model


def tune_logistic_regression(X_train_scaled, Y_train, X_test_scaled, Y_test):
    """Perform hyperparameter tuning on Logistic Regression using GridSearchCV."""
    print("\nHyperparameter Tuning:-")

    lr_model = LogisticRegression(max_iter=1000)

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"]
    }

    # Define "Approved" as the positive class for scoring
    f1_approved = make_scorer(f1_score, pos_label="Approved")

    grid_search = GridSearchCV(
        estimator=lr_model,
        param_grid=param_grid,
        cv=5,
        scoring=f1_approved
    )

    grid_search.fit(X_train_scaled, Y_train)

    print("\nBest Parameters:-")
    print(grid_search.best_params_)

    print("\nBest CV F1 Score:-")
    print(grid_search.best_score_)

    print("\nTuned Logistic Regression:-")
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_scaled, Y_train)

    tuned_pred = best_model.predict(X_test_scaled)
    print("Tuned Model Training Completed!!")

    tuned_accuracy = accuracy_score(Y_test, tuned_pred)
    tuned_precision = precision_score(Y_test, tuned_pred, pos_label="Approved")
    tuned_recall = recall_score(Y_test, tuned_pred, pos_label="Approved")
    tuned_f1 = f1_score(Y_test, tuned_pred, pos_label="Approved")
    tuned_confusion = confusion_matrix(Y_test, tuned_pred)

    print("\nTuned Model Metrics:-")
    print("Accuracy:-", tuned_accuracy)
    print("Precision:-", tuned_precision)
    print("Recall:-", tuned_recall)
    print("F1 Score:-", tuned_f1)
    print("\nConfusion Matrix:-")
    print(tuned_confusion)

    return best_model


def save_model(model, scaler, model_path="../loan_approval_model.pkl", scaler_path="../loan_approval_scaler.pkl"):
    """Save the trained model and scaler to disk using joblib."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("\nModel & Scaler Saved Successfully!!")


def main():
    # Load and explore data
    df = load_and_explore_data()
    df = remove_duplicates(df)
    analyze_columns(df)
    analyze_outliers(df)
    analyze_correlation(df)

    print("========================")
    print("\nData Preprocessing:-")
    print("========================")

    # Prepare features and target
    X, y = prepare_features_and_target(df)
    X = handle_missing_values(X)
    X = encode_categorical(X)

    # Split and scale
    X_train, X_test, X_train_scaled, X_test_scaled, Y_train, Y_test, scaler = split_and_scale(X, y)

    # Train baseline models
    train_logistic_regression(X_train_scaled, Y_train, X_test_scaled, Y_test)

    print("===================")
    train_decision_tree(X_train, Y_train, X_test, Y_test)

    print("\n===================")
    train_random_forest(X_train, Y_train, X_test, Y_test)

    print("======================")
    train_knn(X_train_scaled, Y_train, X_test_scaled, Y_test)

    print("\n======================")
    train_svm(X_train_scaled, Y_train, X_test_scaled, Y_test)

    print("\n======================")
    # Hyperparameter tuning -> final model
    best_model = tune_logistic_regression(X_train_scaled, Y_train, X_test_scaled, Y_test)

    # Save the final tuned model and scaler
    save_model(best_model, scaler)

    return best_model, scaler, X.columns


if __name__ == "__main__":
    main()