# ============================================
# Disease Risk Prediction (Diabetes)
# Author: Asante Kofi Akrasi
# ============================================

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# ----------- LOAD DATA -----------
def load_data(path="diabetes.csv"):
    print("Loading dataset...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        sys.exit(1)
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit(1)
    print("Dataset loaded successfully.\n")
    return df


# ----------- DATA CLEANING -----------
def clean_data(df):
    print("Cleaning data...")

    cols_missing_zero = ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']

    for col in cols_missing_zero:
        if col in df.columns:
            non_zero = df.loc[df[col] != 0, col]

            if len(non_zero) > 0:
                replacement = non_zero.median()
            else:
                replacement = df[col].median()

            df[col] = df[col].replace(0, replacement)

    print("Data cleaning complete.\n")
    return df


# ----------- EDA -----------
def perform_eda(df):
    print("Performing EDA...")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.histplot(df['Glucose'], kde=True)
    plt.title("Glucose Distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Outcome', data=df)
    plt.title("Outcome Distribution (0 = No Diabetes, 1 = Diabetes)")
    plt.tight_layout()
    plt.show()

    print("EDA complete.\n")


# ----------- DATA PREPARATION -----------
def prepare_data(df, test_size=0.2, random_state=RANDOM_STATE):
    print("Preparing data...")

    if 'Outcome' not in df.columns:
        raise ValueError("Dataset must contain 'Outcome' column")

    x = df.drop('Outcome', axis=1)
    y = df['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    feature_names = list(x.columns)

    print("Data preparation complete.\n")
    return x_train, x_test, y_train, y_test, feature_names


# ----------- TRAIN MODELS -----------
def train_models(x_train, y_train):
    print("Training models...")

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(x_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(x_train, y_train)

    print("Models trained successfully.\n")
    return lr, rf


# ----------- EVALUATION -----------
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"{name} Results:")
    print("  Accuracy:", round(acc, 4))
    print("  Precision:", round(prec, 4))
    print("  Recall:", round(rec, 4))
    print()

    return y_pred


# ----------- CONFUSION MATRIX -----------
def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# ----------- FEATURE IMPORTANCE -----------
def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        # Sort features
        sorted_indices = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_indices]
        sorted_names = np.array(feature_names)[sorted_indices]

        plt.figure(figsize=(8, 6))
        sns.barplot(x=sorted_importances, y=sorted_names)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()
    else:
        print("Model does not support feature importance.\n")


# ----------- MAIN -----------
def main():
    print("=== DISEASE RISK PREDICTION PROJECT ===\n")

    df = load_data("diabetes.csv")
    df = clean_data(df)
    perform_eda(df)

    x_train, x_test, y_train, y_test, feature_names = prepare_data(df)

    lr, rf = train_models(x_train, y_train)

    y_pred_lr = evaluate_model(lr, x_test, y_test, "Logistic Regression")
    y_pred_rf = evaluate_model(rf, x_test, y_test, "Random Forest")

    plot_confusion_matrix(y_test, y_pred_rf, title="Confusion Matrix (Random Forest)")
    plot_feature_importance(rf, feature_names)

    print("Project completed successfully!")


if __name__ == "__main__":
    main()