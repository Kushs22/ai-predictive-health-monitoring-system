"""
====================================================
AI Predictive Health Monitoring System
Step 3: Train Machine Learning Risk Prediction Model
====================================================

Purpose:
This script trains a Random Forest model to predict patient risk
using vital signs and trend-based features.

It also saves the trained model so the Streamlit dashboard can load it
without retraining every time.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ================================
# Load processed dataset
# ================================

data_path = "data/processed/patient_vitals_processed.csv"
df = pd.read_csv(data_path)


# ================================
# Select model features
# ================================

features = [
    "heart_rate",
    "oxygen_saturation",
    "temperature",
    "respiratory_rate",
    "systolic_bp",
    "heart_rate_delta",
    "oxygen_saturation_delta",
]

X = df[features]
y = df["risk_label"]


# ================================
# Train-test split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ================================
# Train Random Forest model
# ================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# ================================
# Make predictions
# ================================

y_pred = model.predict(X_test)


# ================================
# Evaluate model
# ================================

accuracy = accuracy_score(y_test, y_pred)

print("Model training completed successfully.")
print(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ================================
# Save trained model
# ================================

os.makedirs("models", exist_ok=True)

model_path = "models/risk_model.pkl"
joblib.dump(model, model_path)

print(f"\nTrained model saved successfully to: {model_path}")