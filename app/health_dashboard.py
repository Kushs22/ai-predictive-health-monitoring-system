"""
AI Predictive Health Monitoring Dashboard

This Streamlit app displays patient vitals, trend risk scores,
machine-learning based risk prediction, explanations, and feature importance.
"""

import os
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Allow app to import files from src folder
sys.path.append(os.path.abspath("src"))


@st.cache_data
def load_data():
    """
    Load processed patient vitals dataset.
    """

    data_path = "data/processed/patient_vitals_processed.csv"
    data = pd.read_csv(data_path)

    # Convert timestamp column into datetime format
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data


@st.cache_resource
def train_model(data):
    """
    Train a Random Forest model using patient vital-sign and trend features.
    """

    features = [
        "heart_rate",
        "oxygen_saturation",
        "temperature",
        "respiratory_rate",
        "systolic_bp",
        "heart_rate_delta",
        "oxygen_saturation_delta",
    ]

    X = data[features]
    y = data["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, features


def generate_risk_explanation(row, prediction):
    """
    Generate a simple human-readable explanation for the prediction.
    """

    explanations = []

    if row["heart_rate_delta"] > 10:
        explanations.append("heart rate increased sharply")

    if row["oxygen_saturation_delta"] < -2:
        explanations.append("oxygen saturation dropped noticeably")

    if row["oxygen_saturation"] < 94:
        explanations.append("oxygen saturation is below the safe range")

    if row["trend_warning_score"] >= 2:
        explanations.append("multiple trend-based warning signals are present")

    if not explanations:
        explanations.append("vital signs appear relatively stable")

    risk_map = {
        0: "Stable",
        1: "Low Risk",
        2: "Moderate Risk",
        3: "High Risk"
    }

    risk_label = risk_map.get(prediction, "Unknown Risk")

    return (
        f"Predicted category: {risk_label}. "
        f"Reason: {', '.join(explanations)}."
    )


# ================================
# Streamlit Page Setup
# ================================

st.set_page_config(
    page_title="AI Predictive Health Monitoring",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Predictive Health Monitoring System")

st.write(
    "This prototype uses patient vital-sign trends to support early warning "
    "and predictive remote patient monitoring."
)

# Load processed dataset
data = load_data()

# Train model
model, features = train_model(data)


# ================================
# Sidebar Patient Selection
# ================================

st.sidebar.header("Patient Selection")

patient_ids = sorted(data["patient_id"].unique())

selected_patient = st.sidebar.selectbox(
    "Select Patient ID",
    patient_ids
)

patient_data = data[data["patient_id"] == selected_patient]

# Select latest available record for selected patient
latest_record = patient_data.iloc[-1]


# ================================
# Current Patient Status
# ================================

st.subheader("📌 Current Patient Status")

# Prepare latest patient row for prediction
latest_features = latest_record[features].to_frame().T

# Predict risk class
prediction = model.predict(latest_features)[0]

# Predict probability/confidence
prediction_proba = model.predict_proba(latest_features)[0]
confidence = round(max(prediction_proba) * 100, 2)

# Map numeric prediction to readable label
risk_map = {
    0: "Stable",
    1: "Low Risk",
    2: "Moderate Risk",
    3: "High Risk"
}

predicted_risk = risk_map.get(prediction, "Unknown Risk")

col1, col2, col3 = st.columns(3)

# Color-coded risk output
with col1:
    st.write("Predicted Risk")

    if predicted_risk == "High Risk":
        st.error(predicted_risk)
    elif predicted_risk == "Moderate Risk":
        st.warning(predicted_risk)
    elif predicted_risk == "Low Risk":
        st.success(predicted_risk)
    else:
        st.info(predicted_risk)

with col2:
    st.metric("Confidence", f"{confidence}%")

with col3:
    st.metric("Trend Warning Score", int(latest_record["trend_warning_score"]))


# ================================
# AI Explanation
# ================================

st.subheader("🧠 AI Explanation")

explanation = generate_risk_explanation(
    latest_record,
    prediction
)

st.write(explanation)


# ================================
# Feature Importance
# ================================

st.subheader("📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))


# ================================
# Patient Vital Trend Chart
# ================================

st.subheader("📈 Patient Vital Trends")

vital_options = [
    "heart_rate",
    "oxygen_saturation",
    "temperature",
    "respiratory_rate",
    "systolic_bp"
]

selected_vital = st.selectbox(
    "Select vital sign to visualise",
    vital_options
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    patient_data["timestamp"],
    patient_data[selected_vital]
)

ax.set_title(f"{selected_vital.replace('_', ' ').title()} Trend")
ax.set_xlabel("Time")
ax.set_ylabel(selected_vital.replace("_", " ").title())

plt.xticks(rotation=45)

st.pyplot(fig)


# ================================
# Trend Warning Score Chart
# ================================

st.subheader("⚠️ Trend Warning Score Over Time")

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(
    patient_data["timestamp"],
    patient_data["trend_warning_score"]
)

ax2.set_title("Trend Warning Score Timeline")
ax2.set_xlabel("Time")
ax2.set_ylabel("Warning Score")

plt.xticks(rotation=45)

st.pyplot(fig2)


# ================================
# Patient Data Preview
# ================================

st.subheader("📋 Patient Data Preview")

st.dataframe(patient_data.tail(10))


# ================================
# Safety Notice
# ================================

st.subheader("⚠️ Safety Notice")

st.info(
    "This prototype is for educational and decision-support purposes only. "
    "It is not clinically validated and must not be used for real medical diagnosis."
)