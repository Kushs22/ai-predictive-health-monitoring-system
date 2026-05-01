import pandas as pd
import numpy as np

# ================================
# Load raw dataset
# ================================
data_path = "data/raw/patient_vitals.csv"
dataset = pd.read_csv(data_path)

# Convert timestamp to datetime
dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])

# Sort by patient and time
dataset = dataset.sort_values(by=["patient_id", "timestamp"])

# ================================
# Feature Engineering (Trend Analysis)
# ================================

# Calculate change (delta) for each patient
dataset["heart_rate_delta"] = dataset.groupby("patient_id")["heart_rate"].diff().fillna(0)
dataset["oxygen_saturation_delta"] = dataset.groupby("patient_id")["oxygen_saturation"].diff().fillna(0)

# ================================
# Trend Warning Score Logic
# ================================

def calculate_trend_score(row):
    score = 0

    # Heart rate abnormal change
    if abs(row["heart_rate_delta"]) > 10:
        score += 1

    # Oxygen drop warning
    if row["oxygen_saturation_delta"] < -2:
        score += 1

    # Low oxygen critical
    if row["oxygen_saturation"] < 94:
        score += 1

    return score

dataset["trend_warning_score"] = dataset.apply(calculate_trend_score, axis=1)

# ================================
# Risk Categorization
# ================================

def assign_risk_category(score):
    if score == 0:
        return "Stable"
    elif score == 1:
        return "Low Trend Risk"
    elif score == 2:
        return "Moderate Trend Risk"
    else:
        return "High Trend Risk"

dataset["trend_risk_category"] = dataset["trend_warning_score"].apply(assign_risk_category)

# ================================
# Final Risk Label (for ML)
# ================================

def create_risk_label(category):
    if category == "Stable":
        return 0
    elif category == "Low Trend Risk":
        return 1
    elif category == "Moderate Trend Risk":
        return 2
    else:
        return 3

dataset["risk_label"] = dataset["trend_risk_category"].apply(create_risk_label)

# ================================
# SAVE PROCESSED DATASET (IMPORTANT FIX)
# ================================

output_path = "data/processed/patient_vitals_processed.csv"
dataset.to_csv(output_path, index=False)

# ================================
# PRINT OUTPUT (FOR DEBUGGING)
# ================================

print("Trend and delta analysis completed successfully.")
print(f"Processed dataset saved to: {output_path}")
print(f"Dataset shape: {dataset.shape}")

print("\nSample Output:\n")
print(
    dataset[
        [
            "patient_id",
            "timestamp",
            "heart_rate",
            "heart_rate_delta",
            "oxygen_saturation",
            "oxygen_saturation_delta",
            "trend_warning_score",
            "trend_risk_category",
            "risk_label"
        ]
    ].head(15)
)