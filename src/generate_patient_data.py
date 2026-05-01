"""
====================================================
AI Predictive Health Monitoring System
Step 1: Synthetic Patient Vitals Data Generator
====================================================

Purpose:
This script generates realistic synthetic patient vital-sign data for an
AI-powered remote patient monitoring prototype.

The dataset includes:
- Heart rate
- Oxygen saturation
- Body temperature
- Respiratory rate
- Systolic blood pressure
- Risk label

Important:
Noise and variation are intentionally added so the model does not become
unrealistically perfect. This makes the project more realistic and
portfolio-ready.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# Reproducibility
np.random.seed(42)


def generate_patient_vitals(patient_id, start_time, periods=144):
    """
    Generate synthetic vital-sign time-series data for one patient.

    periods=144 means 24 hours of readings if data is recorded every 10 minutes.
    """

    # Create timestamps every 10 minutes
    timestamps = [
        start_time + timedelta(minutes=10 * i)
        for i in range(periods)
    ]

    # Give each patient a slightly different baseline
    patient_hr_baseline = np.random.normal(78, 6)
    patient_spo2_baseline = np.random.normal(97, 1)
    patient_temp_baseline = np.random.normal(36.8, 0.25)
    patient_rr_baseline = np.random.normal(16, 1.5)
    patient_bp_baseline = np.random.normal(120, 8)

    # Generate normal vital signs with realistic variation
    heart_rate = np.random.normal(patient_hr_baseline, 8, periods)
    oxygen_saturation = np.random.normal(patient_spo2_baseline, 1.5, periods)
    temperature = np.random.normal(patient_temp_baseline, 0.35, periods)
    respiratory_rate = np.random.normal(patient_rr_baseline, 2, periods)
    systolic_bp = np.random.normal(patient_bp_baseline, 10, periods)

    # Decide whether patient deteriorates
    deterioration = np.random.choice([0, 1], p=[0.7, 0.3])

    if deterioration == 1:
        # Deterioration begins later in the monitoring period
        deterioration_start = np.random.randint(periods // 2, periods - 20)

        # Gradual worsening trends
        trend_length = periods - deterioration_start

        heart_rate[deterioration_start:] += np.linspace(0, 25, trend_length)
        oxygen_saturation[deterioration_start:] -= np.linspace(0, 6, trend_length)
        temperature[deterioration_start:] += np.linspace(0, 1.2, trend_length)
        respiratory_rate[deterioration_start:] += np.linspace(0, 6, trend_length)
        systolic_bp[deterioration_start:] += np.linspace(0, 12, trend_length)

    # Add extra real-world noise so prediction is not unrealistically perfect
    heart_rate += np.random.normal(0, 4, periods)
    oxygen_saturation += np.random.normal(0, 1.0, periods)
    temperature += np.random.normal(0, 0.2, periods)
    respiratory_rate += np.random.normal(0, 1.2, periods)
    systolic_bp += np.random.normal(0, 5, periods)

    # Keep values within realistic medical ranges
    heart_rate = np.clip(heart_rate, 45, 160)
    oxygen_saturation = np.clip(oxygen_saturation, 82, 100)
    temperature = np.clip(temperature, 35.0, 41.0)
    respiratory_rate = np.clip(respiratory_rate, 8, 35)
    systolic_bp = np.clip(systolic_bp, 85, 190)

    # Risk label:
    # 0 = stable patient
    # 1 = patient with simulated deterioration pattern
    risk_label = deterioration

    patient_data = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": timestamps,
        "heart_rate": heart_rate.round(1),
        "oxygen_saturation": oxygen_saturation.round(1),
        "temperature": temperature.round(1),
        "respiratory_rate": respiratory_rate.round(1),
        "systolic_bp": systolic_bp.round(1),
        "risk_label": risk_label
    })

    return patient_data


def generate_dataset(num_patients=80):
    """
    Generate dataset for multiple patients.
    """

    all_patients = []
    start_time = datetime(2026, 1, 1, 8, 0, 0)

    for patient_id in range(1, num_patients + 1):
        patient_data = generate_patient_vitals(
            patient_id=patient_id,
            start_time=start_time
        )
        all_patients.append(patient_data)

    full_dataset = pd.concat(all_patients, ignore_index=True)

    return full_dataset


if __name__ == "__main__":

    # Generate synthetic dataset
    dataset = generate_dataset(num_patients=80)

    # Save raw dataset
    output_path = "data/raw/patient_vitals.csv"
    dataset.to_csv(output_path, index=False)

    # Print summary
    print("Synthetic patient vitals dataset created successfully.")
    print(f"Saved to: {output_path}")
    print(f"Dataset shape: {dataset.shape}")
    print("\nSample data:")
    print(dataset.head())
    print("\nRisk label distribution:")
    print(dataset["risk_label"].value_counts())