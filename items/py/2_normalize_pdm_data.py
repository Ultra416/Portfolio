import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "../data/pdm_data.csv"
OUTPUT_FILE = "../data/pdm_data_norm.csv"
SCALER_FILE = "../data/scaler.pkl"

TARGET_COL = "failure_in_next_24_steps"

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "ambient_temperature",
    "humidity",
    "load_percent",
    "current",
    "voltage",
    "power",
    "vibration_rms",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "shift_id",
    "hours_since_last_maintenance",
    "maintenance_count_last_30_steps",
    "warning_count_last_24_steps"
]

def main():
    df = pd.read_csv(INPUT_FILE)

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    out = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    out[TARGET_COL] = y.values

    out.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, SCALER_FILE)

    print(f"Готово: {OUTPUT_FILE}")
    print(f"Scaler: {SCALER_FILE}")
    print(out.head())

if __name__ == "__main__":
    main()
