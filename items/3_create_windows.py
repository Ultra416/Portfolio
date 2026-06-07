import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_FILE = "../data/pdm_data_norm.csv"

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

WINDOW = 24
TEST_SIZE = 0.2

def make_windows(df, feature_cols, target_col, window):
    X, y = [], []
    vals = df[feature_cols].values
    tgt = df[target_col].values

    for i in range(window, len(df)):
        X.append(vals[i - window:i].reshape(-1))
        y.append(tgt[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    df = pd.read_csv(INPUT_FILE)
    X, y = make_windows(df, FEATURE_COLS, TARGET_COL, WINDOW)

    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    np.save("../data/X_train.npy", X_train)
    np.save("../data/X_test.npy", X_test)
    np.save("../data/y_train.npy", y_train)
    np.save("../data/y_test.npy", y_test)

    print("Готово:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

if __name__ == "__main__":
    main()