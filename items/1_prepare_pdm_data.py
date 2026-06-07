import pandas as pd
import numpy as np
import os

INPUT_FILE = "../data/ai4i2020.csv"
OUTPUT_FILE = "../data/pdm_data.csv"

def add_time_features(df):
    n = len(df)
    df["time_step"] = np.arange(n)
    df["hour"] = df["time_step"] % 24
    df["day_index"] = df["time_step"] // 24
    df["day_of_week"] = df["day_index"] % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = (df["day_index"] // 30) % 12 + 1
    df["is_holiday"] = ((df["day_index"] % 30) == 0).astype(int)
    df["shift_id"] = pd.cut(
        df["hour"],
        bins=[-1, 7, 15, 23],
        labels=[0, 1, 2]
    ).astype(int)
    return df

def add_environment_features(df):
    np.random.seed(42)
    df["ambient_temperature"] = (
        20
        + 8 * np.sin(2 * np.pi * df["hour"] / 24)
        + 4 * np.sin(2 * np.pi * df["month"] / 12)
        + np.random.normal(0, 1.2, len(df))
    )
    df["humidity"] = (
        55
        + 10 * np.sin(2 * np.pi * df["hour"] / 24 + 1.5)
        + np.random.normal(0, 3, len(df))
    )
    df["ambient_temperature"] = df["ambient_temperature"].round(2)
    df["humidity"] = np.clip(df["humidity"], 20, 95).round(2)
    return df

def add_operating_features(df):
    torque_col = "Torque [Nm]"
    rpm_col = "Rotational speed [rpm]"
    wear_col = "Tool wear [min]"

    torque_norm = (df[torque_col] - df[torque_col].min()) / (df[torque_col].max() - df[torque_col].min() + 1e-8)
    rpm_norm = (df[rpm_col] - df[rpm_col].min()) / (df[rpm_col].max() - df[rpm_col].min() + 1e-8)
    wear_norm = (df[wear_col] - df[wear_col].min()) / (df[wear_col].max() - df[wear_col].min() + 1e-8)

    df["load_percent"] = (0.45 * torque_norm + 0.35 * rpm_norm + 0.20 * wear_norm) * 100
    df["load_percent"] = df["load_percent"].round(2)

    df["current"] = (4.0 + 0.08 * df["load_percent"] + np.random.normal(0, 0.2, len(df))).round(3)
    df["voltage"] = (220 + np.random.normal(0, 2.5, len(df))).round(3)
    df["power"] = (df["current"] * df["voltage"] / 1000.0).round(3)

    df["vibration_rms"] = (
        0.3
        + 0.0035 * df[wear_col]
        + 0.0020 * df["load_percent"]
        + np.random.normal(0, 0.03, len(df))
    )
    df["vibration_rms"] = np.clip(df["vibration_rms"], 0.05, None).round(4)

    return df

def add_maintenance_features(df):
    failure_col = "Machine failure"
    n = len(df)

    hours_since_last_maintenance = []
    maintenance_count_last_30_steps = []
    warning_count_last_24_steps = []

    last_maintenance = -1
    maintenance_log = []
    warning_log = []

    for i in range(n):
        if i == 0:
            hours_since_last_maintenance.append(0)
        else:
            hours_since_last_maintenance.append(i - last_maintenance - 1 if last_maintenance >= 0 else i)

        base_warn = 0
        if df.loc[i, "Tool wear [min]"] > df["Tool wear [min]"].quantile(0.8):
            base_warn += 1
        if df.loc[i, "Process temperature [K]"] > df["Process temperature [K]"].quantile(0.85):
            base_warn += 1
        if df.loc[i, "Torque [Nm]"] > df["Torque [Nm]"].quantile(0.85):
            base_warn += 1

        warning_log.append(base_warn)
        warning_count_last_24_steps.append(sum(warning_log[max(0, i - 23):i + 1]))

        if df.loc[i, failure_col] == 1:
            last_maintenance = i
            maintenance_log.append(i)

        maintenance_count_last_30_steps.append(sum(x >= i - 29 for x in maintenance_log))

    df["hours_since_last_maintenance"] = hours_since_last_maintenance
    df["maintenance_count_last_30_steps"] = maintenance_count_last_30_steps
    df["warning_count_last_24_steps"] = warning_count_last_24_steps
    return df

def add_target(df, horizon=24):
    failure_col = "Machine failure"
    y = np.zeros(len(df), dtype=int)

    failure_idx = df.index[df[failure_col] == 1].tolist()

    for idx in failure_idx:
        start = max(0, idx - horizon)
        y[start:idx] = 1

    df[f"failure_in_next_{horizon}_steps"] = y
    return df

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Не знайдено файл: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    drop_cols = [c for c in ["UDI", "Product ID"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = add_time_features(df)
    df = add_environment_features(df)
    df = add_operating_features(df)
    df = add_maintenance_features(df)
    df = add_target(df, horizon=24)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Готово: {OUTPUT_FILE}")
    print(df.head())
    print(df.columns.tolist())

if __name__ == "__main__":
    main()