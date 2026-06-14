import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import struct

# === 1. Файли ===
input_file = "energy_data.csv"
output_csv = "energy_data_norm.csv"
output_bin = "energy_data_norm.bin"
scaler_file = "scaler.pkl"

# === 2. Зчитування CSV ===
df = pd.read_csv(input_file)
print(f"📘 Завантажено {len(df)} днів із {input_file}")

# === 3. Нормалізація ===
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(df)
df_norm = pd.DataFrame(scaled_values, columns=df.columns)

# === 4. Збереження нормалізованого CSV ===
df_norm.to_csv(output_csv, index=False)
joblib.dump(scaler, scaler_file)

print(f"✅ CSV нормалізовано → {output_csv}")
print(f"💾 Масштабатор збережено → {scaler_file}")

# === 5. Формування бінарного файлу ===
# Використовуємо float32 для сумісності з C99
binary_data = scaled_values.astype(np.float32)
rows, cols = binary_data.shape

with open(output_bin, "wb") as f:
    # записуємо заголовок: кількість рядків і колонок (int32)
    f.write(struct.pack('ii', rows, cols))
    # записуємо дані пострядково
    for row in binary_data:
        f.write(struct.pack(f'{cols}f', *row))

print(f"✅ Бінарний файл сформовано → {output_bin}")
print(f"   Розмір: {rows} рядків × {cols} колонок (float32)")

"""
scaler = joblib.load("scaler.pkl")
# денормалізуємо DataFrame або масив
denorm = pd.DataFrame(scaler.inverse_transform(norm_values), columns=columns)
"""