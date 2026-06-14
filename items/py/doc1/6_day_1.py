import numpy as np
import pandas as pd
import struct

# === 1. Вхідні файли ===
norm_output_bin = "forecast_output_norm.bin"   # нормалізований прогноз (float32)
reference_csv = "energy_data.csv"              # оригінальні історичні дані для меж
output_csv = "forecast_output_denorm.csv"

# === 2. Завантаження нормалізованих прогнозів ===
with open(norm_output_bin, "rb") as f:
    n_values = struct.unpack('i', f.read(4))[0]
    y_norm = struct.unpack(f"{n_values}f", f.read(4 * n_values))
y_norm = np.array(y_norm, dtype=np.float32)

print(f"📘 Завантажено {n_values} нормалізованих значень з {norm_output_bin}")

# === 3. Визначення меж для денормалізації ===
cols = [f"h{i+1}" for i in range(24)]
df_ref = pd.read_csv(reference_csv)[cols]
mins = df_ref.min().values
maxs = df_ref.max().values

# === 4. Денормалізація ===
y_real = y_norm * (maxs - mins) + mins

# === 5. Вивід і збереження ===
for i, val in enumerate(y_real, start=1):
    print(f"Год {i:02d}: {val:.3f} МВт·год")

pd.DataFrame([y_real], columns=cols).to_csv(output_csv, index=False)
print(f"\n✅ Денормалізований прогноз збережено у {output_csv}")
