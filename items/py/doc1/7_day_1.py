import pandas as pd
import numpy as np
import struct
from sklearn.model_selection import train_test_split

# === 1. Вхідні файли ===
data_file = "energy_data_norm.csv"

# === 2. Параметри ===
SEQ_DAYS = 30
FEATURES_PER_DAY = 7 + 24
INPUT_SIZE = SEQ_DAYS * FEATURES_PER_DAY
OUTPUT_SIZE = 24
TEST_RATIO = 0.2

# === 3. Завантаження нормалізованих даних ===
df = pd.read_csv(data_file)
values = df.values.astype(np.float32)
num_days = len(values)

X, y = [], []
for i in range(SEQ_DAYS, num_days - 1):
    X.append(values[i-SEQ_DAYS:i].flatten())
    y.append(values[i, -24:])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"📘 Створено {len(X)} прикладів (вхід: {INPUT_SIZE}, вихід: {OUTPUT_SIZE})")

# === 4. Розділення на train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, shuffle=False)
print(f"🔹 Навчальних: {len(X_train)} | Тестових: {len(X_test)}")

# === 5. Функція для запису у бінарний формат ===
def save_bin(filename, array):
    rows, cols = array.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("ii", rows, cols))
        for row in array:
            f.write(struct.pack(f"{cols}f", *row))
    print(f"💾 {filename} збережено ({rows}×{cols})")

# === 6. Експорт усіх файлів ===
save_bin("train_input_1.bin", X_train)
save_bin("train_output_1.bin", y_train)
save_bin("test_input_1.bin", X_test)
save_bin("test_output_1.bin", y_test)

print("\n✅ Усі бінарні файли готові для використання у C99 / FANN")
