import pandas as pd
import numpy as np
import joblib
import struct

# === 1. Вхідні файли ===
data_file = "energy_data_norm.csv"
scaler_file = "scaler.pkl"
output_csv = "forecast_input.csv"
output_bin = "forecast_input.bin"

# === 2. Параметри ===
SEQ_DAYS = 30
FEATURES_PER_DAY = 7 + 24  # 7 погодних + 24 годинних

# === 3. Завантаження нормалізованих даних ===
df = pd.read_csv(data_file)
num_days = len(df)
if num_days < SEQ_DAYS:
    raise ValueError("Недостатньо даних для формування 30-денного вікна!")

# === 4. Вибір останніх 30 днів історії ===
input_block = df.iloc[-SEQ_DAYS:].copy()

# === 5. Прогноз погоди на завтра (до 7 ознак) ===
# У реальних умовах ці дані беруться з WeatherAPI або OpenWeatherMap
# Нижче приклад (у нормалізованому вигляді [0..1])
# Ви можете підставити реальні прогнози і нормалізувати через scaler.pkl
forecast_weather = {
    'day_of_year': 0.65,
    'day_of_week': 0.57,
    'is_holiday': 0.0,
    'avg_temp': 0.73,
    'precip': 0.12,
    'cloud': 0.45,
    'humidity': 0.58
}

# === 6. Створення вхідного вектора для прогнозу ===
# 30 днів історії + 7 параметрів погоди наступного дня
flattened_history = input_block.values.flatten()
forecast_features = np.array(list(forecast_weather.values()), dtype=np.float32)
input_vector = np.concatenate([flattened_history, forecast_features])

# === 7. Збереження CSV для перевірки ===
pd.DataFrame([input_vector]).to_csv(output_csv, index=False, header=False)
print(f"✅ Вхідні дані для прогнозу збережено у {output_csv}")
print(f"   Розмір вектора: {len(input_vector)} значень")

# === 8. Збереження у бінарний формат (для C99 / FANN) ===
with open(output_bin, "wb") as f:
    f.write(struct.pack('i', len(input_vector)))  # кількість float32
    f.write(struct.pack(f'{len(input_vector)}f', *input_vector))

print(f"💾 Бінарний файл створено: {output_bin}")
