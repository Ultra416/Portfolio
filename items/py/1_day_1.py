import pandas as pd

# === 1. Вхідні та вихідні файли ===
input_file = "energy_data.csv"
output_file = "normalized_energy_data.csv"

# === 2. Зчитування єдиного файлу ===
print(f"📖 Читання файлу {input_file}...")
df = pd.read_csv(input_file)

# === 3. Перевірка та підготовка дат ===
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    
    # Вираховуємо календарні параметри
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.weekday  # 0=Пн, 6=Нд
    
    # Маркування вихідних (субота=5, неділя=6)
    df['is_holiday'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
else:
    print("⚠️ Попередження: Колонку 'date' не знайдено. Календарні параметри не оновлено.")

# === 4. НОРМАЛІЗАЦІЯ годинних колонок (h1-h24) ===
hours_cols = [f"h{i}" for i in range(1, 25)]

if all(col in df.columns for col in hours_cols):
    MAX_CZ = 12000
    # Ділимо на MAX_CZ, щоб отримати значення від 0 до 1
    df[hours_cols] = df[hours_cols] / MAX_CZ
    print(f"📉 Дані годин h1-h24 успішно НОРМАЛІЗОВАНО (поділено на {MAX_CZ} МВт)")
else:
    print("⚠️ Попередження: Колонки h1-h24 не знайдено у файлі!")

# === 5. Впорядкування колонок ===
# Збираємо правильну послідовність: календар -> погода -> години
weather_cols = [col for col in ['avg_temp', 'precip', 'cloud', 'humidity'] if col in df.columns]
base_cols = [col for col in ['day_of_year', 'day_of_week', 'is_holiday'] if col in df.columns]

final_cols = base_cols + weather_cols + [col for col in hours_cols if col in df.columns]
df_final = df[final_cols]

# === 6. Збереження результату ===
df_final.to_csv(output_file, index=False)
print(f"✅ Готово! Оброблено {len(df_final)} днів і збережено у '{output_file}'")

"""
import pandas as pd
import datetime as dt

# === 1. Вхідні файли ===
# Ці шляхи треба змінити під ваші CSV
weather_file = "weather.csv"       # прогноз або архів погоди
consumption_file = "consumption.csv"  # історія споживання (по годинах)
output_file = "energy_data.csv"

# === 2. Зчитування даних ===
weather = pd.read_csv(weather_file)
consumption = pd.read_csv(consumption_file)

# Очікувані колонки у файлах:
# weather: date, avg_temp, precip, cloud, humidity
# consumption: date, hour, load_mwh

# === 3. Підготовка базових календарних параметрів ===
weather['date'] = pd.to_datetime(weather['date'])
consumption['date'] = pd.to_datetime(consumption['date'])

# день року, день тижня
weather['day_of_year'] = weather['date'].dt.dayofyear
weather['day_of_week'] = weather['date'].dt.weekday  # 0=Mon,6=Sun

# === 4. Маркування вихідних і свят ===
def is_holiday_func(d):
    # У вас може бути власний список свят
    weekends = [5, 6]  # субота, неділя
    return 1 if d.weekday() in weekends else 0

weather['is_holiday'] = weather['date'].apply(is_holiday_func)

# === 5. Агрегуємо споживання по добах (24 колонки) ===
pivot = consumption.pivot_table(index='date', columns='hour', values='load_mwh')
pivot.columns = [f"h{i+1}" for i in range(24)]
pivot.reset_index(inplace=True)

# === 6. Об’єднуємо погоду і споживання ===
merged = pd.merge(weather, pivot, on='date', how='inner')

# === 7. Вибір та порядок колонок ===
cols = ['day_of_year', 'day_of_week', 'is_holiday',
        'avg_temp', 'precip', 'cloud', 'humidity'] + [f"h{i+1}" for i in range(24)]
merged = merged[cols]

# === 8. Збереження у CSV ===
merged.to_csv(output_file, index=False)
print(f"✅ Об’єднано {len(merged)} днів і збережено у {output_file}")
"""