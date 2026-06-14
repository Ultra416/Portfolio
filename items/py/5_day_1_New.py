# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 14:15:47 2025

@author: CyberKids6
"""
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Файли ===
model_file = "energy_ann.pth"
data_norm_file = "energy_data_norm.csv" #Файл з нормалізованими даними (як у коді 2)
original_data_file = "energy_data.csv"   # Для денормалізації (мін/макс)
scaler_file = "scaler.pkl"

# === 2. Параметри ===
SEQ_DAYS = 30
FEATURES_PER_DAY = 7 + 24
OLD_INPUT_SIZE = SEQ_DAYS * FEATURES_PER_DAY       # 930 (30 днів * 31 ознаку)
INPUT_SIZE = OLD_INPUT_SIZE + 7                    # 937 (Додаємо 7 ознак погоди на день прогнозу)
OUTPUT_SIZE = 24 

Delta = 0.045
# === 3. Підготовка даних (ЗАМІСТЬ БІНАРНОГО ФАЙЛУ) ===
print("📂 Завантаження даних для тесту...")

# Завантажуємо нормалізовані дані
df_norm = pd.read_csv(data_norm_file)
data_values = df_norm.values.astype(np.float32)

# Вибираємо випадковий день для тесту (або останній доступний)
# Нам потрібно мати 30 днів історії ДО цього дня
test_index = len(data_values) - 1  # Беремо останній доступний день у файлі
# test_index = 500  # Розкоментуй, щоб взяти будь-який інший день із середини
# test_index = 180

if test_index < SEQ_DAYS:
    print("❌ Недостатньо даних для історії.")
    exit()

# А. Беремо історію за 30 днів (30 рядків по 31 ознаці)
history_data = data_values[test_index - SEQ_DAYS : test_index] # shape (30, 31)
history_flat = history_data.flatten() # shape (930,)

# Б. Беремо погоду на день прогнозу (перші 7 колонок рядка test_index)
target_weather = data_values[test_index, :7] # shape (7,)

# В. Беремо РЕАЛЬНЕ споживання на цей день (останні 24 колонки) для перевірки
target_actual_norm = data_values[test_index, -24:] # shape (24,)

# Г. Формуємо вхідний вектор для нової моделі (930 історії + 7 погоди = 937)
input_vector = np.concatenate([history_flat, target_weather])
print(f"📘 Сформовано вхідний вектор розміром: {input_vector.shape[0]}")


# === 4. Модель (Transfer Learning) ===
class EnergyANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(EnergyANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""
class EnergyANN(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(EnergyANN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        #self.fc5 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(64, OUTPUT_SIZE)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2) 

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # Опціонально: відключає частину нейронів для кращого навчання
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x)) # Новий шар
        x = self.relu(self.fc4(x)) # Новий шар
        #x = self.relu(self.fc5(x))
        x = self.fc_out(x)
        return x """

# Створюємо нову модель
model = EnergyANN(INPUT_SIZE, OUTPUT_SIZE)

# Завантажуємо старі ваги (від моделі на 930 входів)
try:
    old_state = torch.load(model_file, map_location='cpu')
except FileNotFoundError:
    print(f"❌ Файл {model_file} не знайдено.")
    exit()

new_state = model.state_dict()

with torch.no_grad():
    # Копіюємо ваги для перших 930 входів
    new_state['fc1.weight'][:, :OLD_INPUT_SIZE] = old_state['fc1.weight']
    
    # Для нових 7 входів (погода) ініціалізуємо ваги нулями або залишаємо випадковими
    # Краще нулями, щоб на початку погода не вносила хаос, поки модель не довчиться
    new_state['fc1.weight'][:, OLD_INPUT_SIZE:] = 0.0 
    
    new_state['fc1.bias'] = old_state['fc1.bias']
    new_state['fc2.weight'] = old_state['fc2.weight']
    new_state['fc2.bias'] = old_state['fc2.bias']
    new_state['fc3.weight'] = old_state['fc3.weight']
    new_state['fc3.bias'] = old_state['fc3.bias']
    """
    new_state['fc4.weight'] = old_state['fc4.weight']
    new_state['fc4.bias'] = old_state['fc4.bias']
    
    new_state['fc5.weight'] = old_state['fc5.weight']
    new_state['fc5.bias'] = old_state['fc5.bias']
    """

model.load_state_dict(new_state)
model.eval()
print("✅ Ваги перенесено. Модель готова прогнозувати з урахуванням точної погоди.")


# === 5. Прогноз ===
x = torch.tensor(input_vector).unsqueeze(0) # додаємо batch dimension
with torch.no_grad():
    y_pred_norm = model(x).numpy().flatten()

print("🔮 Прогноз зроблено.")

# === 6. Денормалізація ===
# Беремо min/max з оригінального файлу для відновлення реальних значень
df_orig = pd.read_csv(original_data_file)
cols_h = [f"h{i+1}" for i in range(24)]
mins = df_orig[cols_h].min().values
maxs = df_orig[cols_h].max().values

# Відновлюємо прогноз і реальні дані
y_pred_real = (y_pred_norm * (maxs - mins) + mins) + Delta
y_actual_real = target_actual_norm * (maxs - mins) + mins

# === 7. Виведення та Графік ===
print("\n📊 Результати (МВт·год):")
print(f"{'Година':<10} | {'Прогноз':<15} | {'Реальність':<15} | {'Різниця':<15}")
print("-" * 65)

mse = 0
for i in range(24):
    diff = y_pred_real[i] - y_actual_real[i]
    mse += diff**2
    print(f"{i+1:02d}:00      | {y_pred_real[i]:.2f}          | {y_actual_real[i]:.2f}          | {diff:+.2f}")

rmse = np.sqrt(mse / 24)
print("-" * 65)
print(f"📉 Середня помилка (RMSE): {rmse:.2f} МВт·год")

# Графік
plt.figure(figsize=(12, 6))
plt.plot(range(1, 25), y_actual_real, marker="o", label="Реальне споживання", color='blue')
plt.plot(range(1, 25), y_pred_real, marker="x", linestyle="--", label="Прогноз моделі", color='red')

plt.title("Порівняння прогнозу з реальними даними (Архітектура 937 входів)")
plt.xlabel("Година доби")
plt.ylabel("Споживання (МВт·год)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Збереження
cols_out = [f"h{i+1}" for i in range(24)]
pd.DataFrame([y_pred_real], columns=cols_out).to_csv("forecast_output_test.csv", index=False)
print("\n💾 Прогноз збережено у forecast_output_test.csv")