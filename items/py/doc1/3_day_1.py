import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# === 1. Вхідні дані ===
data_file = "energy_data_norm.csv"
model_file = "energy_ann.pth"

# === 2. Параметри ===
SEQ_DAYS = 30              # історія в днях
INPUT_FEATURES = 7 + 24    # 7 погодних + 24 годинних параметрів
INPUT_SIZE = SEQ_DAYS * INPUT_FEATURES
OUTPUT_SIZE = 24           # прогноз на 24 години
EPOCHS = 2500 # 10000
LR = 0.001

# === 3. Завантаження нормалізованих даних ===
df = pd.read_csv(data_file)
values = df.values.astype(np.float32)
num_days = len(values)
print(f"📘 Завантажено {num_days} днів з {data_file}")

# === 4. Формування навчальних прикладів ===
X, y = [], []
for i in range(SEQ_DAYS, num_days - 1):
    X.append(values[i-SEQ_DAYS:i].flatten())   # 30 днів історії
    y.append(values[i, -24:])                  # споживання наступного дня

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"🔹 Навчальних прикладів: {len(X)}  (вхід: {INPUT_SIZE}, вихід: {OUTPUT_SIZE})")

# === 5. Розділення на train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# === 6. Модель ===
class EnergyANN(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(EnergyANN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, OUTPUT_SIZE)
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


model = EnergyANN(INPUT_SIZE, OUTPUT_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === 7. Навчання ===
print("🚀 Починаємо навчання...")
for epoch in range(EPOCHS):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0: # 500
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test)
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

# === 8. Збереження моделі ===
torch.save(model.state_dict(), model_file)
print(f"✅ Модель збережено: {model_file}")

# === 9. Перевірка прогнозу для останніх даних ===
with torch.no_grad():
    last_input = torch.tensor(values[-SEQ_DAYS:].flatten()).unsqueeze(0)
    forecast = model(last_input).numpy().flatten()

print("🔮 Прогноз (нормалізований) на 24 години:")
print(np.round(forecast, 3))
