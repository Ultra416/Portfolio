import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

MODEL_FILE = "../models/pdm_ann.pth"
LOSS_PLOT = "../plots/loss.png"

BATCH_SIZE = 64
EPOCHS = 80
LR = 0.001
PATIENCE = 10

class PdMANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

def main():
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../plots", exist_ok=True)

    X_train = np.load("../data/X_train.npy")
    X_test = np.load("../data/X_test.npy")
    y_train = np.load("../data/y_train.npy")
    y_test = np.load("../data/y_test.npy")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PdMANN(X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_train += loss.item() * xb.size(0)

        train_loss = total_train / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_val += loss.item() * xb.size(0)

        val_loss = total_val / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PdM ANN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=150)
    plt.show()

    print(f"Модель збережена: {MODEL_FILE}")
    print(f"Графік збережено: {LOSS_PLOT}")

if __name__ == "__main__":
    main()