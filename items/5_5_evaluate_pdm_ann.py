import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
import matplotlib.pyplot as plt

MODEL_FILE = "../models/pdm_ann.pth"
CM_PLOT = "../plots/confusion_matrix_5.png"
ROC_PLOT = "../plots/roc_curve_5.png"

class PdMANN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def main():
    os.makedirs("../plots", exist_ok=True)

    X_test = np.load("../data/X_test.npy")
    y_test = np.load("../data/y_test.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PdMANN(X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=False)) # добавлено weights_only=False для зникнення попередження
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_test_t).cpu().numpy().reshape(-1)

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = float("nan")

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_PLOT, dpi=150)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROC_PLOT, dpi=150)
    plt.show()

    print(f"Збережено: {CM_PLOT}")
    print(f"Збережено: {ROC_PLOT}")

if __name__ == "__main__":
    main()