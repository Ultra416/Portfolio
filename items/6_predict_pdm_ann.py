import numpy as np
import torch
import torch.nn as nn

MODEL_FILE = "../models/pdm_ann.pth"

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
    X_test = np.load("../data/X_test.npy")
    y_test = np.load("../data/y_test.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PdMANN(X_test.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=False)) # добавлено weights_only=False для зникнення попередження
    model.eval()

    sample = X_test[-1]
    real = y_test[-1]

    x = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(x).cpu().numpy()[0, 0]

    prob = 1 / (1 + np.exp(-logit))
    pred = int(prob >= 0.5)

    print("=== PdM Prediction ===")
    print(f"Real label          : {int(real)}")
    print(f"Predicted class     : {pred}")
    print(f"Failure probability : {prob:.4f}")

if __name__ == "__main__":
    main()