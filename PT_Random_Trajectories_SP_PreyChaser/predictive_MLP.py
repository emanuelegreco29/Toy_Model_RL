import os
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class PredictiveModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.feature = nn.Sequential(*layers)
        self.head = nn.Linear(last, 3)

    def forward(self, x):
        return self.head(self.feature(x))

def train_predictive(dataset_path, model_path,
                     batch_size=512, lr=1e-3, epochs=20, device="cpu"):
    data = np.load(dataset_path)
    print(f"Loaded dataset.")
    states = torch.from_numpy(data["states"])
    next_pos = torch.from_numpy(data["next_positions"])
    dataset = TensorDataset(states, next_pos)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = PredictiveModel(input_dim=states.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        print(f"Starting epoch {epoch}...")
        model.train()
        total_loss = 0.0
        for batch_states, batch_next in loader:
            batch_states = batch_states.to(device)
            batch_next = batch_next.to(device)
            optimizer.zero_grad()
            preds = model(batch_states)
            loss = criterion(preds, batch_next)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_states.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved predictive model to {model_path}")


ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

PATH = "dataset_50000.npz"
SAVE_PATH = os.path.join("models", f"predictive_{ts}.pth")
batch_size = 512
lr = 1e-3
epochs = 20
device = "cuda"

train_predictive(
    dataset_path=PATH,
    model_path=SAVE_PATH,
    batch_size=batch_size,
    lr=lr,
    epochs=epochs,
    device=device
)