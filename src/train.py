import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import RFCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

data = np.load("data/rf_spectrograms.npz")
X = data["X"]
y = data["y"]

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = RFCNN(num_classes=len(np.unique(y))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "results/rf_cnn.pth")
print("Model saved.")
