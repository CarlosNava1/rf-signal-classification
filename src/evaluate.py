import numpy as np
import torch
from sklearn.metrics import classification_report
from model import RFCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

data = np.load("data/rf_spectrograms.npz")
X = torch.tensor(data["X"], dtype=torch.float32).unsqueeze(1).to(device)
y = torch.tensor(data["y"]).to(device)

model = RFCNN(num_classes=len(torch.unique(y))).to(device)
model.load_state_dict(torch.load("results/rf_cnn.pth"))
model.eval()

with torch.no_grad():
    preds = model(X).argmax(dim=1)

print(classification_report(y.cpu(), preds.cpu()))
