import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/rf_spectrograms.npz")
X = data["X"]
y = data["y"]

plt.figure(figsize=(6,4))
plt.imshow(20 * np.log10(X[0] + 1e-6), aspect="auto", origin="lower")
plt.colorbar(label="Magnitude (dB)")
plt.title(f"Spectrogram example | Class {y[0]}")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
