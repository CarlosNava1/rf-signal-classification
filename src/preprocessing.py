import numpy as np
from scipy.signal import stft

def compute_spectrogram(signal, n_fft=256, hop_length=128):
    """
    Compute magnitude spectrogram using STFT
    """
    _, _, Zxx = stft(
        signal,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        return_onesided=False
    )
    return np.abs(Zxx)


def preprocess_dataset(input_path="data/rf_dataset.npz",
                       output_path="data/rf_spectrograms.npz"):
    data = np.load(input_path)
    X = data["X"]
    y = data["y"]

    specs = []
    for sig in X:
        spec = compute_spectrogram(sig)
        specs.append(spec)

    specs = np.array(specs)

    np.savez(output_path, X=specs, y=y)
    print("Spectrogram dataset created:", specs.shape, y.shape)


if __name__ == "__main__":
    preprocess_dataset()
