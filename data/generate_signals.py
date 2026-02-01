import numpy as np

def awgn(signal, snr_db):
    snr = 10 ** (snr_db / 10)
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / snr
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def bpsk(num_symbols):
    bits = np.random.randint(0, 2, num_symbols)
    return 2 * bits - 1


def qpsk(num_symbols):
    bits = np.random.randint(0, 4, num_symbols)
    mapping = {
        0: 1 + 1j,
        1: -1 + 1j,
        2: -1 - 1j,
        3: 1 - 1j
    }
    symbols = np.array([mapping[b] for b in bits])
    return symbols / np.sqrt(2)


def narrowband_interference(num_samples, freq=0.1):
    t = np.arange(num_samples)
    return np.exp(1j * 2 * np.pi * freq * t)


def generate_sample(num_samples=1024, snr_db=20):
    signal_type = np.random.choice(
        ["bpsk", "qpsk", "noise", "interference"]
    )

    if signal_type == "bpsk":
        sig = bpsk(num_samples)
        label = 0
    elif signal_type == "qpsk":
        sig = qpsk(num_samples)
        label = 1
    elif signal_type == "interference":
        sig = narrowband_interference(num_samples)
        label = 2
    else:
        sig = np.zeros(num_samples)
        label = 3

    sig = awgn(sig, snr_db)
    return sig, label


def generate_dataset(num_samples=2000):
    X = []
    y = []

    for _ in range(num_samples):
        sig, label = generate_sample()
        X.append(sig)
        y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = generate_dataset()
    np.savez("data/rf_dataset.npz", X=X, y=y)
    print("Dataset generated:", X.shape, y.shape)
