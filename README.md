\# RF Signal Classification with Deep Learning



End-to-end RF signal classification pipeline combining classical signal

processing and deep learning. The project generates synthetic RF signals,

extracts time–frequency representations, and trains a CNN to classify

different signal types based on their spectrograms.



This project bridges \*\*telecommunications engineering\*\* and \*\*applied deep

learning\*\*, following a realistic workflow used in RF sensing and wireless

intelligence systems.



---



\## 📡 Problem Overview



Automatic classification of RF signals is a key problem in:

\- Wireless monitoring

\- Spectrum sensing

\- Cognitive radio

\- Interference detection



Unlike vision tasks, RF signals often exhibit overlapping spectral

characteristics, making classification challenging even with deep learning

models.



---



\## 🧠 Methodology



\### Signal Generation

\- Synthetic RF signals generated in baseband

\- Multiple signal classes with different characteristics

\- Controlled dataset for reproducibility



\### Feature Extraction

\- Short-Time Fourier Transform (STFT)

\- Magnitude spectrograms used as input features

\- Time–frequency representation captures spectral dynamics



\### Deep Learning Model

\- 2D Convolutional Neural Network (CNN)

\- Input: spectrograms

\- Loss: Cross-Entropy

\- Optimizer: Adam



\### Evaluation

\- Accuracy, precision, recall, F1-score

\- Confusion matrix analysis

\- Interpretation focused on RF-specific challenges



---



\## 📊 Results



\- Overall accuracy: \*\*~76%\*\*

\- Certain RF classes are perfectly classified

\- One class presents confusion due to overlapping spectral features



This behavior reflects real-world RF conditions, where some signal classes

are intrinsically difficult to separate using time–frequency features alone.



---



\## 📁 Project Structure



rf-signal-classification/

│

├── data/

│ ├── generate\_signals.py

│ ├── rf\_signals.npy

│ ├── rf\_labels.npy

│ └── rf\_spectrograms.npz

│

├── src/

│ ├── preprocessing.py

│ ├── model.py

│ ├── train.py

│ └── evaluate.py

│

├── results/

│ └── rf\_cnn.pth

│

├── requirements.txt

└── README.md





---



\## ▶️ How to Run



```bash

\# Generate RF signals

py -3.9 data/generate\_signals.py



\# Create spectrogram dataset

py -3.9 src/preprocessing.py



\# Train CNN

py -3.9 src/train.py



\# Evaluate model

py -3.9 src/evaluate.py

🛠 Technologies Used

Python



NumPy / SciPy



PyTorch



scikit-learn



Signal Processing (STFT)



Deep Learning (CNNs)



🎓 Author

Carlos Navarro

M.Sc. Artificial Intelligence — Universidad Politécnica de Madrid

Telecommunications Engineer





