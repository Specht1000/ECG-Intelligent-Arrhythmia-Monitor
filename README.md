# 🫀 ECG Intelligent Arrhythmia Monitor

Low-cost intelligent real-time ECG monitoring system using embedded systems, digital signal processing, and artificial intelligence.

This project combines ECG acquisition hardware, real-time embedded signal processing, and AI-assisted cardiac rhythm classification to create a portable and affordable arrhythmia monitoring platform.

---

# 📌 Project Overview

The system acquires ECG signals using an AD8232 analog front-end connected to an ESP32-S3 microcontroller. The acquired signal is digitally processed using a Pan–Tompkins-inspired pipeline for R-peak detection and BPM estimation.

A Python-based AI monitor performs real-time visualization and convolutional neural network (CNN) inference for cardiac rhythm classification.

The project was developed as part of an embedded systems and biomedical engineering research initiative involving:

- Polytech Montpellier
- University of Montpellier
- PUCRS

---

# 🚀 Main Features

## Embedded ECG Acquisition
- Real-time ECG acquisition
- Low-cost hardware architecture
- Serial streaming
- OLED support

## Digital Signal Processing
- ECG smoothing
- Derivative filtering
- Squaring stage
- Moving Window Integration (MWI)
- Adaptive threshold
- R-peak detection
- BPM estimation
- RR interval analysis

## Artificial Intelligence
- CNN-based ECG classification
- MIT-BIH Arrhythmia Database support
- Real-time inference
- Signal quality estimation
- AI-assisted monitoring

## Real-Time Monitoring Interface
- Live ECG waveform visualization
- BPM display
- RR interval monitoring
- Signal quality indicator
- CNN prediction display
- Medical terminal visualization

---

# 🧠 AI Classification

The AI system uses a 1D Convolutional Neural Network trained on ECG windows extracted from:

- MIT-BIH Arrhythmia Database
- Synthetic ECG datasets
- Real ECG signals acquired from hardware

The CNN performs:
- heartbeat classification
- rhythm analysis
- confidence estimation

---

# 🏗️ System Architecture

```text
AD8232 ECG Sensor
        ↓
ADS1115 16-bit ADC
        ↓
ESP32-S3
        ↓
Serial Communication
        ↓
Python Realtime Monitor
        ↓
CNN Classification
```

---

# ⚙️ Hardware

## Main Components

| Component | Description |
|---|---|
| ESP32-S3 DevKitC-1 | Main embedded processor |
| AD8232 | Analog ECG acquisition front-end |
| ADS1115 | 16-bit external ADC |
| SH1106 OLED | Real-time embedded display |
| ECG electrodes | Biomedical signal acquisition |

---

# 💻 Software Stack

## Embedded Side
- Arduino Framework
- ESP-IDF compatible
- C++
- FreeRTOS concepts

## Desktop Side
- Python
- TensorFlow / Keras
- NumPy
- SciPy
- Matplotlib
- Scikit-learn
- PySerial
- WFDB

---

# 📈 Signal Processing Pipeline

The ECG processing pipeline is inspired by the Pan–Tompkins algorithm.

```text
Raw ECG
   ↓
Derivative
   ↓
Squaring
   ↓
Moving Window Integration
   ↓
Adaptive Threshold
   ↓
R-Peak Detection
   ↓
BPM / RR Calculation
```

---

# 🧪 Dataset

## MIT-BIH Arrhythmia Database

Used for:
- CNN training
- ECG segmentation
- heartbeat classification
- validation

Database:
- PhysioNet MIT-BIH Arrhythmia Database

---

# 📊 Real-Time AI Monitor

The realtime AI monitor provides:

- ECG waveform plotting
- BPM monitoring
- RR interval calculation
- CNN predictions
- Confidence estimation
- Signal quality analysis

Two monitor modes were developed:

## Doctor Monitor
Medical-oriented terminal visualization.

## Realtime Honest Monitor
AI-assisted ECG monitor with CNN inference.

---

# 🧬 Supported Rhythm Types

Current project support includes:

- Normal rhythm
- Bradycardia
- Tachycardia
- Generic abnormal rhythms
- Experimental AF-like behavior detection

---

# 📂 Project Structure

```text
pfe_ecg/
│
├── src/                  # Embedded firmware
├── include/              # Headers
├── tools/                # Python analysis tools
├── models/               # Trained CNN models
├── data/                 # ECG datasets
├── report_figures/       # Figures for article/report
├── reports/              # LaTeX report
├── README.md
└── platformio.ini
```

---

# 🔌 Serial Communication

The ESP32 streams ECG information through serial communication:

```text
[ECG] RAW=20368 | FILT=-3.5 | INT=3.5 | TH=64986.3
[RPEAK] RR=0.830 | BPM_INST=72.3 | STATUS=NORMAL
```

---

# ▶️ Running the Project

# 1. Clone repository

```bash
git clone https://github.com/Specht1000/ECG-Intelligent-Arrhythmia-Monitor.git
```

# 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

# 3. Upload firmware

Using PlatformIO:

```bash
pio run -t upload
```

# 4. Start realtime AI monitor

```bash
python tools/realtime_honest_monitor.py
```

---

# 📷 Example Results

The system provides:
- real-time ECG plotting
- BPM estimation
- AI classification
- signal quality analysis

Example screenshots and CNN analysis figures are available in:

```text
report_figures/
```

---

# 🔬 Research Context

This project explores the intersection between:

- Embedded Systems
- Biomedical Engineering
- Artificial Intelligence
- Digital Signal Processing
- Low-Cost Medical Devices

---

# ⚠️ Disclaimer

This project is intended for:
- educational purposes
- research
- prototyping

It is NOT a certified medical device and must not be used for clinical diagnosis.

---

# 👨‍💻 Author

## Guilherme Martins Specht

- Computer Engineering
- Embedded Systems
- Biomedical Signal Processing
- Artificial Intelligence

### Institutions
- PUCRS
- Polytech Montpellier
- University of Montpellier

---

# 📚 References

- MIT-BIH Arrhythmia Database
- Pan–Tompkins Algorithm
- PhysioNet
- TensorFlow
- ESP32 Documentation
- AD8232 Datasheet

---

# 📜 License

This project is open-source for academic and research purposes.
