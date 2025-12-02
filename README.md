# 🌍 SeismicQuake - AI-Powered Earthquake Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyQt6-Desktop-green.svg" alt="PyQt6">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A comprehensive real-time earthquake detection and seismic wave classification system powered by deep learning. This project uses Convolutional Neural Networks (CNNs) with attention mechanisms to detect earthquakes, classify seismic wave types (P, S, Surface), and predict earthquake magnitudes from waveform data.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [AI Models](#-ai-models)
- [Dataset](#-dataset)
- [Technical Specifications](#-technical-specifications)
- [Installation](#-installation)
- [Usage](#-usage)
- [Performance Metrics](#-performance-metrics)
- [File Structure](#-file-structure)
- [API Reference](#-api-reference)

---

## ✨ Features

### Core Capabilities
- **🔍 Earthquake Detection**: Binary classification (earthquake vs. noise) with 96.8% accuracy
- **🌊 Wave Classification**: Identify P-waves, S-waves, and Surface waves with 99.7% accuracy
- **📊 Magnitude Prediction**: Estimate earthquake magnitude from P-wave data (MAE: 0.37)
- **⚡ Real-time Monitoring**: Continuous stream processing with STA/LTA triggering
- **🖥️ Desktop Application**: Full-featured PyQt6 GUI for analysis and visualization

### Supported Formats
- **MiniSEED** (`.mseed`) - Standard seismological format
- **WAV Audio** (`.wav`) - Audio waveform files
- **NumPy Arrays** (`.npy`) - Multi-channel seismic data (E, N, Z components)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SeismicQuake Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Input Data   │───▶│ Preprocessor │───▶│ AI Models Pipeline       │   │
│  │ .mseed/.wav/ │    │              │    │                          │   │
│  │ .npy         │    │ • Normalize  │    │ ┌──────────────────────┐ │   │
│  └──────────────┘    │ • Resample   │    │ │ 1. Earthquake        │ │   │
│                      │ • Extract Z  │    │ │    Detector (Binary) │ │   │
│                      └──────────────┘    │ └──────────┬───────────┘ │   │
│                                          │            │             │   │
│                                          │            ▼             │   │
│  ┌──────────────┐                       │ ┌──────────────────────┐  │   │
│  │ STA/LTA      │◀──────────────────────│ │ 2. Wave Classifier   │  │   │
│  │ Trigger      │                       │ │    (P/S/Surface)     │  │   │
│  │ Algorithm    │                       │ └──────────┬───────────┘  │   │
│  └──────┬───────┘                       │            │              │   │
│         │                               │            ▼              │   │
│         │    ┌──────────────────────────┼──────────────────────┐    │   │
│         └───▶│ Wave Arrival Detection   │ 3. Magnitude         │    │   │
│              │ • P-wave arrival time    │    Predictor         │    │   │
│              │ • S-wave arrival time    │    (from P-wave)     │    │   │
│              │ • Surface wave arrival   │                      │    │   │
│              └──────────────────────────┴──────────────────────┘    │   │
│                                         └───────────┬───────────────┘   │
|                                                     ▼                   |
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                     Output Results                           │       │
│  │  • Earthquake detected: Yes/No                               │       │
│  │  • Wave types: P-wave @ 7.2s, S-wave @ 18.5s, Surface @ 22s  │       │
│  │  • Estimated magnitude: 3.5 ± 0.5                            │       │
│  │  • Confidence scores for each detection                      │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         |
|                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 AI Models

### Model 1: Earthquake Detector (Binary Classifier)

**Purpose**: Distinguish earthquake signals from background noise

**Architecture**: 1D Convolutional Neural Network (CNN)

```
Input Layer         : (400, 1) - 4 seconds @ 100Hz
├── Conv1D Block 1  : 32 filters, kernel=7, MaxPool(2)
├── Conv1D Block 2  : 64 filters, kernel=5, MaxPool(2)
├── Conv1D Block 3  : 128 filters, kernel=3, MaxPool(2)
├── Conv1D Block 4  : 256 filters, kernel=3, MaxPool(2)
├── GlobalAvgPool1D
├── Dropout(0.3)
├── Dense(128, ReLU)
├── Dropout(0.3)
├── Dense(64, ReLU)
└── Dense(1, Sigmoid) → Earthquake Probability
```

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.81%** |
| Test AUC-ROC | **99.59%** |
| Test Loss | 0.0795 |
| Model Size | 2.12 MB |

---

### Model 2: Wave Type Classifier

**Purpose**: Classify seismic waves into P-wave, S-wave, or Surface wave

**Architecture**: 1D CNN with Attention Mechanism

```
Input Layer         : (400, 1) - 4 seconds @ 100Hz
├── Conv1D(32, k=7) + BatchNorm + MaxPool(2)
├── Conv1D(64, k=5) + BatchNorm + MaxPool(2)
├── Conv1D(128, k=3) + BatchNorm + MaxPool(2)
├── ┌─ Attention Layer ─┐
│   │ Conv1D(1, k=1, sigmoid) → Attention Weights
│   │ Element-wise multiply with features
│   └────────────────────┘
├── Conv1D(256, k=3) + BatchNorm
├── GlobalAvgPool1D
├── Dropout(0.4)
├── Dense(128, ReLU)
├── Dropout(0.3)
├── Dense(64, ReLU)
└── Dense(3, Softmax) → [P-wave, S-wave, Surface-wave]
```

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.69%** |
| Test Loss | 0.0088 |
| Model Size | 2.12 MB |
| Classes | P-wave, S-wave, Surface-wave |

---

### Model 3: Magnitude Predictor

**Purpose**: Estimate earthquake magnitude from P-wave segment

**Architecture**: CNN + Bidirectional LSTM (Hybrid)

```
Input Layer         : (400, 1) - P-wave segment
├── Conv1D(32, k=7) + BatchNorm + MaxPool(2)
├── Conv1D(64, k=5) + BatchNorm + MaxPool(2)
├── Conv1D(128, k=3) + BatchNorm + MaxPool(2)
├── Conv1D(256, k=3) + BatchNorm
├── Bidirectional LSTM(64) ← Captures temporal dependencies
├── Dropout(0.4)
├── Dense(128, ReLU)
├── Dropout(0.3)
├── Dense(64, ReLU)
├── Dense(32, ReLU)
└── Dense(1, Linear) → Magnitude value
```

| Metric | Value |
|--------|-------|
| Test MAE | **0.374** |
| Test MSE | 0.279 |
| Within ±0.5 | 75.03% |
| Within ±1.0 | **93.07%** |
| Model Size | 3.85 MB |

---

## 📊 Dataset

### Source
**STEAD (Stanford Earthquake Dataset)** - One of the largest publicly available seismic datasets

### Statistics

| Data Type | Samples | Description |
|-----------|---------|-------------|
| P-waves | 589,792 | Primary wave arrivals |
| S-waves | 589,792 | Secondary wave arrivals |
| Surface waves | 375,541 | Surface/Coda waves |
| Noise | 235,426 | Background seismic noise |
| **Total** | **1,790,551** | Labeled waveform segments |

### Archive Details
- **Format**: HDF5 (merge.hdf5) + CSV metadata (merge.csv)
- **Size**: 91.09 GB
- **Records**: 1,265,657 seismic traces
- **Waveform Shape**: (6000, 3) - 60 seconds × 3 channels (E, N, Z)
- **Sample Rate**: 100 Hz
- **Magnitude Range**: 0.0 - 7.9

### Data Preprocessing
1. **Channel Selection**: Z-component (vertical) extracted for analysis
2. **Normalization**: Waveforms scaled to [-1, 1] range
3. **Windowing**: 400 samples (4 seconds) per analysis window
4. **Padding/Truncation**: Segments standardized to INPUT_LENGTH

---

## ⚙️ Technical Specifications

### Signal Processing

#### STA/LTA Algorithm (Wave Arrival Detection)
```python
STA/LTA = Short-Term Average / Long-Term Average

Parameters:
- STA window: 30 samples (0.3 seconds)
- LTA window: 150 samples (1.5 seconds)
- P-wave trigger threshold: 3.0
- Detection cooldown: 2.0 seconds
```

#### Wave Arrival Logic
- **P-wave**: First STA/LTA trigger above threshold
- **S-wave**: Amplitude increase (>1.5×) after P-wave arrival
- **Surface wave**: 2+ seconds after S-wave arrival

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 256 |
| Epochs | 50 (with early stopping) |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss (Binary) | Binary Cross-Entropy |
| Loss (Multiclass) | Categorical Cross-Entropy |
| Loss (Regression) | Mean Squared Error |
| Validation Split | 15% |
| Test Split | 10% |
| Max Samples/Class | 100,000 |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16+ GB |
| GPU | - | NVIDIA RTX 3050+ |
| Storage | 100 GB | 200+ GB (for full dataset) |
| Python | 3.10+ | 3.12 |

---

## 🚀 Installation

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install libxcb-cursor0 python3-pip python3-venv

# Clone repository
git clone https://github.com/JustineBijuPaul/seismicquake.git
cd seismicquake
```

### Setup Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install tensorflow numpy pandas h5py obspy scipy matplotlib PyQt6 tqdm
```

### Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "from seismic_analyzer import SeismicAnalyzer; print('SeismicAnalyzer OK')"
```

---

## 📖 Usage

### Command Line Interface

```bash
# Analyze a single file
python seismic_analyzer.py analyze earthquake.mseed

# Analyze with visualization
python seismic_analyzer.py analyze recording.wav --visualize --output result.png

# Real-time simulation from file
python seismic_analyzer.py monitor --file seismic_data.npy

# Batch process multiple files
python seismic_analyzer.py batch *.mseed --output results.json

# Run demo
python seismic_analyzer.py demo
```

### Desktop Application

```bash
# Launch GUI application
python earthquake_desktop_app.py
```

**Features:**
- 📊 File Analysis Tab: Load and analyze .mseed/.wav/.npy files
- 📡 Real-time Monitor: Live waveform display with earthquake alerts
- 📋 Results Tab: History table with export options (JSON/CSV)
- ⚙️ Settings: Adjust detection threshold, window size, alerts

### Python API

```python
from seismic_analyzer import SeismicAnalyzer

# Initialize analyzer
analyzer = SeismicAnalyzer(verbose=True)

# Analyze a file
result = analyzer.analyze_file("earthquake.mseed")

# Access results
print(f"Earthquake: {result.is_earthquake}")
print(f"Confidence: {result.earthquake_confidence:.1%}")
print(f"P-wave arrival: {result.p_wave_arrival}s")
print(f"S-wave arrival: {result.s_wave_arrival}s")
print(f"Magnitude: {result.estimated_magnitude}")

# Process individual segments
data, sr = analyzer.load_file("data.npy")
is_eq, confidence = analyzer.detect_earthquake(data[:400])
wave_type, wave_conf = analyzer.classify_wave(data[:400])
magnitude, uncertainty = analyzer.predict_magnitude(data[:400])
```

---

## 📈 Performance Metrics

### Model Accuracy Summary

| Model | Accuracy/MAE | AUC | Precision | Recall |
|-------|--------------|-----|-----------|--------|
| Earthquake Detector | 96.81% | 99.59% | ~97% | ~97% |
| Wave Classifier | 99.69% | - | ~99% | ~99% |
| Magnitude Predictor | 0.374 MAE | - | 75% ±0.5 | 93% ±1.0 |

### P-wave Arrival Detection Accuracy

| Sample | True P (s) | Detected P (s) | Error |
|--------|------------|----------------|-------|
| 1 | 7.0 | 7.2 | +0.2s |
| 2 | 6.0 | 6.1 | +0.1s |
| 3 | 5.0 | 5.0 | 0.0s |
| 4 | 9.0 | 9.1 | +0.1s |
| 5 | 7.0 | 7.1 | +0.1s |

**Average P-wave timing error: ~0.1 seconds**

---

## 📁 File Structure

```
seismicquake/
├── archive/                          # Raw seismic data archive
│   ├── merge.hdf5                   # 91 GB HDF5 waveform database
│   └── merge.csv                    # Metadata with arrival times
├── earthquake_ai_models/            # Trained AI models
│   ├── earthquake_detector_best.h5  # Binary classifier (2.12 MB)
│   ├── wave_classifier_best.h5      # Wave type classifier (2.12 MB)
│   ├── magnitude_predictor_best.h5  # Magnitude regressor (3.85 MB)
│   └── training_summary.json        # Training metrics
├── extracted_waves/                 # Preprocessed training data
│   ├── p_wave/                      # 589,792 P-wave samples
│   ├── s_wave/                      # 589,792 S-wave samples
│   ├── surface_wave/                # 375,541 Surface wave samples
│   └── noise/                       # 235,426 Noise samples
├── extracted_audio_samples/         # Test samples (5,000 files)
├── seismic_analyzer.py             # Core analysis engine
├── earthquake_desktop_app.py       # PyQt6 desktop application
├── train_earthquake_ai.py          # Model training script
├── main.py                         # Legacy entry point
└── README.md                       # This documentation
```

---

## 🔧 API Reference

### SeismicAnalyzer Class

```python
class SeismicAnalyzer:
    """Main class for seismic wave detection and classification."""
    
    # Constants
    EARTHQUAKE_THRESHOLD = 0.5      # Detection threshold
    WAVE_TYPES = ['P', 'S', 'Surface']
    
    # Methods
    def load_file(filepath) -> Tuple[np.ndarray, float]
    def detect_earthquake(segment) -> Tuple[bool, float]
    def classify_wave(segment) -> Tuple[str, float]
    def predict_magnitude(segment) -> Tuple[float, float]
    def analyze_trace(data, sample_rate) -> List[WaveDetection]
    def analyze_file(filepath) -> AnalysisResult
    def visualize(filepath, output_path, show) -> Figure
```

### Data Classes

```python
@dataclass
class WaveDetection:
    wave_type: str          # 'P', 'S', 'Surface', 'Noise'
    confidence: float       # 0.0 - 1.0
    start_sample: int
    end_sample: int
    start_time: float       # seconds
    end_time: float
    magnitude: Optional[float]

@dataclass
class AnalysisResult:
    filename: str
    duration_seconds: float
    sample_rate: float
    is_earthquake: bool
    earthquake_confidence: float
    detections: List[WaveDetection]
    p_wave_arrival: Optional[float]
    s_wave_arrival: Optional[float]
    surface_wave_arrival: Optional[float]
    estimated_magnitude: Optional[float]
    processing_time: float
```

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Data Processing** | NumPy, Pandas, H5Py |
| **Seismology** | ObsPy (MiniSEED parsing) |
| **Signal Processing** | SciPy, STA/LTA algorithm |
| **Visualization** | Matplotlib |
| **Desktop GUI** | PyQt6 |
| **GPU Acceleration** | CUDA, cuDNN |

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **STEAD Dataset**: Stanford Earthquake Dataset for training data
- **ObsPy**: Seismological data processing
- **TensorFlow Team**: Deep learning framework

---

## 📧 Contact

**Author**: Justine Biju Paul  
**Repository**: [github.com/JustineBijuPaul/seismicquake](https://github.com/JustineBijuPaul/seismicquake)

---

<p align="center">
  Made with ❤️ for earthquake early warning systems
</p>
