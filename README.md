# Seismic Earthquake Detection System

A machine learning-based earthquake analysis system with real-time detection, wave classification, and magnitude prediction.

## 🎯 Main Application

**`earthquake_gui.py`** - Primary web interface for earthquake analysis
- Upload seismic files (.mseed, .wav, .ms)
- Automatic P-wave, S-wave, and surface wave detection
- Magnitude prediction from waveforms
- Interactive visualizations
- Downloadable analysis reports

**Usage:** `python3 earthquake_gui.py` then open http://localhost:7860

---

## 📊 Analysis Tools

- **`main.py`** - Command-line earthquake analysis tool
- **`analyze_mseed_traditional.py`** - Traditional MiniSEED file analysis
- **`realtime_web_interface.py`** - Real-time monitoring web interface

---

## 🧠 Training Scripts

- **`train_multiclass_waves.py`** - Train wave type classifier (P/S/Coda/Noise)
- **`train_magnitude_predictor.py`** - Train magnitude prediction models
- **`train_all_models.py`** - Train all models in sequence
- **`train_on_archive.py`** - Train on archived earthquake data

---

## 📁 Key Directories

- **`trained_models_multiclass/`** - Trained wave classification models
- **`magnitude_model_outputs/`** - Trained magnitude prediction models
- **`multi_model_outputs/`** - Multi-model comparison outputs
- **`dataset/`** - Training and test data
- **`archive/`** - Archived earthquake data (merge.hdf5)
- **`gui_test_samples/`** - Test audio samples for GUI testing
- **`venv/`** - Python virtual environment

---

## 📄 Test Data Files

- **`tohoku_2011.mseed`** - 2011 Tōhoku M9.1 earthquake
- **`TEST10.mseed`** - Test earthquake data #10
- **`TEST11.mseed`** - Test earthquake data #11
- **`BK.CMB..BHZ_*.mseed`** - Berkeley station earthquake recording
- **`fdsnws.mseed`** - FDSN web service test data
- **`(Sep 4 2010) M7.2 Darfield Earthquake.wav`** - Darfield earthquake audio

---

## 📖 Documentation

- **`TRAINING_SUMMARY.md`** - Training results and model performance
- **`MULTICLASS_TRAINING_SUMMARY.md`** - Multiclass training details

---

## 🚀 Quick Start

1. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Run the web interface:
   ```bash
   python3 earthquake_gui.py
   ```

3. Open browser to: http://localhost:7860

4. Upload a seismic file and analyze!

---

## 🎓 Model Capabilities

- **Wave Classification**: P-wave, S-wave, Surface/Coda waves, Noise
- **Magnitude Prediction**: M1-M9+ earthquakes
- **Real-time Detection**: Live monitoring support
- **Multiple Formats**: .mseed, .wav, .ms files

---

## 📝 Notes

- Models are pre-trained and ready to use
- Large earthquake detection (M7+) includes automatic magnitude correction
- Supports both local and teleseismic recordings

