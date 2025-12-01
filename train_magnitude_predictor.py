#!/usr/bin/env python3
"""
Earthquake Magnitude Prediction System

Trains models to predict earthquake magnitude from seismic waveforms.
Uses regression to predict magnitude values (e.g., 2.5, 4.7, 6.2, etc.)

Two approaches:
1. Direct Magnitude Regression (from P-wave only - early warning)
2. Full Event Magnitude (from complete waveform - most accurate)

Dataset: STEAD (Stanford Earthquake Dataset)
Target: source_magnitude (Richter/Moment magnitude)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime
import json
import ast

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
CSV_PATH = Path("./archive/merge.csv")
HDF5_PATH = Path("./archive/merge.hdf5")
OUTPUT_DIR = Path("./magnitude_model_outputs")
SR = 100  # Hz
N_MFCC = 40
N_FFT = 256
HOP_LENGTH = 128
RANDOM_SEED = 42
VAL_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 60

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)

# Set seed
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("="*80)
print(" "*15 + "EARTHQUAKE MAGNITUDE PREDICTION SYSTEM")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ GPU detected: {gpus[0].name}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print(f"\n⚠ No GPU detected - using CPU")

# ==============================
# DATA LOADING
# ==============================
print("\n" + "="*80)
print("STEP 1: Loading Earthquake Data with Magnitudes")
print("="*80)

print(f"\nLoading metadata from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Total records: {len(df):,}")

# Filter for earthquake events with magnitude data
earthquake_df = df[
    (df['trace_category'] == 'earthquake_local') & 
    (df['source_magnitude'].notna()) &
    (df['p_arrival_sample'].notna())
].copy()

print(f"\nFiltered earthquakes with magnitude data:")
print(f"  Total earthquakes: {len(earthquake_df):,}")
print(f"  Magnitude range: {earthquake_df['source_magnitude'].min():.2f} - {earthquake_df['source_magnitude'].max():.2f}")
print(f"  Mean magnitude: {earthquake_df['source_magnitude'].mean():.2f}")
print(f"  Median magnitude: {earthquake_df['source_magnitude'].median():.2f}")

# Magnitude distribution
mag_bins = [0, 2, 3, 4, 5, 6, 10]
mag_labels = ['0-2 (Minor)', '2-3 (Minor)', '3-4 (Light)', '4-5 (Moderate)', '5-6 (Strong)', '6+ (Major)']
earthquake_df['magnitude_category'] = pd.cut(earthquake_df['source_magnitude'], bins=mag_bins, labels=mag_labels)

print(f"\nMagnitude distribution:")
for cat in mag_labels:
    count = len(earthquake_df[earthquake_df['magnitude_category'] == cat])
    pct = count / len(earthquake_df) * 100
    print(f"  {cat:20s}: {count:6,} ({pct:5.2f}%)")

# ==============================
# HELPER FUNCTIONS
# ==============================

def find_waveform_dataset(h5file):
    """Find the waveform dataset in HDF5 file."""
    if 'data' in h5file:
        return h5file['data']
    raise RuntimeError("Could not find waveform dataset in HDF5 file")

def read_waveform_from_hdf5(h5file, trace_name):
    """Read waveform from HDF5 file."""
    dataset = find_waveform_dataset(h5file)
    if trace_name in dataset:
        waveform = dataset[trace_name][()]
        if waveform.ndim == 2:
            waveform = waveform[:, 2]  # Use vertical component (Z)
        return waveform
    return None

def extract_p_wave_window(waveform, p_arrival_sample, window_samples=400):
    """Extract P-wave window for early warning prediction."""
    if p_arrival_sample is None or np.isnan(p_arrival_sample):
        return None
    
    p_idx = int(p_arrival_sample)
    start_idx = max(0, p_idx - 50)  # Start slightly before P-wave
    end_idx = start_idx + window_samples
    
    if end_idx > len(waveform):
        return None
    
    return waveform[start_idx:end_idx]

def extract_full_event_window(waveform, p_arrival_sample, window_samples=400):
    """Extract full event window including P and S waves."""
    if p_arrival_sample is None or np.isnan(p_arrival_sample):
        return None
    
    p_idx = int(p_arrival_sample)
    start_idx = p_idx
    end_idx = start_idx + window_samples
    
    if end_idx > len(waveform):
        # Pad if needed
        segment = waveform[start_idx:]
        segment = np.pad(segment, (0, window_samples - len(segment)))
        return segment
    
    return waveform[start_idx:end_idx]

def extract_mfcc_features(audio_segment, sr=SR):
    """Extract MFCC features."""
    if len(audio_segment) < 100:
        return None
    
    mfcc = librosa.feature.mfcc(
        y=audio_segment,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    return mfcc

# ==============================
# EXTRACT TRAINING DATA
# ==============================
print("\n" + "="*80)
print("STEP 2: Extracting Waveforms and Features")
print("="*80)

# Sample data for training (use subset for faster training)
SAMPLES_TO_EXTRACT = 15000
print(f"\nSampling {SAMPLES_TO_EXTRACT:,} earthquakes for training...")

# Stratified sampling by magnitude
sample_per_category = SAMPLES_TO_EXTRACT // len(mag_labels)
sampled_df = pd.concat([
    earthquake_df[earthquake_df['magnitude_category'] == cat].sample(
        n=min(sample_per_category, len(earthquake_df[earthquake_df['magnitude_category'] == cat])),
        random_state=RANDOM_SEED
    )
    for cat in mag_labels
])

print(f"Sampled {len(sampled_df):,} earthquakes")

# Extract features
print("\nExtracting P-wave features (for early warning)...")
p_wave_mfccs = []
full_event_mfccs = []
magnitudes = []
trace_names = []

with h5py.File(HDF5_PATH, 'r') as h5file:
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing"):
        trace_name = row['trace_name']
        magnitude = row['source_magnitude']
        p_arrival = row['p_arrival_sample']
        
        # Read waveform
        waveform = read_waveform_from_hdf5(h5file, trace_name)
        if waveform is None:
            continue
        
        # Extract P-wave window (early warning)
        p_window = extract_p_wave_window(waveform, p_arrival, 400)
        if p_window is None:
            continue
        
        # Extract full event window
        full_window = extract_full_event_window(waveform, p_arrival, 400)
        if full_window is None:
            continue
        
        # Extract MFCC features
        p_mfcc = extract_mfcc_features(p_window)
        full_mfcc = extract_mfcc_features(full_window)
        
        if p_mfcc is None or full_mfcc is None:
            continue
        
        p_wave_mfccs.append(p_mfcc)
        full_event_mfccs.append(full_mfcc)
        magnitudes.append(magnitude)
        trace_names.append(trace_name)

print(f"\nExtracted {len(magnitudes):,} samples")

# Pad MFCCs to same length
max_frames_p = max(mfcc.shape[1] for mfcc in p_wave_mfccs)
max_frames_full = max(mfcc.shape[1] for mfcc in full_event_mfccs)

X_p = np.zeros((len(p_wave_mfccs), N_MFCC, max_frames_p, 1), dtype=np.float32)
X_full = np.zeros((len(full_event_mfccs), N_MFCC, max_frames_full, 1), dtype=np.float32)

for i, (p_mfcc, full_mfcc) in enumerate(zip(p_wave_mfccs, full_event_mfccs)):
    X_p[i, :, :p_mfcc.shape[1], 0] = p_mfcc
    X_full[i, :, :full_mfcc.shape[1], 0] = full_mfcc

y = np.array(magnitudes, dtype=np.float32)

print(f"\nFeature shapes:")
print(f"  P-wave features: {X_p.shape}")
print(f"  Full event features: {X_full.shape}")
print(f"  Magnitudes: {y.shape}")

# Split data
X_p_train, X_p_test, X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_p, X_full, y,
    test_size=0.2,
    random_state=RANDOM_SEED
)

print(f"\nData split:")
print(f"  Training samples: {len(X_p_train):,}")
print(f"  Test samples: {len(X_p_test):,}")

# ==============================
# BUILD MODELS
# ==============================
print("\n" + "="*80)
print("STEP 3: Building Magnitude Prediction Models")
print("="*80)

def build_magnitude_predictor(input_shape, model_name="Magnitude_CNN"):
    """Build CNN for magnitude regression."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 1), padding='same'),
        
        layers.Conv2D(64, (3, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 1), padding='same'),
        
        layers.Conv2D(128, (3, 2), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 1), padding='same'),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)  # Single output for magnitude
    ], name=model_name)
    
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model

# Build two models
print("\n1. P-wave Early Warning Model (predicts from P-wave only)")
model_p = build_magnitude_predictor(X_p_train.shape[1:], "P_Wave_Magnitude")
model_p.summary()

print("\n2. Full Event Model (predicts from complete waveform)")
model_full = build_magnitude_predictor(X_full_train.shape[1:], "Full_Event_Magnitude")
model_full.summary()

# ==============================
# TRAIN MODELS
# ==============================
print("\n" + "="*80)
print("STEP 4: Training Models")
print("="*80)

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

# Train P-wave model
print("\nTraining P-wave Early Warning Model...")
checkpoint_p = callbacks.ModelCheckpoint(
    str(OUTPUT_DIR / 'p_wave_magnitude_best.h5'),
    monitor='val_mae',
    save_best_only=True,
    mode='min',
    verbose=1
)

history_p = model_p.fit(
    X_p_train, y_train,
    validation_split=VAL_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr, checkpoint_p],
    verbose=1
)

# Train full event model
print("\nTraining Full Event Model...")
checkpoint_full = callbacks.ModelCheckpoint(
    str(OUTPUT_DIR / 'full_event_magnitude_best.h5'),
    monitor='val_mae',
    save_best_only=True,
    mode='min',
    verbose=1
)

history_full = model_full.fit(
    X_full_train, y_train,
    validation_split=VAL_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr, checkpoint_full],
    verbose=1
)

# ==============================
# EVALUATE MODELS
# ==============================
print("\n" + "="*80)
print("STEP 5: Evaluating Models")
print("="*80)

# P-wave model evaluation
y_pred_p = model_p.predict(X_p_test, verbose=0).flatten()
mae_p = mean_absolute_error(y_test, y_pred_p)
rmse_p = np.sqrt(mean_squared_error(y_test, y_pred_p))
r2_p = r2_score(y_test, y_pred_p)

print("\nP-Wave Early Warning Model:")
print(f"  MAE (Mean Absolute Error): {mae_p:.3f} magnitude units")
print(f"  RMSE: {rmse_p:.3f}")
print(f"  R² Score: {r2_p:.3f}")

# Full event model evaluation
y_pred_full = model_full.predict(X_full_test, verbose=0).flatten()
mae_full = mean_absolute_error(y_test, y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
r2_full = r2_score(y_test, y_pred_full)

print("\nFull Event Model:")
print(f"  MAE (Mean Absolute Error): {mae_full:.3f} magnitude units")
print(f"  RMSE: {rmse_full:.3f}")
print(f"  R² Score: {r2_full:.3f}")

# ==============================
# VISUALIZATIONS
# ==============================
print("\n" + "="*80)
print("STEP 6: Creating Visualizations")
print("="*80)

# 1. Training history
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# P-wave model
axes[0, 0].plot(history_p.history['loss'], label='Train', linewidth=2)
axes[0, 0].plot(history_p.history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].set_title('P-Wave Model - Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history_p.history['mae'], label='Train', linewidth=2)
axes[0, 1].plot(history_p.history['val_mae'], label='Validation', linewidth=2)
axes[0, 1].set_title('P-Wave Model - MAE', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Mean Absolute Error')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Full event model
axes[1, 0].plot(history_full.history['loss'], label='Train', linewidth=2)
axes[1, 0].plot(history_full.history['val_loss'], label='Validation', linewidth=2)
axes[1, 0].set_title('Full Event Model - Loss', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MSE Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history_full.history['mae'], label='Train', linewidth=2)
axes[1, 1].plot(history_full.history['val_mae'], label='Validation', linewidth=2)
axes[1, 1].set_title('Full Event Model - MAE', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Mean Absolute Error')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'training_history.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plots/training_history.png")
plt.close()

# 2. Prediction scatter plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# P-wave model
axes[0].scatter(y_test, y_pred_p, alpha=0.5, s=30)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('True Magnitude', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Magnitude', fontsize=12, fontweight='bold')
axes[0].set_title(f'P-Wave Model\nMAE: {mae_p:.3f}, R²: {r2_p:.3f}', 
                 fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Full event model
axes[1].scatter(y_test, y_pred_full, alpha=0.5, s=30, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('True Magnitude', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Predicted Magnitude', fontsize=12, fontweight='bold')
axes[1].set_title(f'Full Event Model\nMAE: {mae_full:.3f}, R²: {r2_full:.3f}', 
                 fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'prediction_scatter.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plots/prediction_scatter.png")
plt.close()

# 3. Error distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

error_p = y_pred_p - y_test
error_full = y_pred_full - y_test

axes[0].hist(error_p, bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Prediction Error (magnitude units)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title(f'P-Wave Model Error Distribution\nMean Error: {error_p.mean():.3f}', 
                 fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].hist(error_full, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Prediction Error (magnitude units)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title(f'Full Event Model Error Distribution\nMean Error: {error_full.mean():.3f}', 
                 fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'error_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plots/error_distribution.png")
plt.close()

# 4. Magnitude category performance
mag_categories = pd.cut(y_test, bins=mag_bins, labels=mag_labels)
results_df = pd.DataFrame({
    'true_mag': y_test,
    'pred_p': y_pred_p,
    'pred_full': y_pred_full,
    'category': mag_categories
})

category_mae_p = results_df.groupby('category').apply(
    lambda x: mean_absolute_error(x['true_mag'], x['pred_p'])
)
category_mae_full = results_df.groupby('category').apply(
    lambda x: mean_absolute_error(x['true_mag'], x['pred_full'])
)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(mag_labels))
width = 0.35

bars1 = ax.bar(x - width/2, category_mae_p, width, label='P-Wave Model', alpha=0.8)
bars2 = ax.bar(x + width/2, category_mae_full, width, label='Full Event Model', alpha=0.8, color='green')

ax.set_xlabel('Magnitude Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
ax.set_title('Prediction Accuracy by Magnitude Category', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mag_labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'category_performance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: plots/category_performance.png")
plt.close()

# ==============================
# SAVE MODELS AND RESULTS
# ==============================
print("\n" + "="*80)
print("STEP 7: Saving Models and Results")
print("="*80)

# Save models
model_p.save(OUTPUT_DIR / 'p_wave_magnitude_final.h5')
model_full.save(OUTPUT_DIR / 'full_event_magnitude_final.h5')
print("✓ Saved: p_wave_magnitude_final.h5")
print("✓ Saved: full_event_magnitude_final.h5")

# Save results summary
results_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'samples_used': len(magnitudes),
    'train_samples': len(X_p_train),
    'test_samples': len(X_p_test),
    'p_wave_model': {
        'mae': float(mae_p),
        'rmse': float(rmse_p),
        'r2_score': float(r2_p),
        'epochs_trained': len(history_p.history['loss'])
    },
    'full_event_model': {
        'mae': float(mae_full),
        'rmse': float(rmse_full),
        'r2_score': float(r2_full),
        'epochs_trained': len(history_full.history['loss'])
    }
}

with open(OUTPUT_DIR / 'results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print("✓ Saved: results_summary.json")

# Save predictions
predictions_df = pd.DataFrame({
    'true_magnitude': y_test,
    'predicted_p_wave': y_pred_p,
    'predicted_full_event': y_pred_full,
    'error_p_wave': error_p,
    'error_full_event': error_full,
    'magnitude_category': mag_categories
})
predictions_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
print("✓ Saved: predictions.csv")

# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "="*80)
print(" "*20 + "TRAINING COMPLETE! 🎉")
print("="*80)

print(f"\n📊 Model Performance:")
print(f"\nP-Wave Early Warning Model (predicts from P-wave only):")
print(f"  Mean Absolute Error: {mae_p:.3f} magnitude units")
print(f"  RMSE: {rmse_p:.3f}")
print(f"  R² Score: {r2_p:.3f}")
print(f"  Use case: Early warning (3-20s before S-wave)")

print(f"\nFull Event Model (predicts from complete waveform):")
print(f"  Mean Absolute Error: {mae_full:.3f} magnitude units")
print(f"  RMSE: {rmse_full:.3f}")
print(f"  R² Score: {r2_full:.3f}")
print(f"  Use case: Accurate magnitude after full event")

print(f"\n💡 Interpretation:")
print(f"  P-wave model error: ±{mae_p:.1f} magnitude units")
print(f"  Full event model error: ±{mae_full:.1f} magnitude units")

print(f"\n📁 All outputs saved to: {OUTPUT_DIR.absolute()}")
print("="*80 + "\n")
