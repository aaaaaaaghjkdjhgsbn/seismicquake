#!/usr/bin/env python3
"""
Earthquake Detection and Classification AI Training

This script trains three models:
1. Binary Classifier: Earthquake vs Noise detection
2. Wave Type Classifier: P-wave, S-wave, Surface wave classification
3. Magnitude Predictor: Estimate earthquake magnitude from P-wave

For real-time earthquake detection and early warning systems.
"""

import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
EXTRACTED_WAVES_DIR = Path("./extracted_waves")
ARCHIVE_DIR = Path("./archive")
OUTPUT_DIR = Path("./earthquake_ai_models")
OUTPUT_DIR.mkdir(exist_ok=True)

# Training parameters
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.10

# Fixed input length for all models (pad/truncate to this)
INPUT_LENGTH = 400
SAMPLE_RATE = 100  # Hz

# Sample limits for faster training (set to None for full dataset)
MAX_SAMPLES_PER_CLASS = 100000  # Use 100k samples per class for balanced training


def set_gpu_memory_growth():
    """Configure GPU memory growth to avoid OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU configured: {len(gpus)} device(s) available")
        except RuntimeError as e:
            print(f"GPU config error: {e}")
    else:
        print("⚠ No GPU found, using CPU")


def normalize_waveform(waveform):
    """Normalize waveform to [-1, 1] range."""
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform


def pad_or_truncate(waveform, target_length=INPUT_LENGTH):
    """Pad or truncate waveform to target length."""
    if len(waveform) >= target_length:
        return waveform[:target_length]
    else:
        # Pad with zeros
        padded = np.zeros(target_length, dtype=np.float32)
        padded[:len(waveform)] = waveform
        return padded


def load_wave_files(wave_type, max_samples=None):
    """Load waveform files from a directory."""
    wave_dir = EXTRACTED_WAVES_DIR / wave_type
    files = sorted(wave_dir.glob("*.npy"))
    
    if max_samples:
        # Random sample for balanced training
        np.random.seed(42)
        if len(files) > max_samples:
            indices = np.random.choice(len(files), max_samples, replace=False)
            files = [files[i] for i in indices]
    
    print(f"  Loading {len(files):,} {wave_type} samples...")
    
    waveforms = []
    valid_indices = []
    
    for i, f in enumerate(files):
        try:
            waveform = np.load(f)
            waveform = pad_or_truncate(waveform)
            waveform = normalize_waveform(waveform)
            waveforms.append(waveform)
            
            # Extract index from filename for magnitude lookup
            match = re.search(r'(\d+)', f.stem)
            if match:
                valid_indices.append(int(match.group(1)))
            else:
                valid_indices.append(-1)
                
        except Exception as e:
            continue
    
    return np.array(waveforms, dtype=np.float32), valid_indices


def load_magnitudes_for_indices(indices):
    """Load magnitude values from CSV for given indices."""
    df = pd.read_csv(ARCHIVE_DIR / "merge.csv", low_memory=False)
    
    magnitudes = []
    for idx in indices:
        if idx >= 0 and idx < len(df):
            mag = df.iloc[idx]['source_magnitude']
            if pd.notna(mag):
                magnitudes.append(float(mag))
            else:
                magnitudes.append(0.0)
        else:
            magnitudes.append(0.0)
    
    return np.array(magnitudes, dtype=np.float32)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def create_1d_cnn_block(x, filters, kernel_size=3, pool_size=2):
    """Create a 1D CNN block with Conv, BatchNorm, Activation, and Pooling."""
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    return x


def create_binary_classifier():
    """
    Model 1: Binary Classifier (Earthquake vs Noise)
    
    Input: Waveform segment
    Output: Probability of earthquake (0 = noise, 1 = earthquake)
    """
    inputs = layers.Input(shape=(INPUT_LENGTH, 1), name='waveform_input')
    
    # CNN feature extraction
    x = create_1d_cnn_block(inputs, 32, kernel_size=7)
    x = create_1d_cnn_block(x, 64, kernel_size=5)
    x = create_1d_cnn_block(x, 128, kernel_size=3)
    x = create_1d_cnn_block(x, 256, kernel_size=3)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid', name='earthquake_prob')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='earthquake_detector')
    return model


def create_wave_classifier():
    """
    Model 2: Wave Type Classifier (P-wave, S-wave, Surface wave)
    
    Input: Waveform segment (already detected as earthquake)
    Output: Probabilities for each wave type
    """
    inputs = layers.Input(shape=(INPUT_LENGTH, 1), name='waveform_input')
    
    # CNN feature extraction with attention
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Attention mechanism
    attention = layers.Conv1D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Output: 3 classes (P, S, Surface)
    outputs = layers.Dense(3, activation='softmax', name='wave_type')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='wave_classifier')
    return model


def create_magnitude_predictor():
    """
    Model 3: Magnitude Predictor (from P-wave)
    
    Input: P-wave segment
    Output: Estimated magnitude (regression)
    """
    inputs = layers.Input(shape=(INPUT_LENGTH, 1), name='p_wave_input')
    
    # CNN feature extraction
    x = layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Bidirectional LSTM for temporal patterns
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    
    # Dense layers
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output: single magnitude value
    outputs = layers.Dense(1, activation='linear', name='magnitude')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='magnitude_predictor')
    return model


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_binary_classifier():
    """Train the earthquake vs noise binary classifier."""
    print("\n" + "="*70)
    print("TRAINING MODEL 1: EARTHQUAKE DETECTOR (Binary Classifier)")
    print("="*70)
    
    # Load earthquake samples (combine P, S, Surface waves)
    print("\nLoading earthquake samples...")
    p_waves, _ = load_wave_files('p_wave', MAX_SAMPLES_PER_CLASS // 3)
    s_waves, _ = load_wave_files('s_wave', MAX_SAMPLES_PER_CLASS // 3)
    surface_waves, _ = load_wave_files('surface_wave', MAX_SAMPLES_PER_CLASS // 3)
    
    earthquake_samples = np.concatenate([p_waves, s_waves, surface_waves])
    earthquake_labels = np.ones(len(earthquake_samples))
    
    # Load noise samples
    print("\nLoading noise samples...")
    noise_samples, _ = load_wave_files('noise', len(earthquake_samples))
    noise_labels = np.zeros(len(noise_samples))
    
    # Combine data
    X = np.concatenate([earthquake_samples, noise_samples])
    y = np.concatenate([earthquake_labels, noise_labels])
    
    # Reshape for CNN (add channel dimension)
    X = X.reshape(-1, INPUT_LENGTH, 1)
    
    print(f"\nTotal samples: {len(X):,}")
    print(f"  Earthquake: {int(y.sum()):,}")
    print(f"  Noise: {int(len(y) - y.sum()):,}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=VALIDATION_SPLIT + TEST_SPLIT, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SPLIT/(VALIDATION_SPLIT + TEST_SPLIT), 
        random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Create model
    model = create_binary_classifier()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            OUTPUT_DIR / 'earthquake_detector_best.h5',
            monitor='val_auc', mode='max', save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "-"*50)
    print("Test Set Evaluation:")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.4f}")
    print(f"  AUC: {test_results[2]:.4f}")
    
    # Save final model
    model.save(OUTPUT_DIR / 'earthquake_detector_final.h5')
    
    return model, history, {'test_loss': test_results[0], 'test_acc': test_results[1], 'test_auc': test_results[2]}


def train_wave_classifier():
    """Train the wave type classifier (P, S, Surface)."""
    print("\n" + "="*70)
    print("TRAINING MODEL 2: WAVE TYPE CLASSIFIER")
    print("="*70)
    
    # Load samples for each wave type
    print("\nLoading wave samples...")
    p_waves, _ = load_wave_files('p_wave', MAX_SAMPLES_PER_CLASS)
    s_waves, _ = load_wave_files('s_wave', MAX_SAMPLES_PER_CLASS)
    surface_waves, _ = load_wave_files('surface_wave', MAX_SAMPLES_PER_CLASS)
    
    # Create labels: 0=P-wave, 1=S-wave, 2=Surface wave
    X = np.concatenate([p_waves, s_waves, surface_waves])
    y = np.concatenate([
        np.zeros(len(p_waves)),
        np.ones(len(s_waves)),
        np.full(len(surface_waves), 2)
    ])
    
    # Reshape for CNN
    X = X.reshape(-1, INPUT_LENGTH, 1)
    
    print(f"\nTotal samples: {len(X):,}")
    print(f"  P-waves: {len(p_waves):,}")
    print(f"  S-waves: {len(s_waves):,}")
    print(f"  Surface waves: {len(surface_waves):,}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=VALIDATION_SPLIT + TEST_SPLIT, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SPLIT/(VALIDATION_SPLIT + TEST_SPLIT), 
        random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, 3)
    y_val_cat = keras.utils.to_categorical(y_val, 3)
    y_test_cat = keras.utils.to_categorical(y_test, 3)
    
    # Create model
    model = create_wave_classifier()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            OUTPUT_DIR / 'wave_classifier_best.h5',
            monitor='val_accuracy', mode='max', save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "-"*50)
    print("Test Set Evaluation:")
    test_results = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  Loss: {test_results[0]:.4f}")
    print(f"  Accuracy: {test_results[1]:.4f}")
    
    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    wave_names = ['P-wave', 'S-wave', 'Surface']
    for i, name in enumerate(wave_names):
        mask = y_test == i
        acc = (y_pred_classes[mask] == i).mean()
        print(f"  {name} Accuracy: {acc:.4f}")
    
    # Save final model
    model.save(OUTPUT_DIR / 'wave_classifier_final.h5')
    
    return model, history, {'test_loss': test_results[0], 'test_acc': test_results[1]}


def train_magnitude_predictor():
    """Train the magnitude predictor from P-wave."""
    print("\n" + "="*70)
    print("TRAINING MODEL 3: MAGNITUDE PREDICTOR")
    print("="*70)
    
    # Load P-wave samples with their indices
    print("\nLoading P-wave samples...")
    p_waves, indices = load_wave_files('p_wave', MAX_SAMPLES_PER_CLASS * 2)  # Use more samples for regression
    
    # Load corresponding magnitudes
    print("Loading magnitude labels...")
    magnitudes = load_magnitudes_for_indices(indices)
    
    # Filter out invalid magnitudes
    valid_mask = (magnitudes > 0) & (magnitudes < 10)
    X = p_waves[valid_mask]
    y = magnitudes[valid_mask]
    
    # Reshape for CNN
    X = X.reshape(-1, INPUT_LENGTH, 1)
    
    print(f"\nTotal valid samples: {len(X):,}")
    print(f"Magnitude range: {y.min():.2f} - {y.max():.2f}")
    print(f"Magnitude mean: {y.mean():.2f} ± {y.std():.2f}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=VALIDATION_SPLIT + TEST_SPLIT, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SPLIT/(VALIDATION_SPLIT + TEST_SPLIT), random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    
    # Create model
    model = create_magnitude_predictor()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            OUTPUT_DIR / 'magnitude_predictor_best.h5',
            monitor='val_mae', mode='min', save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor='val_mae', mode='min', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "-"*50)
    print("Test Set Evaluation:")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  MSE: {test_results[0]:.4f}")
    print(f"  MAE: {test_results[1]:.4f}")
    print(f"  RMSE: {np.sqrt(test_results[0]):.4f}")
    
    # Additional metrics
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Within 0.5 magnitude units
    within_05 = np.mean(np.abs(y_pred - y_test) <= 0.5) * 100
    within_1 = np.mean(np.abs(y_pred - y_test) <= 1.0) * 100
    print(f"  Within ±0.5 magnitude: {within_05:.1f}%")
    print(f"  Within ±1.0 magnitude: {within_1:.1f}%")
    
    # Save final model
    model.save(OUTPUT_DIR / 'magnitude_predictor_final.h5')
    
    return model, history, {
        'test_mse': test_results[0], 
        'test_mae': test_results[1],
        'within_05': within_05,
        'within_1': within_1
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    print("="*70)
    print("EARTHQUAKE DETECTION AND CLASSIFICATION AI TRAINING")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure GPU
    set_gpu_memory_growth()
    
    # Store results
    results = {}
    
    # Train all three models
    try:
        # Model 1: Binary Classifier
        eq_model, eq_history, eq_results = train_binary_classifier()
        results['earthquake_detector'] = eq_results
        
        # Model 2: Wave Type Classifier  
        wave_model, wave_history, wave_results = train_wave_classifier()
        results['wave_classifier'] = wave_results
        
        # Model 3: Magnitude Predictor
        mag_model, mag_history, mag_results = train_magnitude_predictor()
        results['magnitude_predictor'] = mag_results
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save training summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'input_length': INPUT_LENGTH,
            'max_samples_per_class': MAX_SAMPLES_PER_CLASS
        },
        'results': results
    }
    
    with open(OUTPUT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("\nTrained models:")
    print("  1. earthquake_detector_best.h5   - Earthquake vs Noise detection")
    print("  2. wave_classifier_best.h5       - P/S/Surface wave classification")
    print("  3. magnitude_predictor_best.h5   - Magnitude estimation from P-wave")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    main()
