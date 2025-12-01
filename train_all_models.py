#!/usr/bin/env python3
"""
Multi-Model Training and Comparison for Seismic Wave Classification
Tests multiple architectures:
- CNN (2D Convolutional)
- CNN with Attention
- ResNet-style CNN
- LSTM (Recurrent)
- Bidirectional LSTM
- GRU (Gated Recurrent Unit)
- CNN-LSTM Hybrid
- 1D CNN
- Dense Neural Network
- Ensemble Model

Trains on: wave_audio_dataset/ (20,000 samples)
Tests on: test_seismic_samples/ (1,000 samples)
Saves all outputs to: multi_model_outputs/
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import time

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils, Model
from tensorflow.keras.layers import (
    Input, Conv1D, Conv2D, MaxPool1D, MaxPool2D, GlobalMaxPooling1D, GlobalAveragePooling1D,
    LSTM, GRU, Bidirectional, Dense, Dropout, Flatten, BatchNormalization,
    Concatenate, Reshape, Permute, Multiply, Add, Activation, TimeDistributed
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
TRAIN_DIR = Path("./wave_audio_dataset")
TEST_DIR = Path("./test_seismic_samples")
OUTPUT_DIR = Path("./multi_model_outputs")
CLASS_NAMES = ["noise", "p_wave", "s_wave", "coda"]
SR = 100  # sampling rate in Hz
N_MFCC = 40
N_FFT = 256
HOP_LENGTH = 128
RANDOM_SEED = 42
VAL_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 50  # Reduced for faster training across multiple models
PATIENCE = 10

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "plots").mkdir(exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)
(OUTPUT_DIR / "reports").mkdir(exist_ok=True)

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("="*80)
print(" "*15 + "MULTI-MODEL SEISMIC WAVE CLASSIFICATION")
print(" "*20 + "Comprehensive Model Comparison")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nConfiguration:")
print(f"  Training directory:  {TRAIN_DIR}")
print(f"  Test directory:      {TEST_DIR}")
print(f"  Output directory:    {OUTPUT_DIR}")
print(f"  Classes:             {CLASS_NAMES}")
print(f"  Models to train:     10 different architectures")
print(f"  Max epochs/model:    {EPOCHS}")

# Check GPU
print(f"\nGPU Configuration:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"  ✓ {len(gpus)} GPU(s) detected")
    for i, gpu in enumerate(gpus):
        print(f"    GPU {i}: {gpu.name}")
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print(f"  ⚠ No GPU detected - training will use CPU")

# ==============================
# HELPER FUNCTIONS
# ==============================

def load_audio_files(data_dir, class_names):
    """Load audio file paths and labels from directory."""
    filepaths = []
    labels = []
    
    print(f"\nLoading files from: {data_dir}")
    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  ✗ Directory not found: {class_dir}")
            continue
        
        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"  {class_name:10s}: {len(wav_files):5d} files")
        
        for wav_file in wav_files:
            filepaths.append(str(wav_file))
            labels.append(class_name)
    
    print(f"Total files loaded: {len(filepaths)}")
    return filepaths, labels

def extract_mfcc_features(filepaths, description="MFCC"):
    """Extract MFCC features from audio files."""
    mfcc_list = []
    max_frames = 0
    
    for filepath in tqdm(filepaths, desc=description, ncols=100):
        y, _ = librosa.load(filepath, sr=SR, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_list.append(mfcc)
        if mfcc.shape[1] > max_frames:
            max_frames = mfcc.shape[1]
    
    # Pad to max_frames
    X = np.zeros((len(mfcc_list), N_MFCC, max_frames), dtype=np.float32)
    for i, mfcc in enumerate(mfcc_list):
        frames = mfcc.shape[1]
        X[i, :, :frames] = mfcc
    
    # Add channel dimension
    X = X[..., np.newaxis]
    return X, max_frames

# ==============================
# MODEL ARCHITECTURES
# ==============================

def build_cnn_2d(input_shape, n_classes):
    """Standard 2D CNN for MFCC spectrograms."""
    model = models.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 1), padding='same'),
        Conv2D(64, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 1), padding='same'),
        Conv2D(128, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 1), padding='same'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], name='CNN_2D')
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_with_attention(input_shape, n_classes):
    """2D CNN with attention mechanism."""
    inputs = Input(shape=input_shape)
    
    # CNN layers
    x = Conv2D(32, (3, 2), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1), padding='same')(x)
    
    x = Conv2D(64, (3, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1), padding='same')(x)
    
    x = Conv2D(128, (3, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Attention mechanism
    attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    x = Multiply()([x, attention])
    
    x = MaxPool2D((2, 1), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_Attention')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet_style(input_shape, n_classes):
    """ResNet-style CNN with skip connections."""
    inputs = Input(shape=input_shape)
    
    # Initial conv
    x = Conv2D(32, (3, 2), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    
    # Residual block 1
    shortcut = x
    x = Conv2D(32, (3, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPool2D((2, 1), padding='same')(x)
    
    # Residual block 2
    x = Conv2D(64, (3, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(64, (1, 1), padding='same')(x)
    x = Conv2D(64, (3, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPool2D((2, 1), padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='ResNet_Style')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape, n_classes):
    """LSTM for sequence classification."""
    # Reshape for LSTM: (batch, time_steps, features)
    # input_shape is (40, frames, 1), need to reshape to (frames, 40)
    model = models.Sequential([
        Input(shape=input_shape),
        Reshape((input_shape[1], input_shape[0])),  # (frames, 40)
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name='LSTM')
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bidirectional_lstm(input_shape, n_classes):
    """Bidirectional LSTM."""
    model = models.Sequential([
        Input(shape=input_shape),
        Reshape((input_shape[1], input_shape[0])),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name='BiLSTM')
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru(input_shape, n_classes):
    """GRU (Gated Recurrent Unit)."""
    model = models.Sequential([
        Input(shape=input_shape),
        Reshape((input_shape[1], input_shape[0])),
        GRU(128, return_sequences=True),
        Dropout(0.3),
        GRU(64),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name='GRU')
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_lstm_hybrid(input_shape, n_classes):
    """Hybrid CNN-LSTM architecture."""
    inputs = Input(shape=input_shape)
    
    # CNN feature extraction
    x = Conv2D(32, (3, 2), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1), padding='same')(x)
    
    x = Conv2D(64, (3, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D((2, 1), padding='same')(x)
    
    # Reshape for LSTM: collapse frequency dimension
    shape = x.shape
    x = Reshape((shape[2], shape[1] * shape[3]))(x)  # (time, freq*channels)
    
    # LSTM layers
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32)(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_1d(input_shape, n_classes):
    """1D CNN treating each MFCC coefficient as a feature."""
    # Reshape from (40, frames, 1) to (frames, 40)
    model = models.Sequential([
        Input(shape=input_shape),
        Reshape((input_shape[1], input_shape[0])),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool1D(pool_size=2, padding='same'),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool1D(pool_size=2, padding='same'),
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], name='CNN_1D')
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_dense_nn(input_shape, n_classes):
    """Simple Dense Neural Network."""
    model = models.Sequential([
        Input(shape=input_shape),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ], name='Dense_NN')
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_deep_cnn(input_shape, n_classes):
    """Deeper 2D CNN architecture."""
    model = models.Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 1), padding='same'),
        
        Conv2D(64, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 1), padding='same'),
        
        Conv2D(128, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 2), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D((2, 1), padding='same'),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ], name='Deep_CNN')
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==============================
# LOAD AND PREPARE DATA
# ==============================
print("\n" + "="*80)
print("STEP 1: Loading Training Data")
print("="*80)

train_files, train_labels = load_audio_files(TRAIN_DIR, CLASS_NAMES)
if len(train_files) == 0:
    print("\n✗ ERROR: No training files found!")
    exit(1)

# Shuffle
combined = list(zip(train_files, train_labels))
random.shuffle(combined)
train_files, train_labels = zip(*combined)

print("\n" + "="*80)
print("STEP 2: Extracting Training MFCC Features")
print("="*80)

X_train_full, max_frames = extract_mfcc_features(train_files, "Training MFCC")
print(f"Feature matrix shape: {X_train_full.shape}")

# Encode labels
le = LabelEncoder()
y_train_int = le.fit_transform(train_labels)
y_train_full = utils.to_categorical(y_train_int, num_classes=len(le.classes_))

# Split training into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=VAL_SPLIT,
    random_state=RANDOM_SEED,
    stratify=y_train_int
)

print(f"\nData split:")
print(f"  Training set:    {X_train.shape[0]:5d} samples")
print(f"  Validation set:  {X_val.shape[0]:5d} samples")

# Load test data
print("\n" + "="*80)
print("STEP 3: Loading Test Data")
print("="*80)

test_files, test_labels = load_audio_files(TEST_DIR, CLASS_NAMES)
X_test_raw, test_max_frames = extract_mfcc_features(test_files, "Test MFCC")

# Adjust test dimensions
if test_max_frames != max_frames:
    X_test = np.zeros((len(test_files), N_MFCC, max_frames, 1), dtype=np.float32)
    for i in range(len(test_files)):
        frames_to_copy = min(test_max_frames, max_frames)
        X_test[i, :, :frames_to_copy, 0] = X_test_raw[i, :, :frames_to_copy, 0]
else:
    X_test = X_test_raw

y_test_int = le.transform(test_labels)
y_test = utils.to_categorical(y_test_int, num_classes=len(le.classes_))

print(f"Test feature matrix shape: {X_test.shape}")

# ==============================
# TRAIN ALL MODELS
# ==============================
print("\n" + "="*80)
print("STEP 4: Training Multiple Models")
print("="*80)

input_shape = X_train.shape[1:]
n_classes = len(le.classes_)

# Define all models to train
model_builders = [
    ('CNN_2D', build_cnn_2d),
    ('CNN_Attention', build_cnn_with_attention),
    ('ResNet_Style', build_resnet_style),
    ('LSTM', build_lstm),
    ('BiLSTM', build_bidirectional_lstm),
    ('GRU', build_gru),
    ('CNN_LSTM', build_cnn_lstm_hybrid),
    ('CNN_1D', build_cnn_1d),
    ('Dense_NN', build_dense_nn),
    ('Deep_CNN', build_deep_cnn)
]

results = []
all_predictions = {}
training_times = {}

for model_name, builder_func in model_builders:
    print("\n" + "="*80)
    print(f"Training Model: {model_name}")
    print("="*80)
    
    # Build model
    model = builder_func(input_shape, n_classes)
    print(f"\nModel Parameters: {model.count_params():,}")
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=0,
        min_lr=1e-6
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        str(OUTPUT_DIR / 'models' / f'{model_name}_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=0
    )
    
    training_time = time.time() - start_time
    training_times[model_name] = training_time
    
    # Evaluate on validation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    # Evaluate on test
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_test_pred_prob = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)
    y_test_true = np.argmax(y_test, axis=1)
    
    # Per-class metrics
    test_f1 = f1_score(y_test_true, y_test_pred, average='weighted')
    
    # Store results
    result = {
        'model_name': model_name,
        'parameters': int(model.count_params()),
        'epochs_trained': len(history.history['loss']),
        'training_time': training_time,
        'val_accuracy': float(val_acc),
        'val_loss': float(val_loss),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'test_f1_score': float(test_f1),
        'best_val_acc': float(max(history.history['val_accuracy']))
    }
    results.append(result)
    
    # Store predictions for ensemble
    all_predictions[model_name] = y_test_pred_prob
    
    # Print summary
    print(f"\n{model_name} Results:")
    print(f"  Training time:    {training_time:.1f}s")
    print(f"  Epochs trained:   {len(history.history['loss'])}")
    print(f"  Val accuracy:     {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test accuracy:    {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test F1-score:    {test_f1:.4f}")
    
    # Save detailed report
    test_report = classification_report(y_test_true, y_test_pred, target_names=le.classes_, digits=4)
    with open(OUTPUT_DIR / 'reports' / f'{model_name}_report.txt', 'w') as f:
        f.write(f"{model_name} - TEST SET CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(test_report)
    
    # Save model
    model.save(OUTPUT_DIR / 'models' / f'{model_name}_final.h5')
    
    # Clean up
    del model
    tf.keras.backend.clear_session()

# ==============================
# ENSEMBLE MODEL
# ==============================
print("\n" + "="*80)
print("STEP 5: Creating Ensemble Model")
print("="*80)

# Average predictions from all models
ensemble_pred_prob = np.mean(list(all_predictions.values()), axis=0)
ensemble_pred = np.argmax(ensemble_pred_prob, axis=1)
y_test_true = np.argmax(y_test, axis=1)

ensemble_acc = accuracy_score(y_test_true, ensemble_pred)
ensemble_f1 = f1_score(y_test_true, ensemble_pred, average='weighted')

print(f"\nEnsemble (Average of {len(all_predictions)} models):")
print(f"  Test accuracy:    {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
print(f"  Test F1-score:    {ensemble_f1:.4f}")

# Add ensemble to results
results.append({
    'model_name': 'Ensemble',
    'parameters': sum(r['parameters'] for r in results),
    'epochs_trained': 0,
    'training_time': sum(training_times.values()),
    'val_accuracy': 0,
    'val_loss': 0,
    'test_accuracy': float(ensemble_acc),
    'test_loss': 0,
    'test_f1_score': float(ensemble_f1),
    'best_val_acc': 0
})

# Save ensemble report
ensemble_report = classification_report(y_test_true, ensemble_pred, target_names=le.classes_, digits=4)
with open(OUTPUT_DIR / 'reports' / 'Ensemble_report.txt', 'w') as f:
    f.write("ENSEMBLE MODEL - TEST SET CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(ensemble_report)

# ==============================
# VISUALIZATIONS
# ==============================
print("\n" + "="*80)
print("STEP 6: Creating Comparison Visualizations")
print("="*80)

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_accuracy', ascending=False)

# Save results
results_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False)
print(f"\n✓ Saved: model_comparison.csv")

# 1. Model Accuracy Comparison
fig, ax = plt.subplots(figsize=(14, 8))
models = results_df['model_name']
test_acc = results_df['test_accuracy'] * 100

colors = ['#2ecc71' if acc == test_acc.max() else '#3498db' for acc in test_acc]
bars = ax.barh(models, test_acc, color=colors, edgecolor='black', linewidth=1.2)

ax.set_xlabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Model Architecture', fontsize=13, fontweight='bold')
ax.set_title('Model Performance Comparison on Test Set', fontsize=15, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim([0, 100])

# Add value labels
for i, (bar, acc) in enumerate(zip(bars, test_acc)):
    ax.text(acc + 1, bar.get_y() + bar.get_height()/2, 
            f'{acc:.2f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'model_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: plots/model_comparison.png")
plt.close()

# 2. Accuracy vs Training Time
fig, ax = plt.subplots(figsize=(12, 8))
results_df_no_ensemble = results_df[results_df['model_name'] != 'Ensemble']

scatter = ax.scatter(results_df_no_ensemble['training_time'], 
                     results_df_no_ensemble['test_accuracy'] * 100,
                     s=results_df_no_ensemble['parameters'] / 1000,
                     alpha=0.6, c=results_df_no_ensemble['test_accuracy'],
                     cmap='viridis', edgecolors='black', linewidth=1.5)

for idx, row in results_df_no_ensemble.iterrows():
    ax.annotate(row['model_name'], 
                (row['training_time'], row['test_accuracy'] * 100),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs Training Time (bubble size = parameters)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Test Accuracy', fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'accuracy_vs_time.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: plots/accuracy_vs_time.png")
plt.close()

# 3. Model Efficiency (Accuracy / Parameters)
results_df_no_ensemble['efficiency'] = (results_df_no_ensemble['test_accuracy'] * 100) / (results_df_no_ensemble['parameters'] / 1000)

fig, ax = plt.subplots(figsize=(14, 8))
models = results_df_no_ensemble['model_name']
efficiency = results_df_no_ensemble['efficiency']

bars = ax.barh(models, efficiency, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.2)
ax.set_xlabel('Efficiency (Accuracy % per 1K parameters)', fontsize=12, fontweight='bold')
ax.set_ylabel('Model Architecture', fontsize=12, fontweight='bold')
ax.set_title('Model Efficiency Comparison', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'model_efficiency.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: plots/model_efficiency.png")
plt.close()

# 4. F1-Score Comparison
fig, ax = plt.subplots(figsize=(14, 8))
models = results_df['model_name']
f1_scores = results_df['test_f1_score'] * 100

bars = ax.barh(models, f1_scores, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1.2)
ax.set_xlabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Model Architecture', fontsize=12, fontweight='bold')
ax.set_title('F1-Score Comparison on Test Set', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim([0, 100])

for bar, f1 in zip(bars, f1_scores):
    ax.text(f1 + 1, bar.get_y() + bar.get_height()/2, 
            f'{f1:.2f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'f1_score_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: plots/f1_score_comparison.png")
plt.close()

# 5. Top 5 Models Detailed Comparison
top_5 = results_df.head(5)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Test Accuracy
axes[0, 0].bar(top_5['model_name'], top_5['test_accuracy'] * 100, color='skyblue', edgecolor='black')
axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
axes[0, 0].set_title('Test Accuracy - Top 5 Models', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# F1-Score
axes[0, 1].bar(top_5['model_name'], top_5['test_f1_score'] * 100, color='lightgreen', edgecolor='black')
axes[0, 1].set_ylabel('F1-Score (%)', fontweight='bold')
axes[0, 1].set_title('F1-Score - Top 5 Models', fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# Training Time
axes[1, 0].bar(top_5['model_name'], top_5['training_time'], color='coral', edgecolor='black')
axes[1, 0].set_ylabel('Time (seconds)', fontweight='bold')
axes[1, 0].set_title('Training Time - Top 5 Models', fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Parameters
axes[1, 1].bar(top_5['model_name'], top_5['parameters'] / 1000, color='plum', edgecolor='black')
axes[1, 1].set_ylabel('Parameters (K)', fontweight='bold')
axes[1, 1].set_title('Model Size - Top 5 Models', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'plots' / 'top5_detailed_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: plots/top5_detailed_comparison.png")
plt.close()

# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "="*80)
print(" "*20 + "TRAINING COMPLETE! 🎉")
print("="*80)

print(f"\n📊 Results Summary:")
print(f"\nTop 5 Models by Test Accuracy:")
for i, row in results_df.head(5).iterrows():
    print(f"  {row['model_name']:20s} - {row['test_accuracy']*100:.2f}% (F1: {row['test_f1_score']*100:.2f}%)")

print(f"\n⚡ Fastest Training:")
fastest = results_df_no_ensemble.nsmallest(3, 'training_time')
for i, row in fastest.iterrows():
    print(f"  {row['model_name']:20s} - {row['training_time']:.1f}s (Acc: {row['test_accuracy']*100:.2f}%)")

print(f"\n💪 Most Efficient (Accuracy/Parameters):")
most_efficient = results_df_no_ensemble.nlargest(3, 'efficiency')
for i, row in most_efficient.iterrows():
    print(f"  {row['model_name']:20s} - {row['efficiency']:.4f} (Acc: {row['test_accuracy']*100:.2f}%)")

print(f"\n📁 All outputs saved to: {OUTPUT_DIR.absolute()}")
print(f"   ├── model_comparison.csv          (Detailed comparison table)")
print(f"   ├── models/                       ({len(model_builders)} trained models)")
print(f"   ├── plots/                        (5 comparison visualizations)")
print(f"   └── reports/                      ({len(model_builders)+1} classification reports)")

print("\n" + "="*80)
print("Total training time: {:.1f} minutes".format(sum(training_times.values()) / 60))
print("="*80 + "\n")
