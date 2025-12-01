#!/usr/bin/env python3
"""
Archive Data Training Pipeline
- Load 1.26M samples from archive (earthquake_local + noise)
- Convert 3-channel seismic waveforms to MFCC features
- Train multiple models using GPU
- Save best models for testing
"""

import os
import h5py
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

# GPU configuration
print("=" * 80)
print("GPU CONFIGURATION")
print("=" * 80)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Available: {len(gpus)} device(s)")
        print(f"GPU Names: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU Error: {e}")
else:
    print("No GPU detected - training on CPU")

# Parameters
SAMPLE_RATE = 100  # Standard seismic sample rate (Hz)
N_MFCC = 40
N_FRAMES = 4
BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 10
VALIDATION_SPLIT = 0.2

# Paths
ARCHIVE_HDF5 = 'archive/merge.hdf5'
ARCHIVE_CSV = 'archive/merge.csv'
OUTPUT_DIR = 'trained_models_archive'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def seismic_to_audio(waveform, sr=100):
    """
    Convert 3-channel seismic waveform to mono audio
    Seismic channels are typically: [E-W, N-S, Z (vertical)]
    We'll combine them using RMS (root mean square)
    """
    # waveform shape: (6000, 3)
    # Combine 3 channels using RMS
    audio = np.sqrt(np.mean(waveform ** 2, axis=1))
    return audio.astype(np.float32)


def extract_mfcc_features(waveform, sr=100, n_mfcc=40, n_frames=4):
    """
    Extract MFCC features from seismic waveform
    Match the GUI implementation exactly
    """
    try:
        # Convert 3-channel seismic to mono audio
        audio = seismic_to_audio(waveform, sr)
        
        # Extract MFCCs (40 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Average across time to get 4 frames
        if mfccs.shape[1] < n_frames:
            # Pad if too short
            mfccs = np.pad(mfccs, ((0, 0), (0, n_frames - mfccs.shape[1])), mode='constant')
        else:
            # Split into n_frames segments and average
            split_size = mfccs.shape[1] // n_frames
            mfccs = np.array([mfccs[:, i*split_size:(i+1)*split_size].mean(axis=1) 
                             for i in range(n_frames)]).T
        
        # Reshape to (40, 4, 1) to match model input
        mfccs = mfccs.reshape(n_mfcc, n_frames, 1)
        
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return np.zeros((n_mfcc, n_frames, 1))


class ArchiveDataGenerator(tf.keras.utils.Sequence):
    """
    Efficient batch generator for 92GB HDF5 file
    Loads data on-demand to avoid memory overflow
    """
    def __init__(self, trace_names, labels, hdf5_path, batch_size=128, shuffle=True):
        self.trace_names = trace_names
        self.labels = labels
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(trace_names))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.trace_names) / self.batch_size))
    
    def __getitem__(self, idx):
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Load batch data
        X_batch = []
        y_batch = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for i in batch_indices:
                trace_name = self.trace_names[i]
                label = self.labels[i]
                
                # Load waveform from HDF5
                try:
                    waveform = f[f'data/{trace_name}'][:]
                    
                    # Extract MFCC features
                    mfcc = extract_mfcc_features(waveform, sr=SAMPLE_RATE, 
                                                 n_mfcc=N_MFCC, n_frames=N_FRAMES)
                    
                    X_batch.append(mfcc)
                    y_batch.append(label)
                except Exception as e:
                    print(f"Error loading {trace_name}: {e}")
                    continue
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_cnn_attention_model(input_shape=(40, 4, 1), num_classes=2):
    """
    CNN with Attention mechanism for seismic classification
    Proven architecture from current GUI system
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN feature extraction
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    attention = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN_Attention')
    return model


def build_lstm_model(input_shape=(40, 4, 1), num_classes=2):
    """
    LSTM model for temporal sequence learning
    """
    inputs = layers.Input(shape=input_shape)
    
    # Reshape for LSTM: (batch, timesteps, features)
    x = layers.Reshape((input_shape[1], input_shape[0]))(inputs)
    
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='LSTM')
    return model


def build_resnet_model(input_shape=(40, 4, 1), num_classes=2):
    """
    ResNet-style model with residual connections
    """
    inputs = layers.Input(shape=input_shape)
    
    def residual_block(x, filters, kernel_size=(3, 3)):
        fx = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        fx = layers.BatchNormalization()(fx)
        fx = layers.Conv2D(filters, kernel_size, padding='same')(fx)
        fx = layers.BatchNormalization()(fx)
        
        # Match dimensions if needed
        if x.shape[-1] != filters:
            x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        
        out = layers.Add()([x, fx])
        out = layers.Activation('relu')(out)
        return out
    
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = residual_block(x, 256)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='ResNet')
    return model


def train_model(model, train_gen, val_gen, model_name):
    """
    Train a model with callbacks and save best weights
    """
    print("=" * 80)
    print(f"TRAINING {model_name.upper()}")
    print("=" * 80)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    model_path = os.path.join(OUTPUT_DIR, f'{model_name}.h5')
    checkpoint = callbacks.ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    tensorboard = callbacks.TensorBoard(
        log_dir=os.path.join(OUTPUT_DIR, 'logs', model_name),
        histogram_freq=1
    )
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop, reduce_lr, tensorboard],
        verbose=1
    )
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }, f, indent=2)
    
    print(f"\n✅ {model_name} training complete!")
    print(f"Best model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    
    return history


def main():
    """
    Main training pipeline
    """
    start_time = datetime.now()
    
    print("=" * 80)
    print("ARCHIVE DATA TRAINING PIPELINE")
    print("=" * 80)
    print(f"Start time: {start_time}")
    print()
    
    # Load CSV metadata
    print("Loading CSV metadata...")
    df = pd.read_csv(ARCHIVE_CSV, low_memory=False)
    print(f"Total samples: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['trace_category'].value_counts())
    
    # Binary classification: earthquake_local vs noise
    # Map to numeric labels
    label_map = {'earthquake_local': 0, 'noise': 1}
    df['label'] = df['trace_category'].map(label_map)
    
    # Remove any rows with NaN labels (if any)
    df = df.dropna(subset=['label'])
    print(f"\nSamples after filtering: {len(df)}")
    
    # Get trace names (should match HDF5 keys)
    trace_names = df['trace_name'].values
    labels = df['label'].values.astype(np.int32)
    
    print(f"\nLabel distribution:")
    print(f"  Earthquake: {np.sum(labels == 0)} ({100 * np.mean(labels == 0):.1f}%)")
    print(f"  Noise: {np.sum(labels == 1)} ({100 * np.mean(labels == 1):.1f}%)")
    
    # Train/validation split
    print(f"\nCreating train/validation split ({100*(1-VALIDATION_SPLIT):.0f}/{100*VALIDATION_SPLIT:.0f})...")
    train_names, val_names, train_labels, val_labels = train_test_split(
        trace_names, labels, 
        test_size=VALIDATION_SPLIT, 
        stratify=labels,
        random_state=42
    )
    
    print(f"Train samples: {len(train_names)}")
    print(f"Validation samples: {len(val_names)}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen = ArchiveDataGenerator(
        train_names, train_labels, ARCHIVE_HDF5, 
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_gen = ArchiveDataGenerator(
        val_names, val_labels, ARCHIVE_HDF5, 
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    print(f"Train batches: {len(train_gen)}")
    print(f"Validation batches: {len(val_gen)}")
    
    # Test data generator with one batch
    print("\nTesting data generator...")
    X_test, y_test = train_gen[0]
    print(f"Batch shape: X={X_test.shape}, y={y_test.shape}")
    print(f"X dtype: {X_test.dtype}, y dtype: {y_test.dtype}")
    print(f"X range: [{X_test.min():.2f}, {X_test.max():.2f}]")
    
    # Build models
    print("\n" + "=" * 80)
    print("BUILDING MODELS")
    print("=" * 80)
    
    models_to_train = [
        ('CNN_Attention', build_cnn_attention_model),
        ('LSTM', build_lstm_model),
        ('ResNet', build_resnet_model)
    ]
    
    results = {}
    
    for model_name, build_fn in models_to_train:
        print(f"\nBuilding {model_name}...")
        model = build_fn(input_shape=(N_MFCC, N_FRAMES, 1), num_classes=2)
        print(f"Parameters: {model.count_params():,}")
        
        # Train model
        history = train_model(model, train_gen, val_gen, model_name)
        
        # Store results
        results[model_name] = {
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_val_loss': float(min(history.history['val_loss'])),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'epochs_trained': len(history.history['loss'])
        }
        
        # Clear memory
        del model
        tf.keras.backend.clear_session()
    
    # Save overall results
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    results_path = os.path.join(OUTPUT_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Best Validation Accuracy: {metrics['best_val_accuracy']:.4f}")
        print(f"  Best Validation Loss: {metrics['best_val_loss']:.4f}")
        print(f"  Final Train Accuracy: {metrics['final_train_accuracy']:.4f}")
        print(f"  Epochs Trained: {metrics['epochs_trained']}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n" + "=" * 80)
    print(f"Total training time: {duration}")
    print(f"End time: {end_time}")
    print("=" * 80)


if __name__ == '__main__':
    main()
