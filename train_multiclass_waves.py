#!/usr/bin/env python3
"""
Multi-Class Wave Type Training
Extracts P-wave, S-wave, Coda, and Noise segments from archive data
Trains CNN to classify wave types using arrival time annotations
"""

import os
import numpy as np
import pandas as pd
import h5py
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
from tqdm import tqdm

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Available: {len(gpus)} device(s)")
        print(f"   {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"⚠️  GPU Configuration Error: {e}")

# Training Parameters
SAMPLE_RATE = 100  # Hz
N_MFCC = 40
N_FRAMES = 4
BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 10

# Wave segment parameters
SEGMENT_DURATION_SAMPLES = 300  # 3 seconds @ 100Hz
PRE_ARRIVAL_SAMPLES = 50  # 0.5 seconds before arrival

# Paths
CSV_PATH = 'archive/merge.csv'
HDF5_PATH = 'archive/merge.hdf5'
OUTPUT_DIR = 'trained_models_multiclass'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names
CLASS_NAMES = ['p_wave', 's_wave', 'coda', 'noise']
NUM_CLASSES = len(CLASS_NAMES)

print("=" * 80)
print("MULTI-CLASS WAVE TYPE TRAINING")
print("=" * 80)
print(f"Classes: {CLASS_NAMES}")
print(f"Segment duration: {SEGMENT_DURATION_SAMPLES/SAMPLE_RATE:.1f} seconds")
print(f"Output: {OUTPUT_DIR}/")
print("=" * 80)


def clean_sample_value(val):
    """Convert arrival sample value to float"""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    # Handle string format like "[[2896.]]"
    val_str = str(val).replace('[', '').replace(']', '').strip()
    return float(val_str) if val_str else np.nan


def seismic_to_audio(waveform, sr=100):
    """Convert 3-channel seismic to mono audio (RMS combination)"""
    if len(waveform.shape) > 1:
        audio = np.sqrt(np.mean(waveform ** 2, axis=0))
    else:
        audio = waveform
    return audio.astype(np.float32)


def extract_mfcc_features(audio, sr=100, n_mfcc=40, n_frames=4):
    """Extract MFCC features from audio segment"""
    try:
        # Ensure minimum length
        if len(audio) < n_frames:
            audio = np.pad(audio, (0, n_frames - len(audio)), mode='constant')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Average across time to get n_frames
        if mfccs.shape[1] < n_frames:
            mfccs = np.pad(mfccs, ((0, 0), (0, n_frames - mfccs.shape[1])), mode='constant')
        else:
            split_size = max(1, mfccs.shape[1] // n_frames)
            mfccs = np.array([mfccs[:, i*split_size:(i+1)*split_size].mean(axis=1) 
                             for i in range(n_frames)]).T
        
        mfccs = mfccs.reshape(n_mfcc, n_frames, 1)
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return np.zeros((n_mfcc, n_frames, 1))


def extract_wave_segment(waveform, arrival_sample, duration=300, pre_arrival=50):
    """Extract segment around wave arrival"""
    start_sample = max(0, int(arrival_sample) - pre_arrival)
    end_sample = min(len(waveform), start_sample + duration)
    
    segment = waveform[start_sample:end_sample]
    
    # Pad if too short
    if len(segment) < duration:
        segment = np.pad(segment, (0, duration - len(segment)), mode='constant')
    
    return segment


class MultiClassWaveGenerator(keras.utils.Sequence):
    """Data generator for multi-class wave type training"""
    
    def __init__(self, csv_data, hdf5_path, batch_size=128, 
                 segment_duration=300, shuffle=True):
        # Keep original index as a column before reset
        self.csv_data = csv_data.copy()
        self.csv_data['original_idx'] = self.csv_data.index
        self.csv_data = self.csv_data.reset_index(drop=True)
        
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.segment_duration = segment_duration
        self.shuffle = shuffle
        
        # Build sample index
        self.samples = []
        self._build_sample_index()
        
        if self.shuffle:
            np.random.shuffle(self.samples)
        
        print(f"   Generator created: {len(self.samples):,} samples")
    
    def _build_sample_index(self):
        """Build index of all extractable wave segments"""
        print("   Building sample index...")
        
        # Process earthquakes - extract P, S, and Coda segments
        eq_data = self.csv_data[self.csv_data['trace_category'] == 'earthquake_local']
        
        for idx in range(len(eq_data)):
            row = eq_data.iloc[idx]
            trace_idx = row['original_idx']  # Original HDF5 index
            
            # Clean arrival values
            p_sample = clean_sample_value(row['p_arrival_sample'])
            s_sample = clean_sample_value(row['s_arrival_sample'])
            coda_sample = clean_sample_value(row['coda_end_sample'])
            
            # Add P-wave segment (class 0)
            if not np.isnan(p_sample) and p_sample > PRE_ARRIVAL_SAMPLES:
                self.samples.append((trace_idx, 0, p_sample))
            
            # Add S-wave segment (class 1)
            if not np.isnan(s_sample) and s_sample > PRE_ARRIVAL_SAMPLES:
                self.samples.append((trace_idx, 1, s_sample))
            
            # Add Coda segment (class 2) - after S-wave
            if not np.isnan(s_sample) and not np.isnan(coda_sample):
                coda_start = s_sample + (coda_sample - s_sample) / 2
                if coda_start < 6000 - SEGMENT_DURATION_SAMPLES:
                    self.samples.append((trace_idx, 2, coda_start))
        
        # Process noise - extract random segments (class 3)
        noise_data = self.csv_data[self.csv_data['trace_category'] == 'noise']
        
        for idx in range(len(noise_data)):
            row = noise_data.iloc[idx]
            trace_idx = row['original_idx']  # Original HDF5 index
            
            # Extract 3 random segments per noise trace for balance
            for _ in range(3):
                random_start = np.random.randint(100, 6000 - SEGMENT_DURATION_SAMPLES)
                self.samples.append((trace_idx, 3, random_start))
        
        print(f"      P-wave segments: {sum(1 for s in self.samples if s[1] == 0):,}")
        print(f"      S-wave segments: {sum(1 for s in self.samples if s[1] == 1):,}")
        print(f"      Coda segments: {sum(1 for s in self.samples if s[1] == 2):,}")
        print(f"      Noise segments: {sum(1 for s in self.samples if s[1] == 3):,}")
    
    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_samples = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X_batch = []
        y_batch = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for trace_idx, class_label, arrival_sample in batch_samples:
                # Load waveform (convert to int for HDF5 indexing)
                waveform_3ch = f['data'][int(trace_idx)]  # (6000, 3)
                
                # Convert to mono
                audio = seismic_to_audio(waveform_3ch, sr=SAMPLE_RATE)
                
                # Extract segment
                segment = extract_wave_segment(audio, arrival_sample, 
                                              duration=self.segment_duration)
                
                # Extract MFCC
                mfcc = extract_mfcc_features(segment, sr=SAMPLE_RATE, 
                                            n_mfcc=N_MFCC, n_frames=N_FRAMES)
                
                X_batch.append(mfcc)
                y_batch.append(class_label)
        
        return np.array(X_batch), keras.utils.to_categorical(y_batch, NUM_CLASSES)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.samples)


def build_cnn_model(input_shape=(40, 4, 1), num_classes=4):
    """Build CNN model for wave type classification"""
    model = keras.Sequential([
        # Conv Block 1
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                           input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Conv Block 2
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Conv Block 3
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling2D(),
        
        # Attention mechanism
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def main():
    start_time = datetime.now()
    
    print("\n📊 Loading CSV metadata...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"   Loaded: {len(df):,} traces")
    print(f"   Earthquakes: {len(df[df['trace_category'] == 'earthquake_local']):,}")
    print(f"   Noise: {len(df[df['trace_category'] == 'noise']):,}")
    
    # Split data
    print("\n✂️  Splitting train/validation...")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42,
                                         stratify=df['trace_category'])
    print(f"   Train: {len(train_df):,} traces")
    print(f"   Validation: {len(val_df):,} traces")
    
    # Create generators
    print("\n🔧 Creating data generators...")
    print("\n   Training generator:")
    train_gen = MultiClassWaveGenerator(train_df, HDF5_PATH, 
                                        batch_size=BATCH_SIZE,
                                        segment_duration=SEGMENT_DURATION_SAMPLES,
                                        shuffle=True)
    
    print("\n   Validation generator:")
    val_gen = MultiClassWaveGenerator(val_df, HDF5_PATH,
                                      batch_size=BATCH_SIZE,
                                      segment_duration=SEGMENT_DURATION_SAMPLES,
                                      shuffle=False)
    
    # Build model
    print("\n🏗️  Building CNN model...")
    model = build_cnn_model(input_shape=(N_MFCC, N_FRAMES, 1), 
                           num_classes=NUM_CLASSES)
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Compile model
    print("\n⚙️  Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'CNN_MultiClass_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_DIR, 'logs', 
                                datetime.now().strftime('%Y%m%d-%H%M%S')),
            histogram_freq=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, 'training_history.csv')
        )
    ]
    
    # Train model
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Train batches: {len(train_gen)}")
    print(f"Val batches: {len(val_gen)}")
    print("=" * 80)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, 'CNN_MultiClass_final.h5')
    model.save(final_model_path)
    print(f"\n✅ Final model saved: {final_model_path}")
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'history': {k: [float(v) for v in vals] 
                       for k, vals in history.history.items()},
            'params': {
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'num_classes': NUM_CLASSES,
                'class_names': CLASS_NAMES,
                'segment_duration_sec': SEGMENT_DURATION_SAMPLES / SAMPLE_RATE
            }
        }, f, indent=2)
    print(f"✅ Training history saved: {history_path}")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("📊 FINAL EVALUATION")
    print("=" * 80)
    
    val_loss, val_acc, val_precision, val_recall = model.evaluate(val_gen, verbose=1)
    
    print(f"\nValidation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.2%}")
    print(f"  Precision: {val_precision:.2%}")
    print(f"  Recall: {val_recall:.2%}")
    print(f"  F1-Score: {2 * (val_precision * val_recall) / (val_precision + val_recall):.2%}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print(f"✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Duration: {duration}")
    print(f"Best model: {os.path.join(OUTPUT_DIR, 'CNN_MultiClass_best.h5')}")
    print(f"Final model: {final_model_path}")
    print("=" * 80)
    
    # Save summary
    summary = {
        'completed_at': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'final_metrics': {
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall)
        },
        'training_config': {
            'classes': CLASS_NAMES,
            'num_classes': NUM_CLASSES,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'segment_duration_sec': SEGMENT_DURATION_SAMPLES / SAMPLE_RATE
        }
    }
    
    summary_path = os.path.join(OUTPUT_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
