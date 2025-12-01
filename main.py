# seismic_classification_stead.py
# Single-file Kaggle notebook script:
# - prepares 4s windows from STEAD merge.hdf5 + merge.csv
# - saves 16-bit PCM .wav files per-class
# - extracts MFCCs (n_mfcc=40, n_fft=256, hop_length=128)
# - trains a 2D CNN that handles "tall-and-thin" inputs
# - prints evaluation + classification report + confusion heatmap

import os
import random
import warnings
import math
from pathlib import Path
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import h5py
import soundfile as sf    # pip-installed in Kaggle usually; fallback to scipy if not available
from scipy.io import wavfile
import librosa
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration (tweak as needed)
# -----------------------------
DATA_DIR = Path("./archive")
CSV_PATH = DATA_DIR / "merge.csv"
HDF5_PATH = DATA_DIR / "merge.hdf5"
OUTPUT_AUDIO_DIR = Path("./wave_audio_dataset")
CLASS_NAMES = ["noise", "p_wave", "s_wave", "coda"]
N_SAMPLES_PER_CLASS = 5000  # set lower for debugging
SR = 100  # sampling rate in Hz (per your spec)
WINDOW_SECONDS = 4
WINDOW_SAMPLES = int(WINDOW_SECONDS * SR)  # = 400
N_MFCC = 40
N_FFT = 256
HOP_LENGTH = 128
RANDOM_SEED = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 60

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Helper: create output directories
for c in CLASS_NAMES:
    (OUTPUT_AUDIO_DIR / c).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helpers for HDF5 waveform access
# -----------------------------
def find_waveform_dataset(h5file):
    """
    Attempt to find the dataset within merge.hdf5 that contains waveform arrays.
    Returns the dataset name (string) and the dataset/group object.
    """
    # Check known likely names first
    candidates = ['data', 'waveforms', 'waveform', 'signals', 'traces', 'X', 'y']
    for name in candidates:
        if name in h5file:
            ds = h5file[name]
            # Accept both Dataset and Group (STEAD uses Group with trace names as keys)
            if isinstance(ds, (h5py.Dataset, h5py.Group)):
                print(f"Found waveform dataset/group at key: '{name}' (type: {type(ds).__name__})")
                return name, ds
    # Fallback: find the first dataset or group with appropriate structure
    for k in h5file.keys():
        obj = h5file[k]
        if isinstance(obj, (h5py.Dataset, h5py.Group)):
            if isinstance(obj, h5py.Dataset):
                shape = getattr(obj, 'shape', None)
                if shape is not None and (len(shape) == 2 or len(shape) == 1):
                    print(f"Using dataset '{k}' as waveform dataset with shape {shape}")
                    return k, obj
            else:
                # It's a group, assume it contains traces
                print(f"Using group '{k}' as waveform container")
                return k, obj
    raise RuntimeError("Could not find waveform dataset inside HDF5 automatically. "
                       "Please inspect the file and set the correct dataset key.")

def read_waveform_from_hdf5(h5file, dataset_obj, record):
    """
    Given an HDF5 dataset/group and a metadata record (a dict from merge.csv),
    extract the associated waveform array.
    Strategy:
    - If dataset_obj is a Group (like STEAD's 'data' group), use trace_name as key
    - If dataset_obj is a 2D Dataset, use index-based access
    - Common STEAD patterns: key may be trace_name or index mapping.
    """
    # Case 1: dataset_obj is a Group (hierarchical structure with trace names)
    if isinstance(dataset_obj, h5py.Group):
        # Try to use trace_name from the record
        for key in ['trace_name', 'filename', 'file_name', 'name']:
            if key in record and pd.notna(record[key]):
                trace_name = str(record[key])
                if trace_name in dataset_obj:
                    arr = np.array(dataset_obj[trace_name], dtype=np.float32)
                    # If multi-channel (e.g., shape (6000, 3)), take first channel or mean
                    if len(arr.shape) > 1 and arr.shape[1] > 1:
                        # Take the first channel (typically Z-component)
                        arr = arr[:, 0]
                    return arr.squeeze()
        return None
    
    # Case 2: dataset_obj is a 2D Dataset (flat array structure)
    ds = dataset_obj
    if isinstance(ds, h5py.Dataset) and len(ds.shape) >= 2:
        # Try to use an index: many csvs include a column 'index' or an integer id column
        for id_key in ['index', 'idx', 'id', 'ID', 'row_id']:
            if id_key in record and pd.notna(record.get(id_key, None)):
                try:
                    idx = int(record[id_key])
                    return np.array(ds[idx], dtype=np.float32).squeeze()
                except Exception:
                    pass
        # Try 'trace_name' or similar mapping to an attribute or separate dataset mapping
        # If merge.hdf5 contains a 'trace_names' dataset, try to find matching entry
        if 'trace_name' in record and record['trace_name'] is not None:
            tn = record['trace_name']
            # attempt to search a text dataset 'trace_names' if present
            if 'trace_names' in h5file:
                names = h5file['trace_names']
                # decode if bytes
                try:
                    names_list = [n.decode('utf8') if isinstance(n, (bytes, bytearray)) else str(n) for n in names[:]]
                    if tn in names_list:
                        idx = names_list.index(tn)
                        return np.array(ds[idx], dtype=np.float32).squeeze()
                except Exception:
                    pass
        # Last resort: try to use an available 'index' field in record named 'trace_idx' or 'trace_index'
        for key in record.keys():
            if 'idx' in key.lower() and pd.notna(record.get(key, None)):
                try:
                    idx = int(record[key])
                    return np.array(ds[idx], dtype=np.float32).squeeze()
                except Exception:
                    pass
        # If nothing matched, attempt to fallback to sequential read by reading position by current loop
        # We'll return None so calling code can skip
        return None
    else:
        # If dataset is group mapping: e.g., dataset contains subkeys per trace_name
        # try to use record['trace_name'] as key
        if isinstance(h5file, h5py.File):
            for key in ['trace_name', 'filename', 'file_name']:
                if key in record and pd.notna(record[key]):
                    keyval = record[key]
                    if keyval in h5file:
                        arr = np.array(h5file[keyval], dtype=np.float32).squeeze()
                        return arr
        return None

# -----------------------------
# Task 1: Data Preparation & .wav Generation
# -----------------------------
def prepare_audio_windows(csv_path, hdf5_path, output_dir, n_per_class=N_SAMPLES_PER_CLASS):
    """
    Reads merge.csv and merge.hdf5 and writes 4s windows as .wav files under output_dir/{class}/
    - Earthquake traces: extract P-wave, S-wave, and coda from seismic events
    - Noise traces: extract background noise samples
    Returns dictionary mapping class -> list of generated file paths.
    """
    print("Loading CSV metadata...")
    df_all = pd.read_csv(csv_path, low_memory=False)
    
    print(f"Total records: {len(df_all)}")
    
    # Split into earthquake and noise traces
    if 'trace_category' in df_all.columns:
        df_earthquake = df_all[df_all['trace_category'] == 'earthquake_local'].copy()
        df_noise = df_all[df_all['trace_category'] == 'noise'].copy()
        print(f"Earthquake traces: {len(df_earthquake)}")
        print(f"Noise traces: {len(df_noise)}")
    else:
        df_earthquake = df_all.copy()
        df_noise = pd.DataFrame()
        print("Warning: 'trace_category' column not found, treating all as earthquake traces")
    
    # Process earthquake traces for P, S, and coda arrival times
    for col in ['p_arrival_sample', 's_arrival_sample']:
        if col in df_earthquake.columns:
            df_earthquake[col] = pd.to_numeric(df_earthquake[col], errors='coerce')
        else:
            df_earthquake[col] = np.nan
    
    # Special handling for coda_end_sample which may be stored as string "[[value]]"
    if 'coda_end_sample' in df_earthquake.columns:
        def parse_coda_end(val):
            if pd.isna(val):
                return np.nan
            try:
                # Try direct numeric conversion first
                return float(val)
            except (ValueError, TypeError):
                # Handle string format like "[[2896.]]"
                try:
                    import ast
                    parsed = ast.literal_eval(str(val))
                    if isinstance(parsed, list):
                        # Flatten nested lists
                        while isinstance(parsed, list) and len(parsed) > 0:
                            parsed = parsed[0]
                        return float(parsed)
                    return float(parsed)
                except:
                    return np.nan
        
        df_earthquake['coda_end_sample'] = df_earthquake['coda_end_sample'].apply(parse_coda_end)
    else:
        df_earthquake['coda_end_sample'] = np.nan
    
    print(f"Valid earthquake traces with P, S, coda: {df_earthquake[['p_arrival_sample', 's_arrival_sample', 'coda_end_sample']].notna().all(axis=1).sum()}")
    
    # Convert to list-of-dicts for fast iteration
    earthquake_records = df_earthquake.to_dict('records')
    noise_records = df_noise.to_dict('records') if len(df_noise) > 0 else []

    # Convert to list-of-dicts for fast iteration
    earthquake_records = df_earthquake.to_dict('records')
    noise_records = df_noise.to_dict('records') if len(df_noise) > 0 else []

    # Open HDF5 once
    print("Opening HDF5...")
    h5 = h5py.File(hdf5_path, 'r')
    dset_key, dset = find_waveform_dataset(h5)  # may raise if not found

    # Counters
    counts = {c: 0 for c in CLASS_NAMES}
    output_files = {c: [] for c in CLASS_NAMES}

    # Helper to write wav (16-bit PCM)
    def write_wav(signal, out_path, sr=SR):
        # Normalize to int16
        if np.max(np.abs(signal)) == 0:
            scaled = np.zeros_like(signal, dtype=np.int16)
        else:
            norm = signal / np.max(np.abs(signal))
            scaled = (norm * 32767).astype(np.int16)
        # Use scipy.io.wavfile to write to ensure 16-bit PCM
        wavfile.write(out_path, sr, scaled)

    # ========== PART 1: Extract P-wave, S-wave, and Coda from earthquake traces ==========
    print("\n=== Extracting P-wave, S-wave, and Coda from earthquake traces ===")
    earthquake_indices = list(range(len(earthquake_records)))
    random.shuffle(earthquake_indices)

    # Create progress bars for each class
    pbar_p = tqdm(total=n_per_class, desc="P-wave", position=0, leave=True, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
    pbar_s = tqdm(total=n_per_class, desc="S-wave", position=1, leave=True, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')
    pbar_c = tqdm(total=n_per_class, desc="Coda  ", position=2, leave=True, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')

    for i in earthquake_indices:
        # Check if we're done with earthquake-related classes
        if (counts['p_wave'] >= n_per_class and 
            counts['s_wave'] >= n_per_class and 
            counts['coda'] >= n_per_class):
            break

        rec = earthquake_records[i]
        
        # read waveform array from HDF5 for this record
        try:
            waveform = read_waveform_from_hdf5(h5, dset, rec)
        except Exception as e:
            waveform = None

        if waveform is None:
            continue

        # ensure waveform is 1D numpy array
        waveform = np.asarray(waveform).squeeze().astype(np.float32)
        L = len(waveform)
        if L < WINDOW_SAMPLES:
            continue

        # Convert arrival samples to ints if available
        p_idx = int(rec['p_arrival_sample']) if 'p_arrival_sample' in rec and not pd.isna(rec['p_arrival_sample']) else None
        s_idx = int(rec['s_arrival_sample']) if 's_arrival_sample' in rec and not pd.isna(rec['s_arrival_sample']) else None
        coda_end_idx = int(rec['coda_end_sample']) if 'coda_end_sample' in rec and not pd.isna(rec['coda_end_sample']) else None

        # 1) P-wave extraction: window centered on p_idx
        if counts['p_wave'] < n_per_class and p_idx is not None and 0 < p_idx < L:
            # center p_idx
            start = p_idx - WINDOW_SAMPLES // 2
            end = start + WINDOW_SAMPLES
            # adjust boundaries
            if start < 0:
                start = 0
                end = WINDOW_SAMPLES
            if end > L:
                end = L
                start = max(0, L - WINDOW_SAMPLES)
            if end - start == WINDOW_SAMPLES:
                window = waveform[start:end]
                out_path = output_dir / "p_wave" / f"p_{counts['p_wave']:06d}.wav"
                write_wav(window, str(out_path), sr=SR)
                output_files['p_wave'].append(str(out_path))
                counts['p_wave'] += 1
                pbar_p.update(1)

        # 2) S-wave extraction: window centered on s_idx
        if counts['s_wave'] < n_per_class and s_idx is not None and 0 < s_idx < L:
            start = s_idx - WINDOW_SAMPLES // 2
            end = start + WINDOW_SAMPLES
            if start < 0:
                start = 0
                end = WINDOW_SAMPLES
            if end > L:
                end = L
                start = max(0, L - WINDOW_SAMPLES)
            if end - start == WINDOW_SAMPLES:
                window = waveform[start:end]
                out_path = output_dir / "s_wave" / f"s_{counts['s_wave']:06d}.wav"
                write_wav(window, str(out_path), sr=SR)
                output_files['s_wave'].append(str(out_path))
                counts['s_wave'] += 1
                pbar_s.update(1)

        # 3) Coda extraction: Extract from coda phase (after S-wave arrival until coda_end)
        # Coda is the tail of the earthquake signal with scattered wave energy
        if counts['coda'] < n_per_class and s_idx is not None and coda_end_idx is not None:
            # Coda phase starts after S-wave and continues to coda_end
            # Add a small buffer after S-wave to ensure we're in the coda phase
            coda_start_buffer = int(SR * 0.5)  # 0.5 second buffer after S-wave
            coda_phase_start = s_idx + coda_start_buffer
            coda_phase_end = int(coda_end_idx)
            
            # Check if there's enough room for a 4s window in the coda phase
            if coda_phase_end > coda_phase_start + WINDOW_SAMPLES and coda_phase_end < L:
                # Sample a random 4s window from the coda phase
                max_start = min(L - WINDOW_SAMPLES, coda_phase_end - WINDOW_SAMPLES)
                min_start = max(0, coda_phase_start)
                
                if max_start > min_start:
                    start = random.randint(min_start, max_start)
                    end = start + WINDOW_SAMPLES
                    if end - start == WINDOW_SAMPLES:
                        window = waveform[start:end]
                        out_path = output_dir / "coda" / f"c_{counts['coda']:06d}.wav"
                        write_wav(window, str(out_path), sr=SR)
                        output_files['coda'].append(str(out_path))
                        counts['coda'] += 1
                        pbar_c.update(1)

    # Close earthquake progress bars
    pbar_p.close()
    pbar_s.close()
    pbar_c.close()

    # ========== PART 2: Extract Noise from noise traces ==========
    print("\n=== Extracting Noise from noise traces ===")
    if len(noise_records) > 0 and counts['noise'] < n_per_class:
        noise_indices = list(range(len(noise_records)))
        random.shuffle(noise_indices)

        # Create progress bar for noise
        pbar_n = tqdm(total=n_per_class, desc="Noise ", position=0, leave=True, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')

        for i in noise_indices:
            if counts['noise'] >= n_per_class:
                break

            rec = noise_records[i]
            
            try:
                waveform = read_waveform_from_hdf5(h5, dset, rec)
            except Exception:
                waveform = None

            if waveform is None:
                continue

            waveform = np.asarray(waveform).squeeze().astype(np.float32)
            L = len(waveform)
            if L < WINDOW_SAMPLES:
                continue

            # For pure noise traces, extract random windows
            attempts = 0
            while attempts < 3 and counts['noise'] < n_per_class:
                start = random.randint(0, L - WINDOW_SAMPLES)
                end = start + WINDOW_SAMPLES
                window = waveform[start:end]
                if len(window) == WINDOW_SAMPLES:
                    out_path = output_dir / "noise" / f"n_{counts['noise']:06d}.wav"
                    write_wav(window, str(out_path), sr=SR)
                    output_files['noise'].append(str(out_path))
                    counts['noise'] += 1
                    pbar_n.update(1)
                    break
                attempts += 1

        pbar_n.close()
    else:
        print("No noise traces available or noise quota already met.")

    h5.close()
    print("Extraction complete. Counts per class:")
    print(counts)
    return output_files

# -----------------------------
# Task 2: Feature Extraction (MFCC)
# -----------------------------
def extract_mfcc_features_from_audio_files(output_files, n_mfcc=N_MFCC, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    For all generated wav files in output_files dict, compute MFCCs with the required n_fft and hop_length.
    Pads time-frames to max_frames and returns X (n_samples, n_mfcc, max_frames, 1) and y labels.
    """
    filepaths = []
    labels = []
    for cls, files in output_files.items():
        for f in files:
            filepaths.append(f)
            labels.append(cls)

    print(f"Total generated WAV files: {len(filepaths)}")
    # Shuffle the list
    combined = list(zip(filepaths, labels))
    random.shuffle(combined)
    filepaths, labels = zip(*combined)

    mfcc_list = []
    max_frames = 0

    print("Extracting MFCCs for each file...")
    for f in tqdm(filepaths, desc="MFCC Extraction", ncols=100):
        # load at desired sr
        y, _ = librosa.load(f, sr=sr, mono=True)
        # compute MFCC with n_fft and hop_length explicitly
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        # mfcc shape will typically be (n_mfcc, n_frames) and for our short input likely (40, 2)
        mfcc_list.append(mfcc)
        if mfcc.shape[1] > max_frames:
            max_frames = mfcc.shape[1]

    print(f"Max frames across MFCCs: {max_frames}")

    # Pad each mfcc to max_frames
    X = np.zeros((len(mfcc_list), n_mfcc, max_frames), dtype=np.float32)
    for i, mf in enumerate(mfcc_list):
        frames = mf.shape[1]
        X[i, :, :frames] = mf

    # Add channel dimension for Conv2D
    X = X[..., np.newaxis]  # shape (n, n_mfcc, max_frames, 1)

    # Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    y_cat = utils.to_categorical(y_int, num_classes=len(le.classes_))

    print(f"Encoded classes: {list(le.classes_)}")
    return X, y_cat, le

# -----------------------------
# Task 3: CNN Model Architecture
# -----------------------------
def build_tall_thin_cnn(input_shape, n_classes):
    """
    Build a Sequential CNN that handles tall-and-thin inputs (e.g., (40, 2, 1)).
    Uses non-square kernels and pool sizes focused on time dimension being small.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Conv block 1
    model.add(layers.Conv2D(32, kernel_size=(3, 2), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 1), padding='same'))

    # Conv block 2
    model.add(layers.Conv2D(64, kernel_size=(3, 2), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 1), padding='same'))

    # Conv block 3
    model.add(layers.Conv2D(128, kernel_size=(3, 2), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 1), padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Plotting helper
# -----------------------------
def plot_confusion_matrix(y_true, y_pred, classes, figsize=(8,6)):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# -----------------------------
# Main execution
# -----------------------------
def main():
    # Step 1: prepare audio windows
    print("=== STEP 1: Generate WAV windows from STEAD ===")
    output_files = prepare_audio_windows(CSV_PATH, HDF5_PATH, OUTPUT_AUDIO_DIR, n_per_class=N_SAMPLES_PER_CLASS)

    # Validate we have files for each class; if not, warn
    for c in CLASS_NAMES:
        print(f"Class '{c}' -> {len(output_files[c])} files")

    # Step 2: extract MFCC features
    print("\n=== STEP 2: Extract MFCC features ===")
    X, y, label_encoder = extract_mfcc_features_from_audio_files(output_files, n_mfcc=N_MFCC, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    print("Feature matrix shape:", X.shape)
    print("Labels shape (one-hot):", y.shape)

    # Step 3: train/test split (stratified)
    print("\n=== STEP 3: Train/Test split ===")
    # convert y back to integer labels for stratify
    y_int = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_int)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # Step 4: build model
    print("\n=== STEP 4: Build and compile CNN ===")
    input_shape = X_train.shape[1:]  # (n_mfcc, n_frames, 1)
    model = build_tall_thin_cnn(input_shape, n_classes=y.shape[1])
    model.summary()

    # Callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    rl = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6)

    # Step 5: train
    print("\n=== STEP 5: Training ===")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rl],
        verbose=2
    )

    # Step 6: evaluate
    print("\n=== STEP 6: Evaluation on test set ===")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predictions and classification report
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_names = list(label_encoder.classes_)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix heatmap
    plot_confusion_matrix(y_true, y_pred, classes=class_names)

if __name__ == "__main__":
    main()
