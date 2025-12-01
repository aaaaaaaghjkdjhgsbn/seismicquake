#!/usr/bin/env python3
"""
Extract and classify wave types from archive.

Extracts:
- P-wave segments
- S-wave segments  
- Surface/Coda wave segments (tail of S-wave)
- Noise segments

Saves to organized folders.
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Configuration
ARCHIVE_DIR = Path("./archive")
OUTPUT_DIR = Path("./extracted_waves")
CSV_FILE = ARCHIVE_DIR / "merge.csv"
HDF5_FILE = ARCHIVE_DIR / "merge.hdf5"

# Wave extraction parameters
SAMPLE_RATE = 100  # Hz
P_WAVE_WINDOW = 200   # samples (2 seconds)
S_WAVE_WINDOW = 300   # samples (3 seconds)
SURFACE_WAVE_WINDOW = 400  # samples (4 seconds)
NOISE_WINDOW = 400    # samples (4 seconds)

def load_metadata():
    """Load CSV metadata."""
    print("Loading metadata from CSV...")
    df = pd.read_csv(CSV_FILE)
    print(f"✓ Loaded {len(df)} entries")
    print(f"  - Earthquake samples: {df['trace_category'].value_counts().get('earthquake_local', 0)}")
    print(f"  - Noise samples: {df['trace_category'].value_counts().get('noise', 0)}")
    return df

def parse_sample_value(value):
    """Parse sample values that may be in various formats like [[2896.]] or simple numbers."""
    if pd.isna(value) or value == '' or value == 'None':
        return None
    
    try:
        # If it's already a number, return it
        if isinstance(value, (int, float)):
            return int(value)
        
        # If it's a string, try to extract the number
        value_str = str(value)
        
        # Handle formats like "[[2896.]]" or "[2896]" or "2896.0"
        import re
        numbers = re.findall(r'[-+]?\d*\.?\d+', value_str)
        if numbers:
            return int(float(numbers[0]))
        
        return None
    except (ValueError, TypeError):
        return None

def extract_p_wave(waveform, p_arrival):
    """Extract P-wave segment around arrival time."""
    p_idx = parse_sample_value(p_arrival)
    if p_idx is None:
        return None
    
    try:
        # Extract window centered on P-wave arrival
        start = max(0, p_idx - P_WAVE_WINDOW // 4)
        end = min(len(waveform), start + P_WAVE_WINDOW)
        
        if end - start < P_WAVE_WINDOW // 2:
            return None
            
        return waveform[start:end]
    except (ValueError, TypeError):
        return None

def extract_s_wave(waveform, s_arrival, coda_end=None):
    """Extract S-wave segment (NOT including surface waves)."""
    s_idx = parse_sample_value(s_arrival)
    if s_idx is None:
        return None
    
    try:
        # S-wave ends before surface waves start
        coda_idx = parse_sample_value(coda_end)
        if coda_idx is not None:
            # S-wave is from arrival to midpoint between S and coda
            end = min(len(waveform), s_idx + (coda_idx - s_idx) // 2)
        else:
            end = min(len(waveform), s_idx + S_WAVE_WINDOW)
        
        start = max(0, s_idx - S_WAVE_WINDOW // 8)
        
        if end - start < S_WAVE_WINDOW // 2:
            return None
            
        return waveform[start:end]
    except (ValueError, TypeError):
        return None

def extract_surface_wave(waveform, s_arrival, coda_end):
    """Extract surface wave segment (the rest of the wave after S-wave until coda end)."""
    s_idx = parse_sample_value(s_arrival)
    coda_idx = parse_sample_value(coda_end)
    
    if s_idx is None or coda_idx is None:
        return None
    
    try:
        # Surface waves are the latter part after S-wave (coda region)
        # Start from midpoint between S-wave arrival and coda end
        start = s_idx + (coda_idx - s_idx) // 2
        end = min(len(waveform), coda_idx)
        
        # Ensure we have enough samples
        if end - start < SURFACE_WAVE_WINDOW // 2:
            return None
        
        # Limit to SURFACE_WAVE_WINDOW if the segment is too long
        if end - start > SURFACE_WAVE_WINDOW:
            end = start + SURFACE_WAVE_WINDOW
            
        return waveform[start:end]
    except (ValueError, TypeError):
        return None

def extract_noise(waveform):
    """Extract noise segment from beginning of trace."""
    end = min(len(waveform), NOISE_WINDOW)
    return waveform[:end]

def save_waveform(waveform, output_path, idx, category):
    """Save waveform as .npy file."""
    filename = f"{category}_{idx:06d}.npy"
    filepath = output_path / filename
    np.save(filepath, waveform)
    return filepath

def process_archive(test_mode=False):
    """Process entire archive and extract wave types."""
    
    # Load metadata
    df = load_metadata()
    
    # Create output directories
    output_dirs = {
        'p_wave': OUTPUT_DIR / 'p_wave',
        's_wave': OUTPUT_DIR / 's_wave',
        'surface_wave': OUTPUT_DIR / 'surface_wave',
        'noise': OUTPUT_DIR / 'noise'
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directories created:")
    for name, path in output_dirs.items():
        print(f"  {name}: {path}")
    
    # Statistics
    stats = {
        'p_wave': 0,
        's_wave': 0,
        'surface_wave': 0,
        'noise': 0,
        'errors': 0
    }
    
    # Open HDF5 file
    print(f"\nOpening HDF5 file: {HDF5_FILE}")
    
    # Process samples
    limit = 100 if test_mode else len(df)
    print(f"\nProcessing {limit} samples...")
    
    with h5py.File(HDF5_FILE, 'r') as h5file:
        dataset = h5file['data']
        
        for idx in tqdm(range(limit), desc="Extracting waves"):
            try:
                row = df.iloc[idx]
                trace_name = row['trace_name']
                
                # Load waveform using trace name as key (shape: 6000, 3 channels)
                if trace_name not in dataset:
                    stats['errors'] += 1
                    if test_mode:
                        print(f"Error processing {trace_name}: Key not found in HDF5")
                    continue
                
                waveform_3ch = dataset[trace_name][()]
                
                # Use first channel (vertical component typically)
                waveform = waveform_3ch[:, 0]
                
                trace_category = row['trace_category']
                
                if trace_category == 'earthquake_local':
                    # Extract P-wave
                    p_wave = extract_p_wave(waveform, row.get('p_arrival_sample'))
                    if p_wave is not None:
                        save_waveform(p_wave, output_dirs['p_wave'], idx, 'p_wave')
                        stats['p_wave'] += 1
                    
                    # Extract S-wave
                    s_wave = extract_s_wave(
                        waveform, 
                        row.get('s_arrival_sample'),
                        row.get('coda_end_sample')
                    )
                    if s_wave is not None:
                        save_waveform(s_wave, output_dirs['s_wave'], idx, 's_wave')
                        stats['s_wave'] += 1
                    
                    # Extract Surface wave (coda)
                    surface = extract_surface_wave(
                        waveform,
                        row.get('s_arrival_sample'),
                        row.get('coda_end_sample')
                    )
                    if surface is not None:
                        save_waveform(surface, output_dirs['surface_wave'], idx, 'surface_wave')
                        stats['surface_wave'] += 1
                
                elif trace_category == 'noise':
                    # Extract noise
                    noise = extract_noise(waveform)
                    if noise is not None:
                        save_waveform(noise, output_dirs['noise'], idx, 'noise')
                        stats['noise'] += 1
            
            except Exception as e:
                stats['errors'] += 1
                if test_mode:
                    print(f"\nError processing index {idx}: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"P-waves extracted:       {stats['p_wave']:,}")
    print(f"S-waves extracted:       {stats['s_wave']:,}")
    print(f"Surface waves extracted: {stats['surface_wave']:,}")
    print(f"Noise samples extracted: {stats['noise']:,}")
    print(f"Errors encountered:      {stats['errors']:,}")
    print(f"\nTotal files created:     {sum(stats.values()) - stats['errors']:,}")
    print("="*60)
    
    return stats

if __name__ == "__main__":
    import sys
    
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    
    if test_mode:
        print("="*60)
        print("TEST MODE - Processing first 100 samples only")
        print("="*60)
    
    stats = process_archive(test_mode=test_mode)
    
    print("\nExtraction complete! Files saved to:", OUTPUT_DIR)
