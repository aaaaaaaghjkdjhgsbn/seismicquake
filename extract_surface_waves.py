#!/usr/bin/env python3
"""
Extract Surface Waves from archive.

Surface waves are the rest of the wave after S-waves until coda end.
"""

import pandas as pd
import numpy as np
import h5py
import re
from pathlib import Path
from tqdm import tqdm

# Configuration
ARCHIVE_DIR = Path("./archive")
OUTPUT_DIR = Path("./extracted_waves/surface_wave")
CSV_FILE = ARCHIVE_DIR / "merge.csv"
HDF5_FILE = ARCHIVE_DIR / "merge.hdf5"

# Wave extraction parameters
SURFACE_WAVE_WINDOW = 400  # samples (4 seconds)

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
        numbers = re.findall(r'[-+]?\d*\.?\d+', value_str)
        if numbers:
            return int(float(numbers[0]))
        
        return None
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

def save_waveform(waveform, idx):
    """Save waveform as .npy file."""
    filename = f"surface_wave_{idx:06d}.npy"
    filepath = OUTPUT_DIR / filename
    np.save(filepath, waveform)
    return filepath

def main(test_mode=False):
    """Extract surface waves from earthquake samples."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load metadata
    print("Loading metadata from CSV...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    
    # Filter earthquake samples only
    eq_samples = df[df['trace_category'] == 'earthquake_local']
    print(f"✓ Found {len(eq_samples):,} earthquake samples")
    
    # Statistics
    extracted = 0
    errors = 0
    
    # Process samples
    limit = 100 if test_mode else len(eq_samples)
    samples_to_process = eq_samples.head(limit)
    
    print(f"\nProcessing {limit:,} earthquake samples...")
    
    with h5py.File(HDF5_FILE, 'r') as h5file:
        dataset = h5file['data']
        
        for idx, row in tqdm(samples_to_process.iterrows(), total=len(samples_to_process), desc="Extracting surface waves"):
            try:
                trace_name = row['trace_name']
                
                # Load waveform using trace name as key
                if trace_name not in dataset:
                    errors += 1
                    if test_mode:
                        print(f"Key not found: {trace_name}")
                    continue
                
                waveform_3ch = dataset[trace_name][()]
                
                # Use first channel (vertical component)
                waveform = waveform_3ch[:, 0]
                
                # Extract surface wave
                surface = extract_surface_wave(
                    waveform,
                    row['s_arrival_sample'],
                    row['coda_end_sample']
                )
                
                if surface is not None:
                    save_waveform(surface, idx)
                    extracted += 1
                else:
                    if test_mode:
                        print(f"Could not extract from idx {idx}: s={row['s_arrival_sample']}, coda={row['coda_end_sample']}")
            
            except Exception as e:
                errors += 1
                if test_mode:
                    print(f"Error at idx {idx}: {e}")
    
    # Print results
    print("\n" + "="*60)
    print("SURFACE WAVE EXTRACTION COMPLETE")
    print("="*60)
    print(f"Surface waves extracted: {extracted:,}")
    print(f"Errors encountered:      {errors:,}")
    print("="*60)
    
    return extracted, errors

if __name__ == "__main__":
    import sys
    
    test_mode = '--test' in sys.argv or '-t' in sys.argv
    
    if test_mode:
        print("="*60)
        print("TEST MODE - Processing first 100 samples only")
        print("="*60)
    
    main(test_mode=test_mode)
    
    print(f"\nFiles saved to: {OUTPUT_DIR}")
