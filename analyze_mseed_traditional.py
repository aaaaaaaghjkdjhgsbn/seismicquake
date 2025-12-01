#!/usr/bin/env python3
"""
MiniSEED Wave Analyzer - Traditional Seismology Methods

Uses classical signal processing to detect and extract P, S, and Surface waves
from MiniSEED files when AI models don't perform well.

Usage:
    python3 analyze_mseed_traditional.py input.mseed
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from obspy import read
    from obspy.signal.trigger import classic_sta_lta, trigger_onset, plot_trigger
    from obspy.signal.filter import bandpass
except ImportError:
    print("Error: obspy is required. Install with: pip install obspy")
    sys.exit(1)

def analyze_mseed_file(mseed_path, output_dir="wave_analysis"):
    """
    Analyze MiniSEED file using traditional seismology methods.
    
    Args:
        mseed_path: Path to MiniSEED file
        output_dir: Directory for output files
    """
    print(f"{'='*80}")
    print(f"TRADITIONAL SEISMOLOGY ANALYSIS")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load MiniSEED file
    print(f"Loading: {mseed_path}")
    stream = read(mseed_path)
    trace = stream[0]  # Use first trace
    
    # Print metadata
    print(f"\nTrace Information:")
    print(f"  Station: {trace.stats.station}")
    print(f"  Channel: {trace.stats.channel}")
    print(f"  Sample Rate: {trace.stats.sampling_rate} Hz")
    print(f"  Duration: {trace.stats.endtime - trace.stats.starttime} seconds")
    print(f"  Start Time: {trace.stats.starttime}")
    print(f"  Samples: {trace.stats.npts}")
    print(f"  Data Range: [{trace.data.min():.2f}, {trace.data.max():.2f}]")
    
    # Apply preprocessing
    print(f"\nPreprocessing...")
    
    # 1. Remove trend
    trace.detrend('linear')
    print(f"  ✓ Detrended")
    
    # 2. Apply bandpass filter (typical seismic frequencies)
    trace_filtered = trace.copy()
    try:
        trace_filtered.filter('bandpass', freqmin=0.5, freqmax=20.0, corners=4, zerophase=True)
        print(f"  ✓ Bandpass filtered (0.5-20 Hz)")
    except:
        print(f"  ⚠ Filtering failed, using original")
        trace_filtered = trace.copy()
    
    # 3. Calculate STA/LTA for automatic picking
    print(f"\nCalculating STA/LTA (Short-Term Average / Long-Term Average)...")
    sr = trace_filtered.stats.sampling_rate
    
    # STA/LTA parameters
    sta_len = 0.5  # seconds
    lta_len = 10.0  # seconds
    
    cft = classic_sta_lta(trace_filtered.data, int(sta_len * sr), int(lta_len * sr))
    print(f"  STA window: {sta_len}s")
    print(f"  LTA window: {lta_len}s")
    
    # Detect triggers (wave arrivals)
    print(f"\nDetecting wave arrivals...")
    threshold_on = 3.5   # Trigger on
    threshold_off = 1.0  # Trigger off
    
    triggers = trigger_onset(cft, threshold_on, threshold_off)
    print(f"  Threshold ON: {threshold_on}")
    print(f"  Threshold OFF: {threshold_off}")
    print(f"  Detected {len(triggers)} potential wave arrivals")
    
    # Identify wave types based on arrival times
    wave_arrivals = []
    for i, (on, off) in enumerate(triggers):
        time_on = on / sr
        time_off = off / sr
        duration = time_off - time_on
        
        # Classify based on characteristics
        if i == 0 and duration < 10:
            wave_type = "P-wave"
        elif i == 1 and duration < 15:
            wave_type = "S-wave"
        elif duration > 10:
            wave_type = "Surface/Coda"
        else:
            wave_type = "Unknown"
        
        arrival = {
            'type': wave_type,
            'start_time': time_on,
            'end_time': time_off,
            'duration': duration,
            'start_sample': on,
            'end_sample': off
        }
        wave_arrivals.append(arrival)
        
        print(f"\n  Arrival {i+1}:")
        print(f"    Type: {wave_type}")
        print(f"    Time: {time_on:.2f} - {time_off:.2f} seconds")
        print(f"    Duration: {duration:.2f} seconds")
    
    # Create comprehensive visualization
    print(f"\nCreating visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Original waveform
    ax1 = plt.subplot(4, 1, 1)
    time_axis = np.arange(trace.stats.npts) / sr
    ax1.plot(time_axis, trace.data, 'k-', linewidth=0.5)
    ax1.set_ylabel('Amplitude\n(Original)', fontweight='bold')
    ax1.set_title(f'MiniSEED Analysis - {Path(mseed_path).name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Mark detected arrivals
    colors = {'P-wave': 'red', 'S-wave': 'blue', 'Surface/Coda': 'green', 'Unknown': 'gray'}
    for arrival in wave_arrivals:
        color = colors.get(arrival['type'], 'gray')
        ax1.axvspan(arrival['start_time'], arrival['end_time'], alpha=0.3, color=color, label=arrival['type'])
    
    if wave_arrivals:
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # 2. Filtered waveform
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(time_axis, trace_filtered.data, 'b-', linewidth=0.5)
    ax2.set_ylabel('Amplitude\n(Filtered 0.5-20Hz)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark detected arrivals
    for arrival in wave_arrivals:
        color = colors.get(arrival['type'], 'gray')
        ax2.axvspan(arrival['start_time'], arrival['end_time'], alpha=0.3, color=color)
        # Add label
        mid_time = (arrival['start_time'] + arrival['end_time']) / 2
        ax2.text(mid_time, ax2.get_ylim()[1] * 0.9, arrival['type'], 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. STA/LTA characteristic function
    ax3 = plt.subplot(4, 1, 3)
    cft_time = np.arange(len(cft)) / sr
    ax3.plot(cft_time, cft, 'g-', linewidth=1)
    ax3.axhline(threshold_on, color='red', linestyle='--', linewidth=2, label=f'Trigger ON ({threshold_on})')
    ax3.axhline(threshold_off, color='blue', linestyle='--', linewidth=2, label=f'Trigger OFF ({threshold_off})')
    ax3.set_ylabel('STA/LTA Ratio', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Mark trigger regions
    for on, off in triggers:
        ax3.axvspan(on/sr, off/sr, alpha=0.2, color='yellow')
    
    # 4. Spectrogram
    ax4 = plt.subplot(4, 1, 4)
    Pxx, freqs, bins, im = ax4.specgram(trace_filtered.data, Fs=sr, NFFT=256, 
                                         noverlap=128, cmap='viridis')
    ax4.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax4.set_xlabel('Time (seconds)', fontweight='bold')
    ax4.set_ylim([0, 25])  # Focus on seismic frequencies
    plt.colorbar(im, ax=ax4, label='Power (dB)')
    
    # Mark detected arrivals
    for arrival in wave_arrivals:
        color = colors.get(arrival['type'], 'gray')
        ax4.axvspan(arrival['start_time'], arrival['end_time'], alpha=0.2, color=color)
    
    plt.tight_layout()
    
    # Save plot
    output_plot = output_dir / f"{Path(mseed_path).stem}_analysis.png"
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved plot: {output_plot}")
    
    plt.close()
    
    # Extract and save individual wave segments
    print(f"\nExtracting wave segments...")
    
    for i, arrival in enumerate(wave_arrivals):
        # Extract segment
        start_sample = max(0, arrival['start_sample'] - int(2 * sr))  # Include 2s before
        end_sample = min(len(trace_filtered.data), arrival['end_sample'] + int(2 * sr))  # Include 2s after
        
        segment = trace_filtered.slice(
            starttime=trace_filtered.stats.starttime + start_sample / sr,
            endtime=trace_filtered.stats.starttime + end_sample / sr
        )
        
        # Save as MiniSEED
        wave_type_safe = arrival['type'].replace('/', '_').replace(' ', '_')
        output_mseed = output_dir / f"{Path(mseed_path).stem}_{wave_type_safe}_{i+1}.mseed"
        segment.write(str(output_mseed), format='MSEED')
        print(f"  ✓ Saved {arrival['type']} segment: {output_mseed}")
        
        # Also save as WAV for universal compatibility
        output_wav = output_dir / f"{Path(mseed_path).stem}_{wave_type_safe}_{i+1}.wav"
        
        # Normalize and convert to 16-bit
        data_normalized = segment.data / (np.abs(segment.data).max() + 1e-8)
        data_16bit = (data_normalized * 32767).astype(np.int16)
        
        import soundfile as sf
        sf.write(str(output_wav), data_normalized, int(segment.stats.sampling_rate), subtype='PCM_16')
        print(f"  ✓ Saved WAV: {output_wav}")
    
    # Generate report
    print(f"\nGenerating report...")
    report_path = output_dir / f"{Path(mseed_path).stem}_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SEISMIC WAVE ANALYSIS REPORT (Traditional Methods)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"File: {mseed_path}\n")
        f.write(f"Analysis Date: {trace.stats.starttime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRACE INFORMATION:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Station: {trace.stats.station}\n")
        f.write(f"  Channel: {trace.stats.channel}\n")
        f.write(f"  Sample Rate: {trace.stats.sampling_rate} Hz\n")
        f.write(f"  Duration: {trace.stats.endtime - trace.stats.starttime} seconds\n")
        f.write(f"  Samples: {trace.stats.npts}\n\n")
        
        f.write("DETECTED WAVE ARRIVALS:\n")
        f.write("-"*80 + "\n")
        
        if wave_arrivals:
            for i, arrival in enumerate(wave_arrivals, 1):
                f.write(f"\nArrival {i}: {arrival['type']}\n")
                f.write(f"  Time Window: {arrival['start_time']:.2f} - {arrival['end_time']:.2f} seconds\n")
                f.write(f"  Duration: {arrival['duration']:.2f} seconds\n")
        else:
            f.write("\nNo clear wave arrivals detected.\n")
            f.write("This could indicate:\n")
            f.write("  - Very low signal-to-noise ratio\n")
            f.write("  - No seismic event in this time window\n")
            f.write("  - Need to adjust STA/LTA parameters\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS METHOD:\n")
        f.write("-"*80 + "\n")
        f.write(f"  Method: Classic STA/LTA (Short-Term Average / Long-Term Average)\n")
        f.write(f"  STA Window: {sta_len} seconds\n")
        f.write(f"  LTA Window: {lta_len} seconds\n")
        f.write(f"  Trigger ON Threshold: {threshold_on}\n")
        f.write(f"  Trigger OFF Threshold: {threshold_off}\n")
        f.write(f"  Bandpass Filter: 0.5-20 Hz\n\n")
        
        f.write("WAVE CLASSIFICATION LOGIC:\n")
        f.write("-"*80 + "\n")
        f.write("  - First arrival (short duration): P-wave\n")
        f.write("  - Second arrival (moderate duration): S-wave\n")
        f.write("  - Long duration arrivals: Surface/Coda waves\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"  ✓ Saved report: {report_path}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput files saved to: {output_dir}/")
    print(f"  - Analysis plot (PNG)")
    print(f"  - Wave segments (MiniSEED + WAV)")
    print(f"  - Text report")
    
    return wave_arrivals

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_mseed_traditional.py <input.mseed> [output_dir]")
        print("\nExample:")
        print("  python3 analyze_mseed_traditional.py earthquake.mseed")
        print("  python3 analyze_mseed_traditional.py earthquake.mseed my_analysis/")
        sys.exit(1)
    
    mseed_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "wave_analysis"
    
    if not Path(mseed_path).exists():
        print(f"Error: File not found: {mseed_path}")
        sys.exit(1)
    
    try:
        analyze_mseed_file(mseed_path, output_dir)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
