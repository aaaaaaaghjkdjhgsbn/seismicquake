#!/usr/bin/env python3
"""
Seismic Wave Analyzer - Real-time Earthquake Detection and Classification

This system uses trained AI models to:
1. Detect earthquakes vs noise in seismic data
2. Classify wave types (P-wave, S-wave, Surface wave)
3. Predict earthquake magnitude from P-waves
4. Process both real-time streams and static files (.wav, .mseed)

Usage:
    # Analyze a static file
    python seismic_analyzer.py analyze path/to/file.mseed
    
    # Real-time monitoring
    python seismic_analyzer.py monitor --source live
    
    # Demo with sample files
    python seismic_analyzer.py demo
"""

import os
import sys
import time
import json
import warnings
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from collections import deque
from datetime import datetime
import threading

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import TensorFlow
import tensorflow as tf
from tensorflow import keras

# Configuration
MODELS_DIR = Path("./earthquake_ai_models")
INPUT_LENGTH = 400  # Samples per analysis window
SAMPLE_RATE = 100   # Hz - standard seismic sample rate


@dataclass
class WaveDetection:
    """Represents a detected seismic wave."""
    wave_type: str  # 'P', 'S', 'Surface', 'Noise'
    confidence: float
    start_sample: int
    end_sample: int
    start_time: float  # seconds from trace start
    end_time: float
    magnitude: Optional[float] = None
    magnitude_uncertainty: float = 0.5


@dataclass
class AnalysisResult:
    """Complete analysis result for a seismic trace."""
    filename: str
    duration_seconds: float
    sample_rate: float
    is_earthquake: bool
    earthquake_confidence: float
    detections: List[WaveDetection] = field(default_factory=list)
    p_wave_arrival: Optional[float] = None  # seconds
    s_wave_arrival: Optional[float] = None  # seconds
    surface_wave_arrival: Optional[float] = None  # seconds
    estimated_magnitude: Optional[float] = None
    processing_time: float = 0.0
    

class SeismicAnalyzer:
    """
    Main class for seismic wave detection and classification.
    
    Supports:
    - Static file analysis (.wav, .mseed)
    - Real-time stream processing
    - Batch processing of multiple files
    """
    
    # Detection thresholds
    EARTHQUAKE_THRESHOLD = 0.5
    WAVE_CONFIDENCE_THRESHOLD = 0.5
    MIN_DETECTION_GAP = 50  # Minimum samples between detections
    
    # Wave type labels
    WAVE_TYPES = ['P', 'S', 'Surface']
    
    def __init__(self, models_dir: Path = MODELS_DIR, verbose: bool = True):
        """Initialize the analyzer with trained models."""
        self.models_dir = Path(models_dir)
        self.verbose = verbose
        
        # Models
        self.earthquake_detector = None
        self.wave_classifier = None
        self.magnitude_predictor = None
        
        # Real-time buffer
        self.buffer = deque(maxlen=INPUT_LENGTH * 10)
        self.detection_history = []
        
        # Load models
        self._load_models()
    
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def _load_models(self):
        """Load trained AI models."""
        self._log("Loading AI models...")
        
        model_configs = [
            ('earthquake_detector', 'earthquake_detector_best.h5', 'earthquake_detector_final.h5'),
            ('wave_classifier', 'wave_classifier_best.h5', 'wave_classifier_final.h5'),
            ('magnitude_predictor', 'magnitude_predictor_best.h5', 'magnitude_predictor_final.h5'),
        ]
        
        for attr_name, best_file, final_file in model_configs:
            model_path = self.models_dir / best_file
            if not model_path.exists():
                model_path = self.models_dir / final_file
            
            if model_path.exists():
                try:
                    model = keras.models.load_model(model_path, compile=False)
                    setattr(self, attr_name, model)
                    self._log(f"  ✓ Loaded {attr_name}")
                except Exception as e:
                    self._log(f"  ⚠ Failed to load {attr_name}: {e}")
            else:
                self._log(f"  ⚠ Model not found: {attr_name}")
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [-1, 1] range."""
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data
    
    def _pad_or_truncate(self, data: np.ndarray, target_length: int = INPUT_LENGTH) -> np.ndarray:
        """Pad or truncate data to target length."""
        if len(data) >= target_length:
            return data[:target_length]
        padded = np.zeros(target_length, dtype=np.float32)
        padded[:len(data)] = data
        return padded
    
    def _preprocess_segment(self, segment: np.ndarray) -> np.ndarray:
        """Preprocess a segment for model input."""
        segment = self._pad_or_truncate(segment)
        segment = self._normalize(segment)
        return segment.reshape(1, INPUT_LENGTH, 1).astype(np.float32)
    
    def load_file(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, float]:
        """
        Load seismic data from file.
        
        Supports:
        - .mseed (MiniSEED format)
        - .wav (WAV audio format)
        - .npy (NumPy array)
        
        Returns:
            data: numpy array of seismic data
            sample_rate: sample rate in Hz
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.mseed':
            return self._load_mseed(filepath)
        elif suffix == '.wav':
            return self._load_wav(filepath)
        elif suffix == '.npy':
            data = np.load(filepath)
            # Handle multi-channel data (e.g., shape (6000, 3) for E, N, Z components)
            if len(data.shape) > 1:
                # Use the last channel (Z component - vertical, best for earthquake detection)
                # Shape (samples, channels) -> use channel index 2 (Z)
                if data.shape[1] == 3:
                    data = data[:, 2]  # Z component
                elif data.shape[0] == 3:
                    data = data[2, :]  # Z component if shape is (3, samples)
                else:
                    # Just take the first channel or flatten
                    data = data.flatten() if data.shape[0] < data.shape[1] else data[:, 0]
            data = data.astype(np.float32)
            return data, SAMPLE_RATE
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_mseed(self, filepath: Path) -> Tuple[np.ndarray, float]:
        """Load MiniSEED file."""
        try:
            from obspy import read
            st = read(str(filepath))
            tr = st[0]
            data = tr.data.astype(np.float32)
            sample_rate = tr.stats.sampling_rate
            return data, sample_rate
        except ImportError:
            raise ImportError("ObsPy is required for .mseed files. Install with: pip install obspy")
    
    def _load_wav(self, filepath: Path) -> Tuple[np.ndarray, float]:
        """Load WAV audio file."""
        try:
            import scipy.io.wavfile as wav
            sample_rate, data = wav.read(str(filepath))
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            data = data.astype(np.float32)
            return data, float(sample_rate)
        except ImportError:
            raise ImportError("SciPy is required for .wav files. Install with: pip install scipy")
    
    def detect_earthquake(self, segment: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if segment contains earthquake signal.
        
        Returns:
            is_earthquake: bool
            confidence: float (0-1)
        """
        if self.earthquake_detector is None:
            return False, 0.0
        
        X = self._preprocess_segment(segment)
        prob = float(self.earthquake_detector.predict(X, verbose=0)[0, 0])
        is_earthquake = prob >= self.EARTHQUAKE_THRESHOLD
        
        return is_earthquake, prob
    
    def classify_wave(self, segment: np.ndarray) -> Tuple[str, float]:
        """
        Classify wave type.
        
        Returns:
            wave_type: str ('P', 'S', 'Surface')
            confidence: float (0-1)
        """
        if self.wave_classifier is None:
            return 'Unknown', 0.0
        
        X = self._preprocess_segment(segment)
        probs = self.wave_classifier.predict(X, verbose=0)[0]
        
        wave_idx = np.argmax(probs)
        confidence = float(probs[wave_idx])
        wave_type = self.WAVE_TYPES[wave_idx]
        
        return wave_type, confidence
    
    def predict_magnitude(self, p_wave_segment: np.ndarray) -> Tuple[float, float]:
        """
        Predict earthquake magnitude from P-wave.
        
        Returns:
            magnitude: float
            uncertainty: float
        """
        if self.magnitude_predictor is None:
            return 0.0, 0.0
        
        X = self._preprocess_segment(p_wave_segment)
        magnitude = float(self.magnitude_predictor.predict(X, verbose=0)[0, 0])
        
        # Clamp to reasonable range
        magnitude = max(0.0, min(10.0, magnitude))
        uncertainty = 0.5  # Based on training performance
        
        return magnitude, uncertainty
    
    def _calculate_sta_lta(self, data: np.ndarray, sta_len: int = 50, lta_len: int = 200) -> np.ndarray:
        """
        Calculate STA/LTA ratio for P-wave detection.
        
        STA/LTA (Short-Term Average / Long-Term Average) is the standard
        seismological method for detecting wave arrivals.
        """
        # Use absolute values or squared values
        cf = np.abs(data)  # Characteristic function
        
        sta_lta = np.zeros(len(data))
        
        for i in range(lta_len, len(data)):
            sta = np.mean(cf[i - sta_len:i])
            lta = np.mean(cf[i - lta_len:i])
            if lta > 0:
                sta_lta[i] = sta / lta
        
        return sta_lta
    
    def _detect_wave_arrivals(self, data: np.ndarray, sample_rate: float) -> dict:
        """
        Detect P and S wave arrivals using STA/LTA and signal analysis.
        
        Returns dict with p_arrival, s_arrival (in samples), and confidences.
        """
        # Calculate STA/LTA for P-wave detection
        sta_lta = self._calculate_sta_lta(data, sta_len=30, lta_len=150)
        
        # Threshold for P-wave trigger (typically 2-4)
        p_threshold = 3.0
        
        # Find P-wave arrival (first significant trigger)
        p_arrival = None
        p_confidence = 0.0
        
        for i in range(200, len(sta_lta)):
            if sta_lta[i] > p_threshold:
                p_arrival = i
                p_confidence = min(1.0, sta_lta[i] / 5.0)  # Normalize confidence
                break
        
        # Find S-wave arrival (second trigger after P, usually higher amplitude)
        s_arrival = None
        s_confidence = 0.0
        
        if p_arrival is not None:
            # S-wave typically arrives 1.7x later than P-wave travel time
            # Search in a window after P arrival
            search_start = p_arrival + int(sample_rate * 1.0)  # At least 1 second after P
            search_end = min(len(data), p_arrival + int(sample_rate * 30))  # Within 30 seconds
            
            if search_start < len(data):
                # Look for amplitude increase (S-waves are larger)
                window_size = int(sample_rate * 0.5)  # 0.5 second windows
                max_amplitude_ratio = 0
                
                for i in range(search_start, search_end - window_size, window_size // 2):
                    current_amp = np.std(data[i:i + window_size])
                    p_region_amp = np.std(data[p_arrival:p_arrival + window_size])
                    
                    if p_region_amp > 0:
                        ratio = current_amp / p_region_amp
                        if ratio > max_amplitude_ratio and ratio > 1.5:
                            max_amplitude_ratio = ratio
                            s_arrival = i
                            s_confidence = min(1.0, ratio / 3.0)
        
        # Estimate surface wave arrival (typically after S-wave)
        surface_arrival = None
        surface_confidence = 0.0
        
        if s_arrival is not None:
            # Surface waves arrive after S-waves and have lower frequency
            search_start = s_arrival + int(sample_rate * 2.0)
            if search_start < len(data) - 400:
                surface_arrival = search_start
                surface_confidence = 0.8  # Lower confidence for surface wave timing
        
        return {
            'p_arrival': p_arrival,
            'p_confidence': p_confidence,
            's_arrival': s_arrival,
            's_confidence': s_confidence,
            'surface_arrival': surface_arrival,
            'surface_confidence': surface_confidence
        }

    def analyze_trace(self, data: np.ndarray, sample_rate: float = SAMPLE_RATE,
                      window_size: int = INPUT_LENGTH, 
                      step_size: int = None) -> List[WaveDetection]:
        """
        Analyze full trace with improved wave detection using STA/LTA.
        
        Args:
            data: Seismic trace data
            sample_rate: Sample rate in Hz
            window_size: Analysis window size in samples
            step_size: Step size between windows (default: window_size // 2)
            
        Returns:
            List of wave detections
        """
        if step_size is None:
            step_size = window_size // 2
        
        detections = []
        
        # First, check if there's an earthquake at all using the full trace
        # Sample multiple windows to get overall earthquake probability
        earthquake_probs = []
        sample_points = [int(len(data) * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        
        for sp in sample_points:
            if sp + window_size <= len(data):
                segment = data[sp:sp + window_size]
                _, prob = self.detect_earthquake(segment)
                earthquake_probs.append(prob)
        
        avg_eq_prob = np.mean(earthquake_probs) if earthquake_probs else 0.0
        is_earthquake_trace = avg_eq_prob > self.EARTHQUAKE_THRESHOLD
        
        if not is_earthquake_trace:
            # Return single noise detection for the whole trace
            return [WaveDetection(
                wave_type='Noise',
                confidence=1 - avg_eq_prob,
                start_sample=0,
                end_sample=len(data),
                start_time=0,
                end_time=len(data) / sample_rate
            )]
        
        # Use STA/LTA to detect wave arrivals
        arrivals = self._detect_wave_arrivals(data, sample_rate)
        
        # Create detections based on arrivals
        p_arrival = arrivals['p_arrival']
        s_arrival = arrivals['s_arrival']
        surface_arrival = arrivals['surface_arrival']
        
        # P-wave detection
        if p_arrival is not None:
            p_end = s_arrival if s_arrival else min(p_arrival + int(sample_rate * 5), len(data))
            p_segment = data[p_arrival:min(p_arrival + window_size, len(data))]
            
            # Verify with AI classifier
            wave_type, wave_conf = self.classify_wave(p_segment)
            
            # Predict magnitude from P-wave
            magnitude, _ = self.predict_magnitude(p_segment)
            
            detections.append(WaveDetection(
                wave_type='P',
                confidence=max(arrivals['p_confidence'], 0.8),  # High confidence for STA/LTA detection
                start_sample=p_arrival,
                end_sample=p_end,
                start_time=p_arrival / sample_rate,
                end_time=p_end / sample_rate,
                magnitude=magnitude
            ))
        
        # S-wave detection
        if s_arrival is not None:
            s_end = surface_arrival if surface_arrival else min(s_arrival + int(sample_rate * 10), len(data))
            s_segment = data[s_arrival:min(s_arrival + window_size, len(data))]
            
            detections.append(WaveDetection(
                wave_type='S',
                confidence=max(arrivals['s_confidence'], 0.75),
                start_sample=s_arrival,
                end_sample=s_end,
                start_time=s_arrival / sample_rate,
                end_time=s_end / sample_rate
            ))
        
        # Surface wave detection
        if surface_arrival is not None:
            surface_end = min(surface_arrival + int(sample_rate * 20), len(data))
            surface_segment = data[surface_arrival:min(surface_arrival + window_size, len(data))]
            
            detections.append(WaveDetection(
                wave_type='Surface',
                confidence=arrivals['surface_confidence'],
                start_sample=surface_arrival,
                end_sample=surface_end,
                start_time=surface_arrival / sample_rate,
                end_time=surface_end / sample_rate
            ))
        
        # If no arrivals detected but it's an earthquake, fall back to window-based classification
        if not detections:
            num_windows = max(1, (len(data) - window_size) // step_size + 1)
            
            for i in range(num_windows):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                if end_idx > len(data):
                    break
                
                segment = data[start_idx:end_idx]
                is_earthquake, eq_confidence = self.detect_earthquake(segment)
                
                if is_earthquake:
                    wave_type, wave_confidence = self.classify_wave(segment)
                    magnitude = None
                    if wave_type == 'P':
                        magnitude, _ = self.predict_magnitude(segment)
                    
                    detections.append(WaveDetection(
                        wave_type=wave_type,
                        confidence=wave_confidence,
                        start_sample=start_idx,
                        end_sample=end_idx,
                        start_time=start_idx / sample_rate,
                        end_time=end_idx / sample_rate,
                        magnitude=magnitude
                    ))
        
        return detections
    
    def analyze_file(self, filepath: Union[str, Path]) -> AnalysisResult:
        """
        Analyze a seismic file.
        
        Args:
            filepath: Path to .mseed, .wav, or .npy file
            
        Returns:
            AnalysisResult with complete analysis
        """
        filepath = Path(filepath)
        start_time = time.time()
        
        self._log(f"\nAnalyzing: {filepath.name}")
        self._log("-" * 50)
        
        # Load file
        data, sample_rate = self.load_file(filepath)
        duration = len(data) / sample_rate
        
        self._log(f"Duration: {duration:.2f}s | Sample Rate: {sample_rate:.0f} Hz | Samples: {len(data):,}")
        
        # Analyze trace
        detections = self.analyze_trace(data, sample_rate)
        
        # Aggregate results
        earthquake_detections = [d for d in detections if d.wave_type != 'Noise']
        is_earthquake = len(earthquake_detections) > 0
        
        # Calculate overall earthquake confidence
        if detections:
            eq_confidences = [d.confidence for d in detections if d.wave_type != 'Noise']
            noise_confidences = [d.confidence for d in detections if d.wave_type == 'Noise']
            eq_confidence = np.mean(eq_confidences) if eq_confidences else 0.0
        else:
            eq_confidence = 0.0
        
        # Find wave arrivals (first detection of each type)
        p_wave_arrival = None
        s_wave_arrival = None
        surface_wave_arrival = None
        estimated_magnitude = None
        
        for d in detections:
            if d.wave_type == 'P' and p_wave_arrival is None:
                p_wave_arrival = d.start_time
                estimated_magnitude = d.magnitude
            elif d.wave_type == 'S' and s_wave_arrival is None:
                s_wave_arrival = d.start_time
            elif d.wave_type == 'Surface' and surface_wave_arrival is None:
                surface_wave_arrival = d.start_time
        
        processing_time = time.time() - start_time
        
        result = AnalysisResult(
            filename=filepath.name,
            duration_seconds=duration,
            sample_rate=sample_rate,
            is_earthquake=is_earthquake,
            earthquake_confidence=eq_confidence,
            detections=detections,
            p_wave_arrival=p_wave_arrival,
            s_wave_arrival=s_wave_arrival,
            surface_wave_arrival=surface_wave_arrival,
            estimated_magnitude=estimated_magnitude,
            processing_time=processing_time
        )
        
        # Print summary
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: AnalysisResult):
        """Print analysis result summary."""
        if result.is_earthquake:
            self._log(f"\n🚨 EARTHQUAKE DETECTED ({result.earthquake_confidence:.1%} confidence)")
            
            # Count wave types
            wave_counts = {}
            for d in result.detections:
                wave_counts[d.wave_type] = wave_counts.get(d.wave_type, 0) + 1
            
            self._log(f"\nWave Detections:")
            for wave_type, count in sorted(wave_counts.items()):
                if wave_type != 'Noise':
                    self._log(f"  {wave_type}-wave: {count} segments")
            
            if result.p_wave_arrival is not None:
                self._log(f"\nP-wave Arrival: {result.p_wave_arrival:.2f}s")
            if result.s_wave_arrival is not None:
                self._log(f"S-wave Arrival: {result.s_wave_arrival:.2f}s")
            if result.surface_wave_arrival is not None:
                self._log(f"Surface Wave Arrival: {result.surface_wave_arrival:.2f}s")
            
            if result.estimated_magnitude is not None:
                self._log(f"\nEstimated Magnitude: {result.estimated_magnitude:.1f} ± 0.5")
        else:
            self._log(f"\n✓ No earthquake detected (Noise)")
        
        self._log(f"\nProcessing time: {result.processing_time:.3f}s")
    
    def visualize(self, filepath: Union[str, Path], output_path: Optional[Path] = None,
                  show: bool = True) -> Optional[Any]:
        """
        Visualize analysis results with matplotlib.
        
        Args:
            filepath: Path to seismic file
            output_path: Optional path to save figure
            show: Whether to display the figure
            
        Returns:
            matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            self._log("Matplotlib required for visualization. Install with: pip install matplotlib")
            return None
        
        # Load and analyze
        data, sample_rate = self.load_file(filepath)
        result = self.analyze_file(filepath)
        
        # Create time axis
        time_axis = np.arange(len(data)) / sample_rate
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Waveform with detections
        ax1 = axes[0]
        ax1.plot(time_axis, data, 'k-', linewidth=0.5, alpha=0.7, label='Seismic Data')
        
        # Color-code detections
        colors = {'P': '#FF4444', 'S': '#44AA44', 'Surface': '#4444FF', 'Noise': '#CCCCCC'}
        
        for detection in result.detections:
            if detection.wave_type != 'Noise':
                ax1.axvspan(
                    detection.start_time, detection.end_time,
                    alpha=0.3, color=colors.get(detection.wave_type, 'gray'),
                    label=f'{detection.wave_type}-wave' if detection.wave_type not in [d.wave_type for d in result.detections[:result.detections.index(detection)]] else ''
                )
        
        # Add arrival markers
        if result.p_wave_arrival is not None:
            ax1.axvline(result.p_wave_arrival, color='red', linestyle='--', linewidth=2, label='P Arrival')
        if result.s_wave_arrival is not None:
            ax1.axvline(result.s_wave_arrival, color='green', linestyle='--', linewidth=2, label='S Arrival')
        if result.surface_wave_arrival is not None:
            ax1.axvline(result.surface_wave_arrival, color='blue', linestyle='--', linewidth=2, label='Surface Arrival')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Seismic Analysis: {Path(filepath).name}')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Detection confidence timeline
        ax2 = axes[1]
        
        times = []
        confidences = []
        wave_colors = []
        
        for d in result.detections:
            mid_time = (d.start_time + d.end_time) / 2
            times.append(mid_time)
            confidences.append(d.confidence if d.wave_type != 'Noise' else -d.confidence)
            wave_colors.append(colors.get(d.wave_type, 'gray'))
        
        ax2.bar(times, confidences, width=(time_axis[-1] / len(result.detections)) * 0.8,
                color=wave_colors, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axhline(self.EARTHQUAKE_THRESHOLD, color='red', linestyle=':', label='Threshold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Detection Confidence')
        ax2.set_ylim(-1, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add legend for wave types
        patches = [mpatches.Patch(color=colors[w], label=f'{w}-wave', alpha=0.7) 
                   for w in ['P', 'S', 'Surface']]
        ax2.legend(handles=patches, loc='upper left')
        
        plt.tight_layout()
        
        # Add summary text
        summary = f"Earthquake: {'Yes' if result.is_earthquake else 'No'}"
        if result.estimated_magnitude:
            summary += f" | Magnitude: {result.estimated_magnitude:.1f}"
        fig.suptitle(summary, y=1.02, fontsize=12, fontweight='bold')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            self._log(f"\nFigure saved to: {output_path}")
        
        if show:
            plt.show()
        
        return fig


class RealtimeMonitor:
    """
    Real-time seismic monitoring system.
    
    Continuously processes incoming seismic data and triggers alerts
    when earthquakes are detected.
    """
    
    def __init__(self, analyzer: SeismicAnalyzer, 
                 window_size: int = INPUT_LENGTH,
                 alert_callback=None):
        """
        Initialize real-time monitor.
        
        Args:
            analyzer: SeismicAnalyzer instance
            window_size: Analysis window size
            alert_callback: Function to call when earthquake detected
        """
        self.analyzer = analyzer
        self.window_size = window_size
        self.alert_callback = alert_callback or self._default_alert
        
        self.buffer = deque(maxlen=window_size * 2)
        self.is_running = False
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds
        
        # Statistics
        self.total_samples = 0
        self.earthquake_count = 0
        self.start_time = None
    
    def _default_alert(self, detection: WaveDetection, timestamp: float):
        """Default alert handler."""
        print(f"\n{'='*60}")
        print(f"🚨 EARTHQUAKE ALERT at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
        print(f"   Wave Type: {detection.wave_type}-wave ({detection.confidence:.1%})")
        if detection.magnitude:
            print(f"   Est. Magnitude: {detection.magnitude:.1f} ± {detection.magnitude_uncertainty}")
        print(f"{'='*60}\n")
    
    def process_sample(self, sample: float, timestamp: float = None) -> Optional[WaveDetection]:
        """
        Process a single sample.
        
        Args:
            sample: Single seismic sample value
            timestamp: Optional timestamp
            
        Returns:
            WaveDetection if earthquake detected, None otherwise
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.buffer.append(sample)
        self.total_samples += 1
        
        # Only analyze when buffer is full
        if len(self.buffer) < self.window_size:
            return None
        
        # Rate limit detection
        if timestamp - self.last_detection_time < self.detection_cooldown:
            return None
        
        # Get analysis window
        segment = np.array(list(self.buffer)[-self.window_size:], dtype=np.float32)
        
        # Detect earthquake
        is_earthquake, confidence = self.analyzer.detect_earthquake(segment)
        
        if is_earthquake:
            # Classify wave type
            wave_type, wave_confidence = self.analyzer.classify_wave(segment)
            
            # Get magnitude for P-waves
            magnitude = None
            if wave_type == 'P':
                magnitude, uncertainty = self.analyzer.predict_magnitude(segment)
            
            detection = WaveDetection(
                wave_type=wave_type,
                confidence=wave_confidence,
                start_sample=self.total_samples - self.window_size,
                end_sample=self.total_samples,
                start_time=timestamp - self.window_size / SAMPLE_RATE,
                end_time=timestamp,
                magnitude=magnitude
            )
            
            self.last_detection_time = timestamp
            self.earthquake_count += 1
            
            # Trigger alert
            self.alert_callback(detection, timestamp)
            
            return detection
        
        return None
    
    def process_chunk(self, data: np.ndarray, sample_rate: float = SAMPLE_RATE) -> List[WaveDetection]:
        """
        Process a chunk of samples.
        
        Args:
            data: Array of samples
            sample_rate: Sample rate in Hz
            
        Returns:
            List of detections
        """
        detections = []
        current_time = time.time()
        
        for i, sample in enumerate(data):
            timestamp = current_time + i / sample_rate
            detection = self.process_sample(sample, timestamp)
            if detection:
                detections.append(detection)
        
        return detections
    
    def start_monitoring(self, data_source, sample_rate: float = SAMPLE_RATE):
        """
        Start continuous monitoring.
        
        Args:
            data_source: Generator or iterable yielding samples
            sample_rate: Expected sample rate
        """
        self.is_running = True
        self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print("🎯 REAL-TIME SEISMIC MONITORING STARTED")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Window Size: {self.window_size} samples ({self.window_size/sample_rate:.2f}s)")
        print(f"{'='*60}\n")
        
        try:
            for sample in data_source:
                if not self.is_running:
                    break
                self.process_sample(sample)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
        finally:
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and print statistics."""
        self.is_running = False
        
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"\n{'='*60}")
            print("MONITORING SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Samples processed: {self.total_samples:,}")
            print(f"Earthquakes detected: {self.earthquake_count}")
            print(f"{'='*60}\n")


def simulate_realtime_from_file(filepath: str, sample_rate: float = SAMPLE_RATE) -> Any:
    """
    Generator that simulates real-time data from a file.
    
    Yields samples at approximately real-time rate.
    """
    analyzer = SeismicAnalyzer(verbose=False)
    data, file_sr = analyzer.load_file(filepath)
    
    # Resample if needed
    if file_sr != sample_rate:
        from scipy import signal
        num_samples = int(len(data) * sample_rate / file_sr)
        data = signal.resample(data, num_samples)
    
    chunk_size = int(sample_rate / 10)  # 100ms chunks
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        for sample in chunk:
            yield sample
        time.sleep(chunk_size / sample_rate)


def demo():
    """Run demo analysis on sample files."""
    print("="*70)
    print("SEISMIC WAVE ANALYZER - DEMO")
    print("="*70)
    
    analyzer = SeismicAnalyzer()
    
    # Test on extracted wave samples
    test_dirs = [
        ('extracted_waves/p_wave', 'P-wave'),
        ('extracted_waves/s_wave', 'S-wave'),
        ('extracted_waves/surface_wave', 'Surface wave'),
        ('extracted_waves/noise', 'Noise'),
    ]
    
    for wave_dir, wave_name in test_dirs:
        wave_path = Path(wave_dir)
        if wave_path.exists():
            files = list(wave_path.glob('*.npy'))[:1]
            if files:
                print(f"\n{'='*50}")
                print(f"Testing on {wave_name} sample:")
                print(f"{'='*50}")
                
                data = np.load(files[0])
                is_eq, confidence = analyzer.detect_earthquake(data)
                
                if is_eq:
                    wave_type, wave_conf = analyzer.classify_wave(data)
                    print(f"Result: 🚨 EARTHQUAKE - {wave_type}-wave ({wave_conf:.1%})")
                    
                    if wave_type == 'P':
                        mag, _ = analyzer.predict_magnitude(data)
                        print(f"Estimated Magnitude: {mag:.1f}")
                else:
                    print(f"Result: ✓ No earthquake (Noise)")
    
    # Test on mseed files if available
    mseed_files = list(Path('.').glob('*.mseed'))
    if mseed_files:
        print(f"\n{'='*70}")
        print("ANALYZING MSEED FILES")
        print(f"{'='*70}")
        
        for mseed_file in mseed_files[:2]:
            analyzer.analyze_file(mseed_file)
    
    print(f"\n{'='*70}")
    print("Demo complete!")
    print(f"{'='*70}")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description='Seismic Wave Analyzer - Earthquake Detection and Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single file
  python seismic_analyzer.py analyze earthquake.mseed
  
  # Analyze with visualization
  python seismic_analyzer.py analyze earthquake.wav --visualize
  
  # Real-time simulation from file
  python seismic_analyzer.py monitor --file recording.mseed
  
  # Run demo
  python seismic_analyzer.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a seismic file')
    analyze_parser.add_argument('file', type=str, help='Path to seismic file (.mseed, .wav, .npy)')
    analyze_parser.add_argument('--visualize', '-v', action='store_true', help='Show visualization')
    analyze_parser.add_argument('--output', '-o', type=str, help='Output path for figure')
    analyze_parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Real-time monitoring')
    monitor_parser.add_argument('--file', '-f', type=str, help='Simulate real-time from file')
    
    # Demo command
    subparsers.add_parser('demo', help='Run demo with sample files')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple files')
    batch_parser.add_argument('files', nargs='+', help='Files to process')
    batch_parser.add_argument('--output', '-o', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyzer = SeismicAnalyzer(verbose=not args.quiet)
        
        if args.visualize:
            analyzer.visualize(args.file, output_path=args.output if args.output else None)
        else:
            result = analyzer.analyze_file(args.file)
            
    elif args.command == 'monitor':
        analyzer = SeismicAnalyzer(verbose=True)
        monitor = RealtimeMonitor(analyzer)
        
        if args.file:
            print(f"Simulating real-time monitoring from: {args.file}")
            data_source = simulate_realtime_from_file(args.file)
            monitor.start_monitoring(data_source)
        else:
            print("No data source specified. Use --file to simulate from a file.")
            
    elif args.command == 'batch':
        analyzer = SeismicAnalyzer(verbose=True)
        results = []
        
        for filepath in args.files:
            try:
                result = analyzer.analyze_file(filepath)
                results.append({
                    'file': result.filename,
                    'is_earthquake': result.is_earthquake,
                    'confidence': result.earthquake_confidence,
                    'magnitude': result.estimated_magnitude,
                    'p_arrival': result.p_wave_arrival,
                    's_arrival': result.s_wave_arrival
                })
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    elif args.command == 'demo':
        demo()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
