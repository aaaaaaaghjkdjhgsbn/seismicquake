#!/usr/bin/env python3
"""
Real-time Earthquake Detection and Classification System

Uses trained AI models to:
1. Detect earthquake signals from noise
2. Classify wave types (P, S, Surface)
3. Predict earthquake magnitude from P-wave

Designed for continuous monitoring of seismic data streams.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time

# Configuration
MODELS_DIR = Path("./earthquake_ai_models")
INPUT_LENGTH = 400
SAMPLE_RATE = 100  # Hz


@dataclass
class Detection:
    """Represents a seismic detection result."""
    timestamp: float
    is_earthquake: bool
    earthquake_confidence: float
    wave_type: Optional[str]  # 'P', 'S', 'Surface', or None
    wave_confidence: float
    magnitude: Optional[float]
    magnitude_uncertainty: float
    raw_waveform: np.ndarray


class EarthquakeDetector:
    """
    Real-time earthquake detection and classification system.
    
    Pipeline:
    1. Detect if signal is earthquake or noise
    2. If earthquake, classify wave type (P/S/Surface)
    3. If P-wave detected, predict magnitude
    """
    
    # Thresholds
    EARTHQUAKE_THRESHOLD = 0.5
    WAVE_CONFIDENCE_THRESHOLD = 0.4
    P_WAVE_MAGNITUDE_THRESHOLD = 0.6
    
    # Wave type labels
    WAVE_TYPES = ['P', 'S', 'Surface']
    
    def __init__(self, models_dir: Path = MODELS_DIR):
        """Initialize detector with trained models."""
        self.models_dir = Path(models_dir)
        self.earthquake_detector = None
        self.wave_classifier = None
        self.magnitude_predictor = None
        self._load_models()
        
        # Buffer for continuous monitoring
        self.buffer = deque(maxlen=INPUT_LENGTH * 2)
        self.detections = []
        
    def _load_models(self):
        """Load trained models."""
        print("Loading earthquake AI models...")
        
        # Try best models first, then final
        detector_path = self.models_dir / 'earthquake_detector_best.h5'
        if not detector_path.exists():
            detector_path = self.models_dir / 'earthquake_detector_final.h5'
        
        wave_path = self.models_dir / 'wave_classifier_best.h5'
        if not wave_path.exists():
            wave_path = self.models_dir / 'wave_classifier_final.h5'
            
        mag_path = self.models_dir / 'magnitude_predictor_best.h5'
        if not mag_path.exists():
            mag_path = self.models_dir / 'magnitude_predictor_final.h5'
        
        # Load models
        if detector_path.exists():
            self.earthquake_detector = keras.models.load_model(detector_path)
            print(f"  ✓ Earthquake detector loaded: {detector_path.name}")
        else:
            print(f"  ⚠ Earthquake detector not found")
            
        if wave_path.exists():
            self.wave_classifier = keras.models.load_model(wave_path)
            print(f"  ✓ Wave classifier loaded: {wave_path.name}")
        else:
            print(f"  ⚠ Wave classifier not found")
            
        if mag_path.exists():
            self.magnitude_predictor = keras.models.load_model(mag_path)
            print(f"  ✓ Magnitude predictor loaded: {mag_path.name}")
        else:
            print(f"  ⚠ Magnitude predictor not found")
    
    def _normalize(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform to [-1, 1]."""
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            return waveform / max_val
        return waveform
    
    def _pad_or_truncate(self, waveform: np.ndarray) -> np.ndarray:
        """Pad or truncate to INPUT_LENGTH."""
        if len(waveform) >= INPUT_LENGTH:
            return waveform[:INPUT_LENGTH]
        else:
            padded = np.zeros(INPUT_LENGTH, dtype=np.float32)
            padded[:len(waveform)] = waveform
            return padded
    
    def _preprocess(self, waveform: np.ndarray) -> np.ndarray:
        """Preprocess waveform for model input."""
        waveform = self._pad_or_truncate(waveform)
        waveform = self._normalize(waveform)
        return waveform.reshape(1, INPUT_LENGTH, 1)
    
    def detect_earthquake(self, waveform: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if waveform contains earthquake signal.
        
        Returns:
            is_earthquake: bool
            confidence: float (0-1)
        """
        if self.earthquake_detector is None:
            return False, 0.0
        
        X = self._preprocess(waveform)
        prob = self.earthquake_detector.predict(X, verbose=0)[0, 0]
        is_earthquake = prob >= self.EARTHQUAKE_THRESHOLD
        
        return is_earthquake, float(prob)
    
    def classify_wave(self, waveform: np.ndarray) -> Tuple[str, float]:
        """
        Classify wave type (P, S, Surface).
        
        Returns:
            wave_type: str ('P', 'S', 'Surface')
            confidence: float (0-1)
        """
        if self.wave_classifier is None:
            return 'Unknown', 0.0
        
        X = self._preprocess(waveform)
        probs = self.wave_classifier.predict(X, verbose=0)[0]
        
        wave_idx = np.argmax(probs)
        confidence = float(probs[wave_idx])
        wave_type = self.WAVE_TYPES[wave_idx]
        
        return wave_type, confidence
    
    def predict_magnitude(self, p_wave: np.ndarray) -> Tuple[float, float]:
        """
        Predict earthquake magnitude from P-wave.
        
        Returns:
            magnitude: float
            uncertainty: float (estimated error)
        """
        if self.magnitude_predictor is None:
            return 0.0, 0.0
        
        X = self._preprocess(p_wave)
        magnitude = float(self.magnitude_predictor.predict(X, verbose=0)[0, 0])
        
        # Estimate uncertainty based on model performance
        # (typically ±0.5 for well-trained models)
        uncertainty = 0.5
        
        # Clamp magnitude to reasonable range
        magnitude = max(0.0, min(10.0, magnitude))
        
        return magnitude, uncertainty
    
    def analyze(self, waveform: np.ndarray, timestamp: float = None) -> Detection:
        """
        Full analysis pipeline for a waveform segment.
        
        Args:
            waveform: Seismic waveform data
            timestamp: Optional timestamp for the detection
            
        Returns:
            Detection object with all results
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Step 1: Detect earthquake
        is_earthquake, eq_confidence = self.detect_earthquake(waveform)
        
        # Initialize results
        wave_type = None
        wave_confidence = 0.0
        magnitude = None
        mag_uncertainty = 0.0
        
        if is_earthquake:
            # Step 2: Classify wave type
            wave_type, wave_confidence = self.classify_wave(waveform)
            
            # Step 3: Predict magnitude if P-wave detected
            if wave_type == 'P' and wave_confidence >= self.P_WAVE_MAGNITUDE_THRESHOLD:
                magnitude, mag_uncertainty = self.predict_magnitude(waveform)
        
        detection = Detection(
            timestamp=timestamp,
            is_earthquake=is_earthquake,
            earthquake_confidence=eq_confidence,
            wave_type=wave_type,
            wave_confidence=wave_confidence,
            magnitude=magnitude,
            magnitude_uncertainty=mag_uncertainty,
            raw_waveform=waveform
        )
        
        self.detections.append(detection)
        return detection
    
    def process_stream(self, data_chunk: np.ndarray, timestamp: float = None) -> List[Detection]:
        """
        Process continuous data stream with sliding window.
        
        Args:
            data_chunk: New seismic data to process
            timestamp: Timestamp for the start of this chunk
            
        Returns:
            List of new detections
        """
        # Add to buffer
        self.buffer.extend(data_chunk)
        
        detections = []
        
        # Process when buffer is full
        if len(self.buffer) >= INPUT_LENGTH:
            waveform = np.array(list(self.buffer))[-INPUT_LENGTH:]
            detection = self.analyze(waveform, timestamp)
            
            if detection.is_earthquake:
                detections.append(detection)
        
        return detections
    
    def get_status_string(self, detection: Detection) -> str:
        """Format detection result as human-readable string."""
        if not detection.is_earthquake:
            return f"[{detection.earthquake_confidence:.1%}] No earthquake detected (noise)"
        
        status = f"[{detection.earthquake_confidence:.1%}] 🚨 EARTHQUAKE DETECTED"
        
        if detection.wave_type:
            status += f" | Wave: {detection.wave_type}-wave ({detection.wave_confidence:.1%})"
        
        if detection.magnitude is not None:
            status += f" | Magnitude: {detection.magnitude:.1f} ± {detection.magnitude_uncertainty:.1f}"
        
        return status


def demo_detection():
    """Demo the detection system with sample files."""
    from pathlib import Path
    import random
    
    print("="*70)
    print("EARTHQUAKE DETECTION SYSTEM - DEMO")
    print("="*70)
    
    # Initialize detector
    detector = EarthquakeDetector()
    
    # Test on sample files
    test_cases = [
        ("extracted_waves/p_wave", "P-wave earthquake"),
        ("extracted_waves/s_wave", "S-wave earthquake"),
        ("extracted_waves/surface_wave", "Surface wave earthquake"),
        ("extracted_waves/noise", "Noise (no earthquake)")
    ]
    
    print("\n" + "="*70)
    print("Testing on sample files:")
    print("="*70)
    
    for folder, description in test_cases:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"\n⚠ Folder not found: {folder}")
            continue
        
        files = list(folder_path.glob("*.npy"))
        if not files:
            continue
        
        # Test on random sample
        sample_file = random.choice(files)
        waveform = np.load(sample_file)
        
        print(f"\n📁 {description}:")
        print(f"   File: {sample_file.name}")
        
        detection = detector.analyze(waveform)
        print(f"   Result: {detector.get_status_string(detection)}")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)


if __name__ == "__main__":
    demo_detection()
