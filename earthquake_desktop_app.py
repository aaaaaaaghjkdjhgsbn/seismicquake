#!/usr/bin/env python3
"""
Earthquake Detection Desktop Application

A PyQt6-based desktop application for real-time earthquake detection
and seismic wave classification using trained AI models.

Features:
- Load and analyze .mseed, .wav, and .npy seismic files
- Real-time waveform visualization
- P, S, and Surface wave classification
- Magnitude prediction from P-waves
- Real-time monitoring mode with alerts
- Export results to JSON/CSV

Usage:
    python earthquake_desktop_app.py
"""

import os
import sys
import json
import time
import warnings
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Optional, List, Tuple
from collections import deque

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QStatusBar,
    QTabWidget, QGroupBox, QGridLayout, QTextEdit, QSpinBox,
    QDoubleSpinBox, QCheckBox, QComboBox, QSlider, QSplitter,
    QFrame, QListWidget, QListWidgetItem, QMessageBox, QToolBar,
    QSizePolicy, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QDialogButtonBox, QFormLayout
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QUrl
)
from PyQt6.QtGui import (
    QAction, QIcon, QFont, QPalette, QColor, QPainter, QPen,
    QBrush, QLinearGradient, QPixmap, QDragEnterEvent, QDropEvent
)

# Matplotlib for plotting
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import our seismic analyzer
from seismic_analyzer import (
    SeismicAnalyzer, RealtimeMonitor, WaveDetection, AnalysisResult,
    INPUT_LENGTH, SAMPLE_RATE
)


# =============================================================================
# Style Constants
# =============================================================================
COLORS = {
    'P': '#FF4444',
    'S': '#44AA44', 
    'Surface': '#4444FF',
    'Noise': '#888888',
    'background': '#1a1a2e',
    'surface': '#16213e',
    'primary': '#0f3460',
    'accent': '#e94560',
    'text': '#eaeaea',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'error': '#f44336'
}

DARK_STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    background-color: #1a1a2e;
    color: #eaeaea;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QGroupBox {
    border: 2px solid #0f3460;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    font-size: 13px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #e94560;
}
QPushButton {
    background-color: #0f3460;
    color: #eaeaea;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1a4a7a;
}
QPushButton:pressed {
    background-color: #e94560;
}
QPushButton:disabled {
    background-color: #333;
    color: #666;
}
QPushButton#startButton {
    background-color: #4CAF50;
}
QPushButton#startButton:hover {
    background-color: #45a049;
}
QPushButton#stopButton {
    background-color: #f44336;
}
QPushButton#stopButton:hover {
    background-color: #da190b;
}
QLabel {
    color: #eaeaea;
}
QLabel#titleLabel {
    font-size: 24px;
    font-weight: bold;
    color: #e94560;
}
QLabel#statusLabel {
    font-size: 14px;
    padding: 5px;
}
QProgressBar {
    border: 2px solid #0f3460;
    border-radius: 5px;
    text-align: center;
    background-color: #16213e;
}
QProgressBar::chunk {
    background-color: #e94560;
    border-radius: 3px;
}
QTextEdit {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 5px;
    font-family: 'Consolas', monospace;
}
QListWidget {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 5px;
}
QListWidget::item {
    padding: 8px;
    border-bottom: 1px solid #0f3460;
}
QListWidget::item:selected {
    background-color: #0f3460;
}
QTableWidget {
    background-color: #16213e;
    border: 1px solid #0f3460;
    gridline-color: #0f3460;
}
QTableWidget::item {
    padding: 5px;
}
QHeaderView::section {
    background-color: #0f3460;
    color: #eaeaea;
    padding: 8px;
    border: none;
    font-weight: bold;
}
QTabWidget::pane {
    border: 2px solid #0f3460;
    border-radius: 5px;
    background-color: #16213e;
}
QTabBar::tab {
    background-color: #0f3460;
    color: #eaeaea;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:selected {
    background-color: #e94560;
}
QComboBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 8px;
}
QComboBox::drop-down {
    border: none;
}
QSpinBox, QDoubleSpinBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 5px;
}
QSlider::groove:horizontal {
    border: 1px solid #0f3460;
    height: 8px;
    background: #16213e;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #e94560;
    border: none;
    width: 18px;
    margin: -5px 0;
    border-radius: 9px;
}
QStatusBar {
    background-color: #0f3460;
    color: #eaeaea;
}
QToolBar {
    background-color: #16213e;
    border: none;
    spacing: 10px;
    padding: 5px;
}
QMenuBar {
    background-color: #16213e;
}
QMenuBar::item {
    padding: 8px 15px;
}
QMenuBar::item:selected {
    background-color: #0f3460;
}
QMenu {
    background-color: #16213e;
    border: 1px solid #0f3460;
}
QMenu::item {
    padding: 8px 30px;
}
QMenu::item:selected {
    background-color: #0f3460;
}
QScrollBar:vertical {
    background-color: #16213e;
    width: 12px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background-color: #0f3460;
    border-radius: 6px;
    min-height: 30px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


# =============================================================================
# Worker Threads
# =============================================================================
class AnalysisWorker(QThread):
    """Worker thread for file analysis."""
    finished = pyqtSignal(object)
    progress = pyqtSignal(int, str)
    error = pyqtSignal(str)
    
    def __init__(self, analyzer: SeismicAnalyzer, filepath: str):
        super().__init__()
        self.analyzer = analyzer
        self.filepath = filepath
    
    def run(self):
        try:
            self.progress.emit(10, "Loading file...")
            data, sample_rate = self.analyzer.load_file(self.filepath)
            
            self.progress.emit(30, "Analyzing waveform...")
            result = self.analyzer.analyze_file(self.filepath)
            
            self.progress.emit(100, "Complete!")
            self.finished.emit((result, data, sample_rate))
        except Exception as e:
            self.error.emit(str(e))


class RealtimeWorker(QThread):
    """Worker thread for real-time monitoring."""
    detection = pyqtSignal(object, float)  # WaveDetection, timestamp
    sample_update = pyqtSignal(np.ndarray)  # Latest samples for visualization
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, analyzer: SeismicAnalyzer, data: np.ndarray, sample_rate: float):
        super().__init__()
        self.analyzer = analyzer
        self.data = data
        self.sample_rate = sample_rate
        self.is_running = True
        self.speed = 1.0  # Playback speed multiplier
    
    def stop(self):
        self.is_running = False
    
    def set_speed(self, speed: float):
        self.speed = max(0.1, min(10.0, speed))
    
    def run(self):
        buffer = deque(maxlen=INPUT_LENGTH * 2)
        chunk_size = int(self.sample_rate / 10)  # 100ms chunks
        last_detection_time = 0
        cooldown = 2.0
        
        self.status_update.emit("Monitoring started...")
        
        i = 0
        while i < len(self.data) and self.is_running:
            chunk = self.data[i:i + chunk_size]
            buffer.extend(chunk)
            
            # Emit samples for visualization
            if len(buffer) >= INPUT_LENGTH:
                self.sample_update.emit(np.array(list(buffer)[-INPUT_LENGTH * 2:]))
            
            # Check for earthquake every chunk
            if len(buffer) >= INPUT_LENGTH:
                current_time = i / self.sample_rate
                
                if current_time - last_detection_time >= cooldown:
                    segment = np.array(list(buffer)[-INPUT_LENGTH:], dtype=np.float32)
                    is_eq, confidence = self.analyzer.detect_earthquake(segment)
                    
                    if is_eq:
                        wave_type, wave_conf = self.analyzer.classify_wave(segment)
                        magnitude = None
                        if wave_type == 'P':
                            magnitude, _ = self.analyzer.predict_magnitude(segment)
                        
                        detection = WaveDetection(
                            wave_type=wave_type,
                            confidence=wave_conf,
                            start_sample=i - INPUT_LENGTH,
                            end_sample=i,
                            start_time=current_time - INPUT_LENGTH / self.sample_rate,
                            end_time=current_time,
                            magnitude=magnitude
                        )
                        self.detection.emit(detection, current_time)
                        last_detection_time = current_time
            
            i += chunk_size
            time.sleep((chunk_size / self.sample_rate) / self.speed)
        
        self.status_update.emit("Monitoring complete")
        self.finished.emit()


# =============================================================================
# Custom Widgets
# =============================================================================
class WaveformCanvas(FigureCanvas):
    """Canvas for displaying seismic waveforms."""
    
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 4), facecolor=COLORS['surface'])
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS['surface'])
        self.ax.tick_params(colors=COLORS['text'])
        self.ax.spines['bottom'].set_color(COLORS['primary'])
        self.ax.spines['top'].set_color(COLORS['primary'])
        self.ax.spines['left'].set_color(COLORS['primary'])
        self.ax.spines['right'].set_color(COLORS['primary'])
        
        self.fig.tight_layout()
    
    def plot_waveform(self, data: np.ndarray, sample_rate: float,
                      detections: List[WaveDetection] = None, title: str = ""):
        """Plot waveform with optional wave detections."""
        self.ax.clear()
        self.ax.set_facecolor(COLORS['surface'])
        
        time_axis = np.arange(len(data)) / sample_rate
        self.ax.plot(time_axis, data, color=COLORS['text'], linewidth=0.5, alpha=0.8)
        
        if detections:
            for d in detections:
                if d.wave_type != 'Noise':
                    self.ax.axvspan(
                        d.start_time, d.end_time,
                        alpha=0.3, color=COLORS.get(d.wave_type, 'gray')
                    )
        
        self.ax.set_xlabel('Time (seconds)', color=COLORS['text'])
        self.ax.set_ylabel('Amplitude', color=COLORS['text'])
        self.ax.set_title(title, color=COLORS['accent'], fontweight='bold')
        self.ax.tick_params(colors=COLORS['text'])
        self.ax.grid(True, alpha=0.2, color=COLORS['primary'])
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_realtime(self, data: np.ndarray, sample_rate: float):
        """Plot real-time waveform."""
        self.ax.clear()
        self.ax.set_facecolor(COLORS['surface'])
        
        time_axis = np.arange(len(data)) / sample_rate
        self.ax.plot(time_axis, data, color='#00ff88', linewidth=1)
        
        self.ax.set_xlabel('Time (seconds)', color=COLORS['text'])
        self.ax.set_ylabel('Amplitude', color=COLORS['text'])
        self.ax.set_title('Real-time Waveform', color=COLORS['accent'], fontweight='bold')
        self.ax.tick_params(colors=COLORS['text'])
        self.ax.grid(True, alpha=0.2, color=COLORS['primary'])
        
        self.fig.tight_layout()
        self.draw()


class DetectionCard(QFrame):
    """Card widget displaying a wave detection."""
    
    def __init__(self, detection: WaveDetection, parent=None):
        super().__init__(parent)
        self.detection = detection
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 2px solid {COLORS.get(self.detection.wave_type, COLORS['primary'])};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout(self)
        
        # Wave type header
        header = QLabel(f"🌊 {self.detection.wave_type}-wave")
        header.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {COLORS.get(self.detection.wave_type, COLORS['text'])};
        """)
        layout.addWidget(header)
        
        # Confidence
        conf_label = QLabel(f"Confidence: {self.detection.confidence:.1%}")
        layout.addWidget(conf_label)
        
        # Time range
        time_label = QLabel(f"Time: {self.detection.start_time:.2f}s - {self.detection.end_time:.2f}s")
        layout.addWidget(time_label)
        
        # Magnitude (if P-wave)
        if self.detection.magnitude is not None:
            mag_label = QLabel(f"Est. Magnitude: {self.detection.magnitude:.1f} ± 0.5")
            mag_label.setStyleSheet("color: #FF9800; font-weight: bold;")
            layout.addWidget(mag_label)


class AlertWidget(QFrame):
    """Widget for displaying earthquake alerts."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.hide()
    
    def setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['error']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        
        layout = QHBoxLayout(self)
        
        # Alert icon
        icon_label = QLabel("🚨")
        icon_label.setStyleSheet("font-size: 32px;")
        layout.addWidget(icon_label)
        
        # Alert text
        self.text_label = QLabel("EARTHQUAKE DETECTED!")
        self.text_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: white;
        """)
        layout.addWidget(self.text_label)
        
        layout.addStretch()
        
        # Dismiss button
        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.clicked.connect(self.hide)
        layout.addWidget(dismiss_btn)
    
    def show_alert(self, wave_type: str, confidence: float, magnitude: float = None):
        """Show alert with detection details."""
        text = f"🚨 EARTHQUAKE: {wave_type}-wave detected ({confidence:.1%})"
        if magnitude is not None:
            text += f" | Magnitude: {magnitude:.1f}"
        self.text_label.setText(text)
        self.show()
        
        # Auto-hide after 10 seconds
        QTimer.singleShot(10000, self.hide)


# =============================================================================
# Main Application Window
# =============================================================================
class EarthquakeDetectorApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌍 Earthquake Detection System")
        self.setMinimumSize(1400, 900)
        
        # Initialize analyzer
        self.analyzer = None
        self.current_result = None
        self.current_data = None
        self.current_sample_rate = None
        self.realtime_worker = None
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        
        # Load AI models
        self.load_models()
    
    def setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Alert widget (hidden by default)
        self.alert_widget = AlertWidget()
        main_layout.addWidget(self.alert_widget)
        
        # Title
        title_label = QLabel("🌍 Earthquake Detection System")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_analysis_tab()
        self.create_realtime_tab()
        self.create_results_tab()
        self.create_settings_tab()
    
    def create_analysis_tab(self):
        """Create the file analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File selection group
        file_group = QGroupBox("📁 File Selection")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 10px;")
        file_layout.addWidget(self.file_label, stretch=1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        
        self.analyze_btn = QPushButton("🔍 Analyze")
        self.analyze_btn.clicked.connect(self.analyze_file)
        self.analyze_btn.setEnabled(False)
        file_layout.addWidget(self.analyze_btn)
        
        layout.addWidget(file_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Splitter for waveform and results
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Waveform canvas
        waveform_group = QGroupBox("📈 Waveform Analysis")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.analysis_canvas = WaveformCanvas()
        self.analysis_toolbar = NavigationToolbar(self.analysis_canvas, self)
        waveform_layout.addWidget(self.analysis_toolbar)
        waveform_layout.addWidget(self.analysis_canvas)
        
        splitter.addWidget(waveform_group)
        
        # Detection results
        results_group = QGroupBox("🎯 Detection Results")
        results_layout = QVBoxLayout(results_group)
        
        self.result_summary = QLabel("Analyze a file to see results")
        self.result_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_summary.setStyleSheet("font-size: 16px; padding: 20px;")
        results_layout.addWidget(self.result_summary)
        
        # Detections scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.detections_widget = QWidget()
        self.detections_layout = QHBoxLayout(self.detections_widget)
        scroll.setWidget(self.detections_widget)
        results_layout.addWidget(scroll)
        
        splitter.addWidget(results_group)
        splitter.setSizes([500, 300])
        
        layout.addWidget(splitter)
        
        self.tab_widget.addTab(tab, "📊 File Analysis")
    
    def create_realtime_tab(self):
        """Create the real-time monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Control panel
        control_group = QGroupBox("🎮 Monitoring Controls")
        control_layout = QHBoxLayout(control_group)
        
        # Source selection
        control_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Load from file", "Simulated data"])
        control_layout.addWidget(self.source_combo)
        
        self.load_source_btn = QPushButton("Load Source")
        self.load_source_btn.clicked.connect(self.load_realtime_source)
        control_layout.addWidget(self.load_source_btn)
        
        control_layout.addStretch()
        
        # Speed control
        control_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(10)
        self.speed_slider.setMaximumWidth(150)
        self.speed_slider.valueChanged.connect(self.update_speed)
        control_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.0x")
        control_layout.addWidget(self.speed_label)
        
        control_layout.addStretch()
        
        # Start/Stop buttons
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_monitoring)
        self.start_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        layout.addWidget(control_group)
        
        # Real-time waveform
        waveform_group = QGroupBox("📡 Live Waveform")
        waveform_layout = QVBoxLayout(waveform_group)
        
        self.realtime_canvas = WaveformCanvas()
        waveform_layout.addWidget(self.realtime_canvas)
        
        layout.addWidget(waveform_group)
        
        # Detection log
        log_group = QGroupBox("📋 Detection Log")
        log_layout = QVBoxLayout(log_group)
        
        self.detection_log = QTextEdit()
        self.detection_log.setReadOnly(True)
        self.detection_log.setMaximumHeight(200)
        log_layout.addWidget(self.detection_log)
        
        # Clear log button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(lambda: self.detection_log.clear())
        log_layout.addWidget(clear_btn)
        
        layout.addWidget(log_group)
        
        self.tab_widget.addTab(tab, "📡 Real-time Monitor")
    
    def create_results_tab(self):
        """Create the results and export tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results table
        table_group = QGroupBox("📊 Analysis History")
        table_layout = QVBoxLayout(table_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Earthquake", "Confidence", "Magnitude",
            "P-wave", "S-wave", "Surface"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table_layout.addWidget(self.results_table)
        
        layout.addWidget(table_group)
        
        # Export controls
        export_group = QGroupBox("💾 Export Results")
        export_layout = QHBoxLayout(export_group)
        
        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.clicked.connect(lambda: self.export_results("json"))
        export_layout.addWidget(self.export_json_btn)
        
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_results("csv"))
        export_layout.addWidget(self.export_csv_btn)
        
        export_layout.addStretch()
        
        clear_history_btn = QPushButton("Clear History")
        clear_history_btn.clicked.connect(self.clear_history)
        export_layout.addWidget(clear_history_btn)
        
        layout.addWidget(export_group)
        
        self.tab_widget.addTab(tab, "📋 Results")
    
    def create_settings_tab(self):
        """Create the settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model settings
        model_group = QGroupBox("🧠 AI Model Settings")
        model_layout = QFormLayout(model_group)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.99)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.05)
        model_layout.addRow("Detection Threshold:", self.threshold_spin)
        
        self.window_spin = QSpinBox()
        self.window_spin.setRange(100, 1000)
        self.window_spin.setValue(INPUT_LENGTH)
        model_layout.addRow("Window Size (samples):", self.window_spin)
        
        layout.addWidget(model_group)
        
        # Display settings
        display_group = QGroupBox("🎨 Display Settings")
        display_layout = QFormLayout(display_group)
        
        self.show_noise_check = QCheckBox()
        self.show_noise_check.setChecked(False)
        display_layout.addRow("Show Noise Segments:", self.show_noise_check)
        
        self.auto_alert_check = QCheckBox()
        self.auto_alert_check.setChecked(True)
        display_layout.addRow("Auto Alert on Detection:", self.auto_alert_check)
        
        layout.addWidget(display_group)
        
        # Model info
        info_group = QGroupBox("ℹ️ Model Information")
        info_layout = QVBoxLayout(info_group)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(200)
        info_layout.addWidget(self.model_info_text)
        
        layout.addWidget(info_group)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "⚙️ Settings")
    
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_action = QAction("&Open File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.browse_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(lambda: self.export_results("json"))
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        analyze_action = QAction("&Analyze Current File", self)
        analyze_action.setShortcut("F5")
        analyze_action.triggered.connect(self.analyze_file)
        analysis_menu.addAction(analyze_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Setup the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Open action
        open_action = QAction("📂 Open", self)
        open_action.triggered.connect(self.browse_file)
        toolbar.addAction(open_action)
        
        # Analyze action
        analyze_action = QAction("🔍 Analyze", self)
        analyze_action.triggered.connect(self.analyze_file)
        toolbar.addAction(analyze_action)
        
        toolbar.addSeparator()
        
        # Export action
        export_action = QAction("💾 Export", self)
        export_action.triggered.connect(lambda: self.export_results("json"))
        toolbar.addAction(export_action)
    
    def setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Model status
        self.model_status = QLabel("Loading models...")
        self.statusbar.addWidget(self.model_status)
        
        # Spacer
        self.statusbar.addPermanentWidget(QLabel(""))
        
        # GPU status
        self.gpu_status = QLabel("")
        self.statusbar.addPermanentWidget(self.gpu_status)
    
    def load_models(self):
        """Load the AI models."""
        try:
            self.model_status.setText("Loading AI models...")
            QApplication.processEvents()
            
            self.analyzer = SeismicAnalyzer(verbose=False)
            
            # Check which models loaded
            models_loaded = []
            if self.analyzer.earthquake_detector:
                models_loaded.append("Detector")
            if self.analyzer.wave_classifier:
                models_loaded.append("Classifier")
            if self.analyzer.magnitude_predictor:
                models_loaded.append("Magnitude")
            
            self.model_status.setText(f"✅ Models loaded: {', '.join(models_loaded)}")
            
            # Update model info
            info = f"Loaded Models:\n"
            info += f"- Earthquake Detector: {'✅' if self.analyzer.earthquake_detector else '❌'}\n"
            info += f"- Wave Classifier: {'✅' if self.analyzer.wave_classifier else '❌'}\n"
            info += f"- Magnitude Predictor: {'✅' if self.analyzer.magnitude_predictor else '❌'}\n"
            info += f"\nInput Window: {INPUT_LENGTH} samples ({INPUT_LENGTH/SAMPLE_RATE:.1f}s)\n"
            info += f"Sample Rate: {SAMPLE_RATE} Hz"
            self.model_info_text.setText(info)
            
            # Check GPU
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_status.setText(f"🎮 GPU: {gpus[0].name.split('/')[-1]}")
            else:
                self.gpu_status.setText("💻 CPU Mode")
            
        except Exception as e:
            self.model_status.setText(f"❌ Error loading models: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load AI models:\n{str(e)}")
    
    def browse_file(self):
        """Open file browser dialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Seismic Data File",
            "",
            "Seismic Files (*.mseed *.wav *.npy);;MiniSEED (*.mseed);;WAV Audio (*.wav);;NumPy (*.npy);;All Files (*)"
        )
        
        if filepath:
            self.current_filepath = filepath
            self.file_label.setText(f"📄 {Path(filepath).name}")
            self.analyze_btn.setEnabled(True)
    
    def analyze_file(self):
        """Analyze the selected file."""
        if not hasattr(self, 'current_filepath') or not self.analyzer:
            return
        
        self.analyze_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # Create worker thread
        self.analysis_worker = AnalysisWorker(self.analyzer, self.current_filepath)
        self.analysis_worker.progress.connect(self.update_progress)
        self.analysis_worker.finished.connect(self.analysis_complete)
        self.analysis_worker.error.connect(self.analysis_error)
        self.analysis_worker.start()
    
    def update_progress(self, value: int, message: str):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}% - {message}")
    
    def analysis_complete(self, result_tuple):
        """Handle analysis completion."""
        result, data, sample_rate = result_tuple
        self.current_result = result
        self.current_data = data
        self.current_sample_rate = sample_rate
        
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        
        # Update waveform plot
        self.analysis_canvas.plot_waveform(
            data, sample_rate, result.detections,
            title=f"Analysis: {result.filename}"
        )
        
        # Update result summary
        if result.is_earthquake:
            summary = f"""
            <h2 style='color: {COLORS["error"]};'>🚨 EARTHQUAKE DETECTED</h2>
            <p><b>Confidence:</b> {result.earthquake_confidence:.1%}</p>
            """
            if result.estimated_magnitude:
                summary += f"<p><b>Estimated Magnitude:</b> {result.estimated_magnitude:.1f} ± 0.5</p>"
            if result.p_wave_arrival:
                summary += f"<p><b>P-wave Arrival:</b> {result.p_wave_arrival:.2f}s</p>"
            if result.s_wave_arrival:
                summary += f"<p><b>S-wave Arrival:</b> {result.s_wave_arrival:.2f}s</p>"
            if result.surface_wave_arrival:
                summary += f"<p><b>Surface Wave Arrival:</b> {result.surface_wave_arrival:.2f}s</p>"
            
            # Show alert
            if self.auto_alert_check.isChecked():
                self.alert_widget.show_alert(
                    "P" if result.p_wave_arrival else "S" if result.s_wave_arrival else "Surface",
                    result.earthquake_confidence,
                    result.estimated_magnitude
                )
        else:
            summary = f"""
            <h2 style='color: {COLORS["success"]};'>✅ No Earthquake Detected</h2>
            <p>The analyzed signal appears to be noise.</p>
            """
        
        summary += f"<p><b>Processing Time:</b> {result.processing_time:.3f}s</p>"
        self.result_summary.setText(summary)
        
        # Update detections cards
        self.update_detection_cards(result.detections)
        
        # Add to results table
        self.add_to_results_table(result)
        
        self.statusbar.showMessage(f"Analysis complete: {result.filename}", 5000)
    
    def update_detection_cards(self, detections: List[WaveDetection]):
        """Update the detection cards display."""
        # Clear existing cards
        while self.detections_layout.count():
            child = self.detections_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add new cards
        earthquake_detections = [d for d in detections if d.wave_type != 'Noise']
        
        for detection in earthquake_detections[:10]:  # Limit to 10 cards
            card = DetectionCard(detection)
            self.detections_layout.addWidget(card)
        
        self.detections_layout.addStretch()
    
    def add_to_results_table(self, result: AnalysisResult):
        """Add analysis result to the history table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(result.filename))
        self.results_table.setItem(row, 1, QTableWidgetItem("Yes" if result.is_earthquake else "No"))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{result.earthquake_confidence:.1%}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(
            f"{result.estimated_magnitude:.1f}" if result.estimated_magnitude else "-"
        ))
        self.results_table.setItem(row, 4, QTableWidgetItem(
            f"{result.p_wave_arrival:.2f}s" if result.p_wave_arrival else "-"
        ))
        self.results_table.setItem(row, 5, QTableWidgetItem(
            f"{result.s_wave_arrival:.2f}s" if result.s_wave_arrival else "-"
        ))
        self.results_table.setItem(row, 6, QTableWidgetItem(
            f"{result.surface_wave_arrival:.2f}s" if result.surface_wave_arrival else "-"
        ))
    
    def analysis_error(self, error_message: str):
        """Handle analysis error."""
        self.progress_bar.hide()
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", f"Error analyzing file:\n{error_message}")
    
    def load_realtime_source(self):
        """Load data source for real-time monitoring."""
        if self.source_combo.currentText() == "Load from file":
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Select Source File",
                "",
                "Seismic Files (*.mseed *.wav *.npy);;All Files (*)"
            )
            
            if filepath:
                try:
                    self.realtime_data, self.realtime_sr = self.analyzer.load_file(filepath)
                    self.start_btn.setEnabled(True)
                    self.statusbar.showMessage(f"Loaded source: {Path(filepath).name}", 3000)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
        else:
            # Generate simulated data
            duration = 60  # seconds
            self.realtime_sr = SAMPLE_RATE
            self.realtime_data = np.random.randn(int(duration * SAMPLE_RATE)).astype(np.float32) * 0.1
            self.start_btn.setEnabled(True)
            self.statusbar.showMessage("Loaded simulated noise data", 3000)
    
    def update_speed(self, value):
        """Update playback speed."""
        speed = value / 10.0
        self.speed_label.setText(f"{speed:.1f}x")
        if self.realtime_worker:
            self.realtime_worker.set_speed(speed)
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if not hasattr(self, 'realtime_data'):
            return
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_source_btn.setEnabled(False)
        
        # Create worker
        self.realtime_worker = RealtimeWorker(
            self.analyzer, self.realtime_data, self.realtime_sr
        )
        self.realtime_worker.set_speed(self.speed_slider.value() / 10.0)
        self.realtime_worker.detection.connect(self.handle_realtime_detection)
        self.realtime_worker.sample_update.connect(self.update_realtime_plot)
        self.realtime_worker.status_update.connect(lambda s: self.statusbar.showMessage(s))
        self.realtime_worker.finished.connect(self.monitoring_finished)
        self.realtime_worker.start()
        
        self.detection_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring started...")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if self.realtime_worker:
            self.realtime_worker.stop()
    
    def monitoring_finished(self):
        """Handle monitoring completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_source_btn.setEnabled(True)
        self.detection_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring stopped")
    
    def handle_realtime_detection(self, detection: WaveDetection, timestamp: float):
        """Handle real-time detection."""
        time_str = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{time_str}] 🚨 {detection.wave_type}-wave detected ({detection.confidence:.1%})"
        if detection.magnitude:
            log_msg += f" | Magnitude: {detection.magnitude:.1f}"
        
        self.detection_log.append(f'<span style="color: {COLORS[detection.wave_type]};">{log_msg}</span>')
        
        # Show alert
        if self.auto_alert_check.isChecked():
            self.alert_widget.show_alert(
                detection.wave_type,
                detection.confidence,
                detection.magnitude
            )
    
    def update_realtime_plot(self, data: np.ndarray):
        """Update real-time waveform plot."""
        self.realtime_canvas.plot_realtime(data, self.realtime_sr)
    
    def export_results(self, format: str):
        """Export results to file."""
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "No Data", "No results to export.")
            return
        
        if format == "json":
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export JSON", "", "JSON Files (*.json)"
            )
            if filepath:
                results = []
                for row in range(self.results_table.rowCount()):
                    results.append({
                        'file': self.results_table.item(row, 0).text(),
                        'earthquake': self.results_table.item(row, 1).text(),
                        'confidence': self.results_table.item(row, 2).text(),
                        'magnitude': self.results_table.item(row, 3).text(),
                        'p_wave': self.results_table.item(row, 4).text(),
                        's_wave': self.results_table.item(row, 5).text(),
                        'surface_wave': self.results_table.item(row, 6).text(),
                    })
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2)
                self.statusbar.showMessage(f"Exported to {filepath}", 3000)
        
        elif format == "csv":
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv)"
            )
            if filepath:
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['File', 'Earthquake', 'Confidence', 'Magnitude',
                                    'P-wave', 'S-wave', 'Surface Wave'])
                    for row in range(self.results_table.rowCount()):
                        writer.writerow([
                            self.results_table.item(row, col).text()
                            for col in range(7)
                        ])
                self.statusbar.showMessage(f"Exported to {filepath}", 3000)
    
    def clear_history(self):
        """Clear results history."""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear all results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.results_table.setRowCount(0)
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Earthquake Detection System",
            """<h2>🌍 Earthquake Detection System</h2>
            <p>Version 1.0</p>
            <p>A desktop application for real-time earthquake detection
            and seismic wave classification using AI.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Earthquake vs Noise detection (96.8% accuracy)</li>
                <li>P, S, Surface wave classification (99.7% accuracy)</li>
                <li>Magnitude prediction from P-waves</li>
                <li>Real-time monitoring</li>
                <li>Support for .mseed, .wav, .npy files</li>
            </ul>
            <p>Powered by TensorFlow and PyQt6</p>
            """
        )
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.realtime_worker and self.realtime_worker.isRunning():
            self.realtime_worker.stop()
            self.realtime_worker.wait()
        event.accept()


# =============================================================================
# Entry Point
# =============================================================================
def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)
    
    # Set application info
    app.setApplicationName("Earthquake Detection System")
    app.setOrganizationName("SeismicQuake")
    
    # Create and show main window
    window = EarthquakeDetectorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
