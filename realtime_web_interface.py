#!/usr/bin/env python3
"""
Real-Time Seismic Monitoring Web Interface

Web-based interface for monitoring live seismic data from global stations.
Integrates with the main earthquake_gui.py system.
"""

import gradio as gr
import threading
import queue
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

try:
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False

import tensorflow as tf
from tensorflow import keras
import librosa

# Global state
monitoring_active = False
detection_log = []
current_data = {}

# ==============================
# STATION PRESETS
# ==============================
STATION_PRESETS = {
    "🇺🇸 USA - West Coast": [
        "IU.ANMO.00.BHZ",  # New Mexico
        "CI.PAS.00.BHZ",    # Pasadena, CA
        "BK.CMB.00.BHZ",    # Columbia, CA
    ],
    "🇺🇸 USA - East Coast": [
        "IU.HRV.00.BHZ",   # Harvard, MA
        "IU.CCM.00.BHZ",   # Missouri
        "N4.USIN.00.BHZ",  # Indiana
    ],
    "🇯🇵 Japan (Ring of Fire)": [
        "IU.MAJO.00.BHZ",  # Matsushiro
        "IU.TATO.00.BHZ",  # Taipei, Taiwan
    ],
    "🌍 Europe": [
        "G.CAN.00.BHZ",    # Canberra
        "GE.APE.00.BHZ",   # Italy
        "IU.PAB.00.BHZ",   # Spain
    ],
    "🌐 Global Network": [
        "IU.ANMO.00.BHZ",  # USA
        "IU.MAJO.00.BHZ",  # Japan
        "G.CAN.00.BHZ",    # Australia
        "IU.PAB.00.BHZ",   # Europe
    ]
}

# ==============================
# REAL-TIME DATA FETCHER
# ==============================
class RealtimeDataFetcher:
    def __init__(self, provider='IRIS'):
        self.provider = provider
        self.client = None
        self.connect()
    
    def connect(self):
        if not OBSPY_AVAILABLE:
            return False
        try:
            self.client = Client(self.provider, timeout=30)
            return True
        except:
            return False
    
    def fetch_latest(self, station_id, duration=30):
        """Fetch latest data from station."""
        if not self.client:
            return None
        
        try:
            parts = station_id.split('.')
            if len(parts) != 4:
                return None
            
            network, station, location, channel = parts
            
            endtime = UTCDateTime()
            starttime = endtime - duration
            
            stream = self.client.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=starttime,
                endtime=endtime
            )
            
            if len(stream) > 0:
                return stream[0]
            return None
            
        except Exception as e:
            return None

# ==============================
# MONITORING FUNCTIONS
# ==============================

# Global state for monitoring
monitoring_active = False
current_stations = []
current_provider = 'IRIS'

def start_realtime_monitoring(stations_text, provider, update_interval):
    """Start monitoring selected stations."""
    global monitoring_active, current_stations, current_provider
    
    if not OBSPY_AVAILABLE:
        return "❌ Error: obspy not installed. Install with: pip install obspy"
    
    stations = [s.strip() for s in stations_text.split('\n') if s.strip()]
    
    if not stations:
        return "❌ No stations selected"
    
    monitoring_active = True
    current_stations = stations
    current_provider = provider
    
    log = []
    log.append(f"{'='*60}")
    log.append(f"🌍 REAL-TIME MONITORING STARTED")
    log.append(f"{'='*60}")
    log.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"Provider: {provider}")
    log.append(f"Stations: {len(stations)}")
    log.append(f"Update Interval: {update_interval} seconds")
    log.append(f"{'='*60}\n")
    
    for station in stations:
        log.append(f"📡 Monitoring: {station}")
    
    log.append(f"\n✓ Monitoring started! Data will appear below.")
    log.append(f"📊 Check the 'Multi-Station Live View' section for real-time data.")
    
    return "\n".join(log)

def stop_realtime_monitoring():
    """Stop monitoring."""
    global monitoring_active
    monitoring_active = False
    return "🛑 Monitoring stopped"

def fetch_multi_station_data(should_update):
    """Fetch data from all monitored stations."""
    global monitoring_active, current_stations, current_provider
    
    if not should_update or not monitoring_active:
        return None, "Monitoring not active. Click 'Start Monitoring' first."
    
    if not current_stations:
        return None, "No stations configured."
    
    if not OBSPY_AVAILABLE:
        return None, "ObsPy not available"
    
    fetcher = RealtimeDataFetcher(current_provider)
    if not fetcher.client:
        return None, f"Could not connect to {current_provider}"
    
    # Fetch data from first station (or combine multiple)
    station = current_stations[0]
    
    trace = fetcher.fetch_latest(station, duration=60)
    
    if trace is None:
        # Try next station if first one fails
        if len(current_stations) > 1:
            station = current_stations[1]
            trace = fetcher.fetch_latest(station, duration=60)
    
    if trace is None:
        return None, f"⚠️ No data from {station} at {datetime.now().strftime('%H:%M:%S')}\nRetrying in 10s..."
    
    # Process data with STA/LTA
    data = trace.data.astype(np.float32)
    data_norm = data / (np.abs(data).max() + 1e-8)
    sr = trace.stats.sampling_rate
    
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset
    cft = recursive_sta_lta(data_norm, int(0.5 * sr), int(10.0 * sr))
    triggers = trigger_onset(cft, 3.5, 1.0)
    
    # Create plot
    fig = Figure(figsize=(14, 8))
    
    # Subplot 1: Waveform
    ax1 = fig.add_subplot(2, 1, 1)
    time_axis = np.arange(len(trace.data)) / sr
    ax1.plot(time_axis, trace.data, 'b-', linewidth=0.8)
    ax1.set_ylabel('Amplitude', fontweight='bold', fontsize=11)
    ax1.set_title(f'🌍 LIVE MONITORING: {station} - {datetime.now().strftime("%H:%M:%S")}', 
                 fontweight='bold', fontsize=13, color='darkgreen')
    ax1.grid(True, alpha=0.3)
    
    # Mark triggers
    if len(triggers) > 0:
        for on, off in triggers:
            ax1.axvspan(on/sr, off/sr, alpha=0.3, color='red', label='🚨 EARTHQUAKE!' if on == triggers[0][0] else '')
        if len(triggers) > 0:
            ax1.legend(loc='upper right', fontsize=12)
    
    # Stats box
    stats_text = f"Station: {station}\n"
    stats_text += f"Rate: {sr} Hz\n"
    stats_text += f"Duration: {len(trace.data)/sr:.1f}s\n"
    stats_text += f"Triggers: {len(triggers)}"
    
    ax1.text(0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
            fontsize=10, fontweight='bold')
    
    # Subplot 2: STA/LTA
    ax2 = fig.add_subplot(2, 1, 2)
    cft_time = np.arange(len(cft)) / sr
    ax2.plot(cft_time, cft, 'g-', linewidth=1.5)
    ax2.axhline(y=3.5, color='red', linestyle='--', linewidth=2, label='Trigger ON')
    ax2.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Trigger OFF')
    ax2.set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('STA/LTA Ratio', fontweight='bold', fontsize=11)
    ax2.set_title('Earthquake Detection', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    for on, off in triggers:
        ax2.axvspan(on/sr, off/sr, alpha=0.2, color='yellow')
    
    fig.tight_layout()
    
    # Status text
    status = f"{'='*60}\n"
    status += f"📡 LIVE MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    status += f"{'='*60}\n\n"
    status += f"Station: {station}\n"
    status += f"Provider: {current_provider}\n"
    status += f"Monitoring: {len(current_stations)} station(s)\n\n"
    
    status += f"DATA:\n"
    status += f"  • Samples: {len(trace.data)}\n"
    status += f"  • Rate: {sr} Hz\n"
    status += f"  • Range: [{data.min():.0f}, {data.max():.0f}]\n\n"
    
    if len(triggers) > 0:
        status += f"🚨 {len(triggers)} TRIGGER(S) DETECTED!\n\n"
        for i, (on, off) in enumerate(triggers[:3], 1):
            status += f"  Trigger {i}: {on/sr:.2f}s - {off/sr:.2f}s (duration: {(off-on)/sr:.2f}s)\n"
        status += f"\n⚠️ Possible earthquake activity!\n"
    else:
        status += f"✅ NO EARTHQUAKE ACTIVITY\n"
        status += f"Background noise levels normal.\n"
    
    status += f"\n🔄 Auto-updating every 10 seconds...\n"
    status += f"{'='*60}"
    
    return fig, status

def fetch_station_status(station_id, provider='IRIS'):
    """Fetch current status of a station with earthquake detection."""
    if not OBSPY_AVAILABLE:
        return None, "ObsPy not available"
    
    fetcher = RealtimeDataFetcher(provider)
    
    if not fetcher.client:
        return None, f"Could not connect to {provider}"
    
    trace = fetcher.fetch_latest(station_id, duration=60)
    
    if trace is None:
        return None, f"No data available from {station_id}"
    
    # Preprocess data
    data = trace.data.astype(np.float32)
    data_norm = data / (np.abs(data).max() + 1e-8)
    sr = trace.stats.sampling_rate
    
    # Run STA/LTA detection
    from obspy.signal.trigger import recursive_sta_lta, trigger_onset
    cft = recursive_sta_lta(data_norm, int(0.5 * sr), int(10.0 * sr))
    triggers = trigger_onset(cft, 3.5, 1.0)
    
    # Create plot with 2 subplots
    fig = Figure(figsize=(14, 8))
    
    # Subplot 1: Waveform
    ax1 = fig.add_subplot(2, 1, 1)
    time_axis = np.arange(len(trace.data)) / sr
    ax1.plot(time_axis, trace.data, 'b-', linewidth=0.8)
    ax1.set_ylabel('Amplitude', fontweight='bold', fontsize=11)
    ax1.set_title(f'🌍 Live Seismic Data: {station_id} - {datetime.now().strftime("%H:%M:%S")}', 
                 fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # Mark triggers on waveform
    if len(triggers) > 0:
        for on, off in triggers:
            ax1.axvspan(on/sr, off/sr, alpha=0.3, color='red', label='Trigger' if on == triggers[0][0] else '')
        if len(triggers) > 0:
            ax1.legend(loc='upper right')
    
    # Add statistics box
    stats_text = f"Sample Rate: {sr} Hz\n"
    stats_text += f"Duration: {len(trace.data)/sr:.1f}s\n"
    stats_text += f"Max Amplitude: {np.abs(trace.data).max():.0f}\n"
    stats_text += f"Triggers: {len(triggers)}"
    
    ax1.text(0.02, 0.98, stats_text,
            transform=ax1.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
            fontsize=10, fontweight='bold')
    
    # Subplot 2: STA/LTA
    ax2 = fig.add_subplot(2, 1, 2)
    cft_time = np.arange(len(cft)) / sr
    ax2.plot(cft_time, cft, 'g-', linewidth=1.5)
    ax2.axhline(y=3.5, color='red', linestyle='--', linewidth=2, label='Trigger ON (3.5)')
    ax2.axhline(y=1.0, color='blue', linestyle='--', linewidth=2, label='Trigger OFF (1.0)')
    ax2.set_xlabel('Time (seconds)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('STA/LTA Ratio', fontweight='bold', fontsize=11)
    ax2.set_title('Earthquake Detection (STA/LTA)', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Highlight trigger zones
    for on, off in triggers:
        ax2.axvspan(on/sr, off/sr, alpha=0.2, color='yellow')
    
    fig.tight_layout()
    
    # Generate detailed status
    status = f"{'='*60}\n"
    status += f"📡 STATION STATUS\n"
    status += f"{'='*60}\n\n"
    status += f"Station: {station_id}\n"
    status += f"Provider: {provider}\n"
    status += f"Status: ✅ ONLINE\n"
    status += f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    status += f"DATA INFO:\n"
    status += f"  • Samples: {len(trace.data)}\n"
    status += f"  • Sample rate: {sr} Hz\n"
    status += f"  • Duration: {len(trace.data)/sr:.1f} seconds\n"
    status += f"  • Amplitude range: [{data.min():.0f}, {data.max():.0f}]\n\n"
    
    status += f"DETECTION RESULTS:\n"
    if len(triggers) > 0:
        status += f"  🚨 {len(triggers)} TRIGGER(S) DETECTED!\n\n"
        for i, (on, off) in enumerate(triggers[:5], 1):
            time_on = on / sr
            time_off = off / sr
            duration = time_off - time_on
            status += f"  Trigger {i}:\n"
            status += f"    • Time: {time_on:.2f} - {time_off:.2f} seconds\n"
            status += f"    • Duration: {duration:.2f} seconds\n"
            status += f"    • Max ratio: {cft[on:off].max():.2f}\n\n"
        
        if len(triggers) > 5:
            status += f"  ... and {len(triggers) - 5} more\n\n"
        
        status += f"⚠️ INTERPRETATION:\n"
        status += f"  • Possible earthquake activity detected\n"
        status += f"  • Could be P-wave, S-wave, or surface waves\n"
        status += f"  • Verify with multiple stations\n"
    else:
        status += f"  ✅ NO EARTHQUAKE ACTIVITY\n"
        status += f"  Station recording normal background noise\n\n"
        status += f"This is typical - earthquakes are rare!\n"
    
    status += f"\n{'='*60}"
    
    return fig, status

def load_station_preset(preset_name):
    """Load preset station list."""
    if preset_name in STATION_PRESETS:
        return '\n'.join(STATION_PRESETS[preset_name])
    return ''

# ==============================
# WEB INTERFACE
# ==============================
def create_realtime_tab():
    """Create the real-time monitoring tab content."""
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 🌍 Real-Time Seismic Monitoring
            
            Monitor live earthquake data from global seismic stations.
            
            **Data Sources:**
            - IRIS (Global)
            - USGS (United States)
            - GEOFON (Europe)
            - ORFEUS (Europe)
            """)
            
            provider_dropdown = gr.Dropdown(
                choices=['IRIS', 'USGS', 'GEOFON', 'ORFEUS'],
                value='IRIS',
                label="Data Provider"
            )
            
            preset_dropdown = gr.Dropdown(
                choices=list(STATION_PRESETS.keys()),
                label="📍 Station Presets",
                value="🌐 Global Network"
            )
            
            stations_input = gr.Textbox(
                label="Stations to Monitor (one per line)",
                placeholder="IU.ANMO.00.BHZ\nIU.MAJO.00.BHZ\nG.CAN.00.BHZ",
                lines=5,
                value='\n'.join(STATION_PRESETS["🌐 Global Network"])
            )
            
            update_interval = gr.Slider(
                minimum=10,
                maximum=60,
                value=30,
                step=5,
                label="Update Interval (seconds)"
            )
            
            with gr.Row():
                start_btn = gr.Button("▶️ Start Monitoring", variant="primary")
                stop_btn = gr.Button("⏹️ Stop Monitoring", variant="stop")
            
            gr.Markdown("""
            ---
            ### 🧪 Test Single Station
            
            Test connection and view live data from a single station:
            """)
            
            test_station = gr.Textbox(
                label="Station ID",
                value="IU.ANMO.00.BHZ",
                placeholder="IU.ANMO.00.BHZ"
            )
            
            auto_refresh = gr.Checkbox(
                label="🔄 Auto-refresh (updates every 10s)",
                value=False,
                info="Continuously update the display"
            )
            
            test_btn = gr.Button("🔍 Fetch Live Data Now", variant="secondary")
        
        with gr.Column(scale=2):
            monitoring_log = gr.Textbox(
                label="📋 Monitoring Log",
                lines=8,
                max_lines=15,
                placeholder="Start monitoring to see live updates..."
            )
            
            gr.Markdown("### 📊 Multi-Station Live View")
            
            multi_station_plot = gr.Plot(
                label="🌍 Live Multi-Station Data (Auto-updates every 10s)"
            )
            
            multi_station_status = gr.Textbox(
                label="� Live Detection Results",
                lines=12,
                placeholder="Data will appear here when monitoring is active..."
            )
            
            gr.Markdown("---")
            gr.Markdown("### 🧪 Single Station Viewer")
            
            station_plot = gr.Plot(
                label="📊 Single Station Data"
            )
            
            station_status = gr.Textbox(
                label="📡 Station Status",
                lines=8
            )
    
    gr.Markdown("""
    ---
    ### 📖 How Real-Time Monitoring Works
    
    #### Data Flow:
    ```
    Seismic Station → FDSN Web Service → Your Computer → AI Analysis
         (Global)          (Real-time)        (Local)       (Instant)
    ```
    
    #### Detection Pipeline:
    1. **Data Fetch**: Every 10-60 seconds, fetch latest data
    2. **Classical Detection**: STA/LTA trigger algorithm
    3. **AI Classification**: Identify P, S, Surface waves
    4. **Alert**: Notify when earthquake waves detected
    
    #### Available Stations:
    - **IU Network**: Global Seismographic Network (100+ stations)
    - **G Network**: GEOSCOPE (France)
    - **BK Network**: Berkeley Digital Seismic Network (California)
    - **CI Network**: Caltech/USGS Southern California
    - **GE Network**: GEOFON (Germany)
    
    #### Station ID Format:
    `NETWORK.STATION.LOCATION.CHANNEL`
    
    Examples:
    - `IU.ANMO.00.BHZ` - Albuquerque, NM, USA
    - `IU.MAJO.00.BHZ` - Matsushiro, Japan  
    - `G.CAN.00.BHZ` - Canberra, Australia
    
    #### Channels:
    - `BHZ` = Broadband High-gain Vertical (best for earthquakes)
    - `BHN` = Broadband High-gain North
    - `BHE` = Broadband High-gain East
    
    #### Early Warning Potential:
    - **P-wave detection**: 3-20 seconds before S-wave
    - **Local earthquakes**: 5-15 seconds warning
    - **Regional earthquakes**: 15-60 seconds warning
    - **Distant earthquakes**: Minutes of warning
    
    ---
    
    ### ⚠️ Important Notes
    
    1. **Rate Limits**: FDSN services have rate limits. Don't poll too frequently.
    2. **Data Delay**: Real-time data may have 1-10 second delay.
    3. **Station Availability**: Not all stations are online 24/7.
    4. **Network Required**: Requires stable internet connection.
    
    ### 🚀 For Production Use
    
    For serious real-time monitoring, use the command-line tool:
    
    ```bash
    python3 realtime_monitor.py --stations IU.ANMO.00.BHZ,IU.MAJO.00.BHZ --provider IRIS
    ```
    
    This provides:
    - Multi-threaded monitoring
    - Continuous operation
    - Lower latency
    - Better error handling
    - Database logging
    """)
    
    # Event handlers
    preset_dropdown.change(
        fn=load_station_preset,
        inputs=[preset_dropdown],
        outputs=[stations_input]
    )
    
    # Multi-station monitoring
    start_btn.click(
        fn=start_realtime_monitoring,
        inputs=[stations_input, provider_dropdown, update_interval],
        outputs=[monitoring_log]
    )
    
    stop_btn.click(
        fn=stop_realtime_monitoring,
        inputs=[],
        outputs=[monitoring_log]
    )
    
    # Multi-station auto-update timer (updates every 10 seconds when monitoring is active)
    multi_timer = gr.Timer(10)
    multi_timer.tick(
        fn=fetch_multi_station_data,
        inputs=[gr.State(True)],  # Always try to update
        outputs=[multi_station_plot, multi_station_status]
    )
    
    # Single station testing
    test_btn.click(
        fn=fetch_station_status,
        inputs=[test_station, provider_dropdown],
        outputs=[station_plot, station_status]
    )
    
    # Single station auto-refresh functionality
    def auto_refresh_data(station, provider, should_refresh):
        """Automatically refresh data when auto-refresh is enabled."""
        if should_refresh:
            return fetch_station_status(station, provider)
        return None, "Auto-refresh disabled. Click 'Fetch Live Data Now' to manually update."
    
    # Single station timer (only when auto-refresh checkbox is enabled)
    single_timer = gr.Timer(10)
    single_timer.tick(
        fn=auto_refresh_data,
        inputs=[test_station, provider_dropdown, auto_refresh],
        outputs=[station_plot, station_status]
    )

if __name__ == "__main__":
    # Standalone launch for testing
    with gr.Blocks(title="Real-Time Seismic Monitoring") as app:
        gr.Markdown("# 🌍 Real-Time Seismic Monitoring")
        create_realtime_tab()
    
    app.launch(server_port=7861)
