# 8-Channel Microphone Array Localization System

A real-time sound source localization system designed for deployment on Raspberry Pi. This web-based application provides live visualization and control of an 8-channel microphone array using TDOA (Time Difference of Arrival) estimation with near-field distance correction.

## Features

- **Real-time Localization**: Continuous estimation of sound source direction (angle) and distance
- **Web-based Interface**: Modern browser-based visualization dashboard
- **Live Data Streaming**: WebSocket-powered real-time data updates (20 FPS)
- **Interactive Controls**: Adjust algorithm parameters through the web interface
- **Comprehensive Visualization**:
  - Polar plot showing source direction relative to microphone array
  - Time series charts for angle and distance history
  - Signal quality indicators (SNR, coherence)
  - Error metrics display

## Project Structure

```
raspberry_pi_localization/
├── app.py                    # Flask server with WebSocket support
├── localization.py           # Core localization algorithm
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Main web interface
└── static/
    ├── css/
    │   └── style.css        # Modern dark theme styles
    └── js/
        └── app.js           # WebSocket client and visualization
```

## Installation

### 1. System Dependencies (Raspberry Pi)

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-numpy python3-scipy
sudo apt-get install -y portaudio19-dev python3-pyaudio
sudo apt-get install -y build-essential swig libasound2-dev
```

### 2. Python Environment

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Packages

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
python app.py --host 0.0.0.0 --port 5000
```

Or for development with debug mode:

```bash
python app.py --host 0.0.0.0 --port 5000 --debug
```

### Accessing the Web Interface

Open your browser and navigate to:

```
http://<raspberry-pi-ip>:5000
```

### Controls

- **Start Engine**: Begin audio capture and localization
- **Stop Engine**: Stop the localization process
- **Clear Charts**: Reset time series data
- **Parameters Panel**: Adjust algorithm settings in real-time

### Keyboard Shortcuts

- `Ctrl+S` / `Cmd+S`: Start engine
- `Ctrl+X` / `Cmd+X`: Stop engine
- `Escape`: Clear charts

## API Endpoints

### GET /api/status
Returns current system status.

```json
{
  "running": true,
  "device_name": "Device Name",
  "channels": 8,
  "samplerate": 48000.0,
  "current_diameter": 0.18,
  "frame_count": 1234
}
```

### GET /api/config
Returns current algorithm configuration.

### POST /api/config
Update algorithm parameters.

Request body:
```json
{
  "diameter": 0.180,
  "c_sound": 343.0,
  "angle_smooth": 0.85,
  "rho_smooth": 0.35,
  "coh_th": 1.10,
  "vad_th": 6.0
}
```

### POST /api/start
Start the localization engine.

### POST /api/stop
Stop the localization engine.

## WebSocket Events

### Emitted by Server

- `localization_data`: Real-time localization results
- `engine_status`: Engine status updates
- `config_update`: Configuration change notifications
- `error`: Error messages

### Received by Server

- `request_status`: Request current engine status
- `update_diameter`: Update array diameter
- `connect`: Client connected
- `disconnect`: Client disconnected

## Configuration

The system can be configured via:

1. **Diameter File**: Create `diameter_m.txt` in the project directory with the desired diameter in meters
2. **Web Interface**: Use the parameters panel to adjust settings
3. **API Calls**: Send configuration updates via HTTP POST

## Algorithm Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| c_sound | 343.0 | 300-400 | Speed of sound in m/s |
| dia_init_meters | 0.180 | 0.06-0.6 | Initial array diameter |
| coh_th | 1.10 | 0.5-3.0 | Coherence threshold |
| vad_th | 6.0 | 0-20 | VAD threshold in dB |
| angle.85 | _smooth | 00-1 | Angle smoothing factor |
| rho_smooth | 0.35 | 0-1 | Distance smoothing factor |

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires modern browser with:
- WebSocket support
- Canvas API
- ES6 JavaScript

## Performance Considerations

- **Recommended Hardware**: Raspberry Pi 4 or newer with 2GB+ RAM
- **Audio Latency**: Configured for low-latency operation (~5ms)
- **Network**: Use wired connection for minimal latency in remote access scenarios
- **Browser**: Close unnecessary tabs for optimal performance

## Troubleshooting

### Audio Device Not Found

List available audio devices:
```bash
python app.py --list-devices
```

### Permission Issues

Ensure user is in audio group:
```bash
sudo usermod -a -G audio $USER
```

### WebSocket Connection Failed

- Check firewall settings (port 5000)
- Verify Raspberry Pi IP address
- Ensure server is running

### High CPU Usage

- Reduce chart update frequency in `app.js`
- Lower audio blocksize when starting engine
- Use Chrome/Firefox for better JavaScript performance

## File Monitoring

The system can monitor a diameter file for automatic updates:

1. Create `diameter_m.txt` in the project directory
2. Write the desired diameter in meters (e.g., `0.220`)
3. The system will detect changes and update geometry automatically

## License

This project is part of Final Year Project (FYP) 2026.

## Credits

Based on the realtime 8-channel localization algorithm with:
- GCC-PHAT for TDOA estimation
- Near-field distance correction with grid search
- Adaptive noise floor tracking
- Optimized for low-latency operation
