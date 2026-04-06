#!/usr/bin/env python3
"""
Flask Web Server for 8-Channel Microphone Array Localization System
Provides real-time visualization and control of the localization algorithm
"""

import sys
import logging
from pathlib import Path
from threading import Thread, Event
import argparse
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from localization import (
    LocalizationEngine, DeviceConfig, AlgorithmConfig,
    CHANNELS, SAMPLERATE, C_SOUND, DIA_INIT_METERS,
    validate_parameters
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask application setup
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / 'templates'),
    static_folder=str(PROJECT_ROOT / 'static')
)
app.config['SECRET_KEY'] = 'localization_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global engine instance
engine: Optional[LocalizationEngine] = None
engine_thread: Optional[Thread] = None
stop_event = Event()


def engine_worker():
    """
    Background worker that runs the localization engine
    and emits results via WebSocket
    """
    global engine

    if engine is None:
        logger.error("Engine not initialized")
        return

    logger.info("Localization engine worker started")

    try:
        while not stop_event.is_set():
            # Process one frame
            result = engine.process_frame()

            if result is not None:
                # Emit result via WebSocket
                socketio.emit('localization_data', {
                    'type': 'data',
                    'timestamp': result.timestamp,
                    'angle': result.angle_deg,
                    'distance': result.distance_m,
                    'snr_db': result.snr_db,
                    'coherence': result.coherence,
                    'diameter_m': result.diameter_m,
                    'update_flag': result.update_flag,
                    'error_plane_us': result.error_plane_us,
                    'error_near_us': result.error_near_us,
                    'rho_hat': result.rho_hat,
                    'valid': result.valid
                })

                # Emit status updates less frequently
                if hasattr(engine, 'frame_count') and engine.frame_count % 100 == 0:
                    socketio.emit('engine_status', {
                        'type': 'status',
                        'running': bool(getattr(engine, "running", False)),
                        'frame_count': engine.frame_count,
                        'device_info': {
                            'name': engine.device_name,
                            'channels': engine.channels,
                            'samplerate': engine.samplerate
                        }
                    })

            # Control emission rate (20 FPS = 50ms interval)
            socketio.sleep(0.05)

    except Exception as e:
        logger.error(f"Engine worker error: {e}")
        socketio.emit('error', {'type': 'error', 'message': str(e)})

        # 发生异常时尽量释放音频资源，避免再次启动失败
        try:
            if engine is not None:
                engine.cleanup()
        except Exception:
            pass

    finally:
        logger.info("Localization engine worker stopped")


@app.route('/')
def index():
    """Serve the main visualization page"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current system status"""
    global engine

    if engine is None:
        return jsonify({
            'running': False,
            'message': 'Engine not initialized'
        })

    # 这里的 running 反映真实采集状态，而不是“engine对象存在”
    running = bool(getattr(engine, "running", False))
    stream_active = bool(getattr(engine, "stream", None) is not None)

    return jsonify({
        'running': running,
        'stream_active': stream_active,
        'device_name': engine.device_name,
        'channels': engine.channels,
        'samplerate': engine.samplerate,
        'current_diameter': engine.diameter_m,
        'frame_count': getattr(engine, 'frame_count', 0)
    })


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update algorithm configuration"""
    global engine

    if request.method == 'GET':
        # Default configuration values
        default_algorithm = {
            'c_sound': C_SOUND,
            'rhoL': 0.10,
            'rhoH': 2.00,
            'angle_smooth': 0.85,
            'rho_smooth': 0.35,
            'coh_th': 1.10,
            'vad_th': 6.0
        }

        if engine is None:
            return jsonify({
                'device': {
                    'channels': CHANNELS,
                    'samplerate': SAMPLERATE if SAMPLERATE else 48000.0,
                    'dia_init_meters': DIA_INIT_METERS
                },
                'algorithm': default_algorithm
            })
        else:
            return jsonify({
                'device': {
                    'channels': engine.channels,
                    'samplerate': engine.samplerate,
                    'dia_init_meters': engine.diameter_m
                },
                'algorithm': {
                    'c_sound': engine.c_sound,
                    'rhoL': engine.rhoL,
                    'rhoH': engine.rhoH,
                    'angle_smooth': engine.angle_smooth,
                    'rho_smooth': engine.rho_smooth,
                    'coh_th': engine.cohTh,
                    'vad_th': engine.vadTh
                }
            })

    # POST - update configuration
    if engine is None:
        return jsonify({'error': 'Engine not initialized'}), 400

    try:
        data = request.get_json()

        # Update allowed parameters
        if 'diameter' in data:
            new_dia = float(data['diameter'])
            if engine.validate_diameter(new_dia):
                engine.set_diameter(new_dia)
                socketio.emit('config_update', {
                    'type': 'config',
                    'parameter': 'diameter',
                    'value': new_dia
                })

        if 'c_sound' in data:
            engine.c_sound = float(data['c_sound'])

        if 'angle_smooth' in data:
            engine.angle_smooth = float(data['angle_smooth'])

        if 'rho_smooth' in data:
            engine.rho_smooth = float(data['rho_smooth'])

        if 'coh_th' in data:
            engine.cohTh = float(data['coh_th'])

        if 'vad_th' in data:
            engine.vadTh = float(data['vad_th'])

        return jsonify({'success': True, 'message': 'Configuration updated'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/start', methods=['POST'])
def start_engine():
    """Start the localization engine"""
    global engine, engine_thread, stop_event

    if engine is not None and engine_thread is not None and engine_thread.is_alive():
        return jsonify({'message': 'Engine already running', 'running': True})

    try:
        # Initialize engine with proper device selection
        device_config = DeviceConfig(
            device_name_hint=(request.json.get("device_name_hint") if request.is_json else None),  # None = auto-select best >=8ch device
            min_channels=8,
            blocksize=request.json.get('blocksize', 256) if request.is_json else 256
        )

        algo_config = AlgorithmConfig(
            c_sound=request.json.get('c_sound', C_SOUND) if request.is_json else C_SOUND,
            dia_init_meters=request.json.get('diameter', DIA_INIT_METERS) if request.is_json else DIA_INIT_METERS
        )

        engine = LocalizationEngine(device_config, algo_config)

        # ✅ 关键修复：必须先启动音频流，否则 process_frame() 永远返回 None
        engine.start()

        # Reset stop event and start worker thread
        stop_event.clear()
        engine_thread = Thread(target=engine_worker, daemon=True)
        engine_thread.start()

        logger.info("Localization engine started")

        return jsonify({
            'success': True,
            'message': 'Engine started successfully',
            'device': {
                'name': engine.device_name,
                'channels': engine.channels,
                'samplerate': engine.samplerate
            }
        })

    except Exception as e:
        logger.error(f"Failed to start engine: {e}")

        # 启动失败时清理，避免占用声卡导致下次启动失败
        try:
            if engine is not None:
                engine.cleanup()
        except Exception:
            pass
        engine = None

        return jsonify({'error': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_engine():
    """Stop the localization engine"""
    global engine, engine_thread, stop_event

    stop_event.set()

    if engine is not None:
        engine.cleanup()
        engine = None

    if engine_thread is not None:
        engine_thread.join(timeout=2.0)
        engine_thread = None

    logger.info("Localization engine stopped")

    return jsonify({
        'success': True,
        'message': 'Engine stopped'
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'sid': request.sid, 'message': 'Connected to localization server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    if engine is not None:
        emit('engine_status', {
            'type': 'status',
            'running': bool(getattr(engine, "running", False)),
            'frame_count': getattr(engine, 'frame_count', 0),
            'device_info': {
                'name': engine.device_name,
                'channels': engine.channels,
                'samplerate': engine.samplerate
            }
        })
    else:
        emit('engine_status', {
            'type': 'status',
            'running': False,
            'frame_count': 0
        })


@socketio.on('update_diameter')
def handle_diameter_update(data):
    """Handle diameter update from client"""
    global engine

    if engine is None:
        emit('error', {'message': 'Engine not running'})
        return

    try:
        new_dia = float(data.get('diameter', engine.diameter_m))
        if engine.validate_diameter(new_dia):
            engine.set_diameter(new_dia)
            emit('config_update', {
                'type': 'config',
                'parameter': 'diameter',
                'value': new_dia
            })
        else:
            emit('error', {'message': 'Invalid diameter value'})
    except Exception as e:
        emit('error', {'message': str(e)})


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='8-Channel Localization Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--list-devices', action='store_true', help='List available audio devices')

    args = parser.parse_args()

    # Validate parameters
    try:
        validate_parameters()
        logger.info("Configuration parameters validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    if args.list_devices:
        from localization import list_audio_devices
        list_audio_devices()
        return

    # Print startup info
    print("\n" + "=" * 60)
    print("  8-Channel Microphone Array Localization Server")
    print("=" * 60)
    print(f"  Web Interface: http://{args.host}:{args.port}")
    print(f"  API Endpoints:")
    print(f"    - GET  /api/status  - System status")
    print(f"    - GET  /api/config  - Get configuration")
    print(f"    - POST /api/config  - Update configuration")
    print(f"    - POST /api/start   - Start engine")
    print(f"    - POST /api/stop    - Stop engine")
    print("=" * 60 + "\n")

    # Run Flask-SocketIO server
    socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        allow_unsafe_werkzeug=True
    )


if __name__ == '__main__':
    main()
