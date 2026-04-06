#!/usr/bin/env python3
"""
Flask Web Server for 8-Channel Microphone Array Localization System
Provides real-time visualization and control of the localization algorithm.

Integrated features:
- safer engine startup defaults
- latest result HTTP API (/api/latest)
- optional cloud upload of localization results
- richer logging for start/stop debugging
"""

import sys
import os
import json
import math
import time
import logging
import argparse
from pathlib import Path
from threading import Thread, Event, Lock
from typing import Optional
from datetime import datetime, timezone
from queue import Queue, Empty
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError

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

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================
def _safe_num(x):
    """Convert NaN/Inf/numpy scalars to JSON-safe Python float/None."""
    try:
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass
    if x is None:
        return None
    try:
        xf = float(x)
        return xf if math.isfinite(xf) else None
    except Exception:
        return None


def _safe_int(x):
    try:
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


# =============================================================================
# Optional cloud uploader (self-contained, no extra module required)
# =============================================================================
class CloudUploader:
    def __init__(self):
        self.enabled = str(os.getenv("CLOUD_UPLOAD_ENABLED", "0")).strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.url = os.getenv("CLOUD_API_INGEST_URL", "").strip()
        self.token = os.getenv("CLOUD_API_INGEST_TOKEN", "").strip()
        self.source_name = os.getenv("CLOUD_SOURCE_NAME", "raspberry-pi").strip() or "raspberry-pi"
        self.queue: Queue = Queue(maxsize=1000)
        self.stop_event = Event()
        self.thread: Optional[Thread] = None

    def start(self):
        if not self.enabled:
            logger.info("Cloud uploader disabled")
            return
        if not self.url:
            logger.warning("Cloud uploader enabled, but CLOUD_API_INGEST_URL is empty; disabling uploader")
            self.enabled = False
            return
        self.stop_event.clear()
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        logger.info("Cloud uploader started -> %s", self.url)

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def enqueue_result(self, payload: dict):
        if not self.enabled:
            return
        try:
            self.queue.put_nowait(payload)
        except Exception:
            logger.warning("Cloud uploader queue full; dropping one payload")

    def _worker(self):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Ingest-Token"] = self.token

        logger.warning("DEBUG uploader token=%r", self.token)
        logger.warning("DEBUG uploader headers=%r", headers)
        logger.warning("DEBUG uploader url=%s", self.url)

        while not self.stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib_request.Request(
                    self.url,
                    data=data,
                    headers=headers,
                    method="POST",
                )

                with urllib_request.urlopen(req, timeout=8) as resp:
                    body = resp.read().decode("utf-8", errors="ignore")
                    logger.info(
                        "Cloud upload success: status=%s body=%s",
                        getattr(resp, "status", "unknown"),
                        body,
                    )

            except HTTPError as e:
                logger.warning(
                    "Cloud upload HTTPError code=%s url=%s",
                    e.code,
                    self.url,
                )
            except URLError as e:
                logger.warning(
                    "Cloud upload failed url=%s error=%s",
                    self.url,
                    e,
                )
            except Exception as e:
                logger.warning(
                    "Cloud upload failed url=%s error=%s",
                    self.url,
                    e,
                )


# =============================================================================
# Flask app setup
# =============================================================================
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / 'templates'),
    static_folder=str(PROJECT_ROOT / 'static')
)
app.config['SECRET_KEY'] = 'localization_secret_key'

# Keep threading mode to avoid extra runtime dependency.
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*",
    ping_interval=25,
    ping_timeout=60,
)


# =============================================================================
# Global runtime state
# =============================================================================
engine: Optional[LocalizationEngine] = None
engine_thread: Optional[Thread] = None
stop_event = Event()
state_lock = Lock()

DEFAULT_DEVICE_HINT = os.getenv("DEFAULT_DEVICE_HINT", "Yundea 8MICA").strip() or "Yundea 8MICA"
DEFAULT_BLOCKSIZE = int(os.getenv("DEFAULT_BLOCKSIZE", "1024"))
EMIT_INTERVAL_S = float(os.getenv("EMIT_INTERVAL_S", "1.0"))
STATUS_INTERVAL_S = float(os.getenv("STATUS_INTERVAL_S", "2.0"))

cloud_uploader = CloudUploader()

latest_result = {
    'timestamp': None,
    'wall_ts': None,
    'wall_time_iso': None,
    'angle': None,
    'distance': None,
    'snr_db': None,
    'coherence': None,
    'diameter_m': None,
    'update_flag': None,
    'error_plane_us': None,
    'error_near_us': None,
    'rho_hat': None,
    'valid': False,
    'engine_running': False,
    'frame_count': 0,
    'device_name': None,
    'channels': None,
    'samplerate': None,
}


def _set_latest_from_engine_status():
    global latest_result, engine
    with state_lock:
        latest_result['engine_running'] = bool(getattr(engine, 'running', False)) if engine is not None else False
        latest_result['frame_count'] = _safe_int(getattr(engine, 'frame_count', 0)) if engine is not None else 0
        latest_result['device_name'] = getattr(engine, 'device_name', None) if engine is not None else None
        latest_result['channels'] = getattr(engine, 'channels', None) if engine is not None else None
        latest_result['samplerate'] = getattr(engine, 'samplerate', None) if engine is not None else None


def _build_cloud_payload(result, now_wall: float, wall_iso: str) -> dict:
    # loudness_db: prefer explicit result value if present; fallback to snr_db as intensity proxy
    loudness_db = _safe_num(getattr(result, 'loudness_db', None))
    if loudness_db is None:
        loudness_db = _safe_num(getattr(result, 'snr_db', None))

    return {
        'source_name': cloud_uploader.source_name,
        'ts_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        'unix_ts': _safe_num(now_wall),
        'wall_time_iso': wall_iso,
        'angle_deg': _safe_num(getattr(result, 'angle_deg', None)),
        'distance_m': _safe_num(getattr(result, 'distance_m', None)),
        'loudness_db': loudness_db,
        'snr_db': _safe_num(getattr(result, 'snr_db', None)),
        'coherence': _safe_num(getattr(result, 'coherence', None)),
        'diameter_m': _safe_num(getattr(result, 'diameter_m', None)),
        'valid': bool(getattr(result, 'valid', False)),
    }


# =============================================================================
# Engine worker
# =============================================================================
def engine_worker():
    global engine

    if engine is None:
        logger.error("Engine not initialized")
        return

    logger.info("Localization engine worker started")
    last_emit_wall = 0.0
    last_status_wall = 0.0

    try:
        while not stop_event.is_set():
            result = engine.process_frame()
            now_wall = time.time()

            if result is not None and (now_wall - last_emit_wall) >= EMIT_INTERVAL_S:
                last_emit_wall = now_wall
                wall_iso = datetime.now().astimezone().isoformat(timespec='milliseconds')

                payload = {
                    'type': 'data',
                    'timestamp': _safe_num(result.timestamp),
                    'wall_ts': _safe_num(now_wall),
                    'wall_time_iso': wall_iso,
                    'angle': _safe_num(result.angle_deg),
                    'distance': _safe_num(result.distance_m),
                    'snr_db': _safe_num(result.snr_db),
                    'coherence': _safe_num(result.coherence),
                    'diameter_m': _safe_num(result.diameter_m),
                    'update_flag': _safe_int(result.update_flag),
                    'error_plane_us': _safe_num(result.error_plane_us),
                    'error_near_us': _safe_num(result.error_near_us),
                    'rho_hat': _safe_num(result.rho_hat),
                    'valid': bool(result.valid),
                }

                with state_lock:
                    latest_result.update(payload)
                    latest_result['engine_running'] = bool(getattr(engine, 'running', False))
                    latest_result['frame_count'] = _safe_int(getattr(engine, 'frame_count', 0))
                    latest_result['device_name'] = getattr(engine, 'device_name', None)
                    latest_result['channels'] = getattr(engine, 'channels', None)
                    latest_result['samplerate'] = getattr(engine, 'samplerate', None)

                socketio.emit('localization_data', payload)
                cloud_uploader.enqueue_result(_build_cloud_payload(result, now_wall, wall_iso))

            if (now_wall - last_status_wall) >= STATUS_INTERVAL_S:
                last_status_wall = now_wall
                status_payload = {
                    'type': 'status',
                    'running': bool(getattr(engine, 'running', False)),
                    'frame_count': _safe_int(getattr(engine, 'frame_count', 0)),
                    'device_info': {
                        'name': getattr(engine, 'device_name', None),
                        'channels': getattr(engine, 'channels', None),
                        'samplerate': getattr(engine, 'samplerate', None)
                    }
                }
                socketio.emit('engine_status', status_payload)
                _set_latest_from_engine_status()

            socketio.sleep(0.05)

    except Exception:
        logger.exception("Engine worker crashed")
        socketio.emit('error', {'type': 'error', 'message': 'Engine worker crashed'})
        try:
            if engine is not None:
                engine.cleanup()
        except Exception:
            pass
    finally:
        _set_latest_from_engine_status()
        logger.info("Localization engine worker stopped")


# =============================================================================
# Routes
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ping')
def ping():
    return jsonify({'ok': True, 'ts': time.time()})


@app.route('/api/latest')
def get_latest():
    with state_lock:
        return jsonify(dict(latest_result))


@app.route('/api/status')
def get_status():
    global engine

    if engine is None:
        return jsonify({
            'running': False,
            'message': 'Engine not initialized'
        })

    running = bool(getattr(engine, 'running', False))
    stream_active = bool(getattr(engine, 'stream', None) is not None)

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
    global engine

    if request.method == 'GET':
        default_algorithm = {
            'c_sound': C_SOUND,
            'rhoL': 0.10,
            'rhoH': 2.00,
            'angle_smooth': 0.85,
            'rho_smooth': 0.35,
            'coh_th': 1.10,
            'vad_th': 6.0,
            'default_device_hint': DEFAULT_DEVICE_HINT,
            'default_blocksize': DEFAULT_BLOCKSIZE,
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
                    'dia_init_meters': engine.diameter_m,
                    'device_name': engine.device_name,
                },
                'algorithm': {
                    'c_sound': engine.c_sound,
                    'rhoL': engine.rhoL,
                    'rhoH': engine.rhoH,
                    'angle_smooth': engine.angle_smooth,
                    'rho_smooth': engine.rho_smooth,
                    'coh_th': engine.cohTh,
                    'vad_th': engine.vadTh,
                    'default_device_hint': DEFAULT_DEVICE_HINT,
                    'default_blocksize': DEFAULT_BLOCKSIZE,
                }
            })

    if engine is None:
        return jsonify({'error': 'Engine not initialized'}), 400

    try:
        data = request.get_json(force=True)

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
    global engine, engine_thread, stop_event

    if engine is not None and engine_thread is not None and engine_thread.is_alive():
        return jsonify({'message': 'Engine already running', 'running': True})

    try:
        req_json = request.get_json(silent=True) or {}
        device_name_hint = (req_json.get('device_name_hint') or DEFAULT_DEVICE_HINT)
        blocksize = int(req_json.get('blocksize', DEFAULT_BLOCKSIZE))

        logger.info(
            "Starting engine with device_name_hint=%r, blocksize=%s, remote_addr=%s, user_agent=%r",
            device_name_hint, blocksize, request.remote_addr, request.headers.get('User-Agent')
        )

        device_config = DeviceConfig(
            device_name_hint=device_name_hint,
            min_channels=8,
            blocksize=blocksize,
            stream_channels=req_json.get('stream_channels'),
            channel_map=req_json.get('channel_map')
        )

        algo_config = AlgorithmConfig(
            c_sound=req_json.get('c_sound', C_SOUND),
            dia_init_meters=req_json.get('diameter', DIA_INIT_METERS)
        )

        engine = LocalizationEngine(device_config, algo_config)
        engine.start()

        stop_event.clear()
        engine_thread = Thread(target=engine_worker, daemon=True)
        engine_thread.start()

        _set_latest_from_engine_status()
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
        logger.error("Failed to start engine: %s", e)
        try:
            if engine is not None:
                engine.cleanup()
        except Exception:
            pass
        engine = None
        _set_latest_from_engine_status()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_engine():
    global engine, engine_thread, stop_event

    logger.info(
        "Stop requested: remote_addr=%s, user_agent=%r, origin=%r, referer=%r",
        request.remote_addr,
        request.headers.get('User-Agent'),
        request.headers.get('Origin'),
        request.headers.get('Referer'),
    )

    stop_event.set()

    if engine is not None:
        try:
            engine.cleanup()
        finally:
            engine = None

    if engine_thread is not None:
        engine_thread.join(timeout=2.0)
        engine_thread = None

    _set_latest_from_engine_status()
    logger.info("Localization engine stopped")
    return jsonify({'success': True, 'message': 'Engine stopped'})


# =============================================================================
# Socket handlers (kept for compatibility)
# =============================================================================
@socketio.on('connect')
def handle_connect():
    logger.info("Client connected: %s", request.sid)
    emit('connected', {'sid': request.sid, 'message': 'Connected to localization server'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected: %s", request.sid)


@socketio.on('request_status')
def handle_status_request():
    if engine is not None:
        emit('engine_status', {
            'type': 'status',
            'running': bool(getattr(engine, 'running', False)),
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


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='8-Channel Localization Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--list-devices', action='store_true', help='List available audio devices')
    args = parser.parse_args()

    try:
        validate_parameters()
        logger.info("Configuration parameters validated successfully")
    except ValueError as e:
        logger.error("Configuration validation failed: %s", e)
        sys.exit(1)

    if args.list_devices:
        from localization import list_audio_devices
        list_audio_devices()
        return

    cloud_uploader.start()

    print("\n" + "=" * 60)
    print("  8-Channel Microphone Array Localization Server")
    print("=" * 60)
    print(f"  Web Interface: http://{args.host}:{args.port}")
    print("  API Endpoints:")
    print("    - GET  /api/status  - System status")
    print("    - GET  /api/latest  - Latest localization result")
    print("    - GET  /api/config  - Get configuration")
    print("    - POST /api/config  - Update configuration")
    print("    - POST /api/start   - Start engine")
    print("    - POST /api/stop    - Stop engine")
    print("    - GET  /api/ping    - Liveness probe")
    print("=" * 60 + "\n")

    try:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True
        )
    finally:
        cloud_uploader.stop()


if __name__ == '__main__':
    main()
