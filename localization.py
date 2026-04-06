#!/usr/bin/env python3
"""
8-Channel Microphone Array Localization Algorithm

This module implements real-time sound source localization using TDOA
(Time Difference of Arrival) estimation with near-field distance correction.

Key Features:
- GCC-PHAT for robust TDOA estimation
- Near-field distance estimation with grid search and refinement
- Adaptive noise floor tracking
- Real-time parameter adjustment
- Support for variable diameter microphone arrays
"""

import sys
import math
import time
import queue
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Deque, Dict, Any
from pathlib import Path
from collections import deque
from threading import Lock

import numpy as np
from scipy.optimize import least_squares
import sounddevice as sd
from itertools import combinations


# =============================================================================
# Configuration Constants
# =============================================================================

# Audio Configuration
CHANNELS: int = 8
SAMPLERATE: Optional[float] = None
DEFAULT_BLOCKSIZE: int = 256
QUEUE_MAX_SIZE: int = 3
DROP_OLD_BLOCKS: bool = True

# Device Configuration
LOCK_DEVICE_NAME_HINT: str = "YDM8MIC"
LOCK_MIN_INPUT_CHANNELS: int = 8

# Microphone Array Geometry
DIA_INIT_METERS: float = 0.180  # Initial array diameter in meters
DIA_MIN_METERS: float = 0.060
DIA_MAX_METERS: float = 0.600

# Diameter File Monitoring
DIAMETER_FILE: str = "diameter_m.txt"
DIAMETER_FILE_POLL_SEC: float = 0.10

# Algorithm Parameters
C_SOUND: float = 343.0  # Speed of sound in m/s

# Distance estimation bounds
RHO_LOW: float = 0.10   # Minimum distance in meters
RHO_HIGH: float = 2.00  # Maximum distance in meters

# Smoothing factors (0.0 = no smoothing, 1.0 = complete smoothing)
ANGLE_SMOOTH: float = 0.85
RHO_SMOOTH: float = 0.35

# Detection thresholds
COH_THRESHOLD: float = 1.10    # Coherence threshold for valid detection
VAD_THRESHOLD: float = 6.0     # Voice Activity Detection threshold in dB

# Noise floor adaptation
NOISE_ADAPTATION: float = 0.985
NOISE_FLOOR_MIN: float = 1e-4

# Grid search parameters for distance estimation
GRID_POINTS: int = 25          # Number of grid points
GRID_MARGIN: float = 0.20      # +/- margin around current estimate
GRID_WIDE_INTERVAL: int = 10   # Frames between wide grid searches

# Error thresholds for distance update
ABS_ERR_US: float = 480.0      # Maximum acceptable error in microseconds
MIN_IMPROVE_US: float = 30.0   # Minimum improvement required for update

# Smoothing window
K_SMOOTH: int = 5              # Moving average window size
TIME_WINDOW: float = 10.0      # Time window for plots in seconds

# Performance settings
MAX_HISTORY_POINTS: int = 2000
ENABLE_TAU_CACHE: bool = True


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DeviceConfig:
    """Configuration for audio device selection"""
    device_name_hint: Optional[str] = None
    min_channels: int = 8
    blocksize: int = 256
    # Optional: number of channels to request from the audio backend (e.g., PipeWire may expose 64ch)
    # If None, the engine will request the device's max_input_channels.
    stream_channels: Optional[int] = None
    # Optional: select/reorder 8 microphone channels from the stream. Length must be 8.
    channel_map: Optional[list[int]] = None


@dataclass
class AlgorithmConfig:
    """Configuration for localization algorithm parameters"""
    c_sound: float = C_SOUND
    dia_init_meters: float = DIA_INIT_METERS
    rhoL: float = RHO_LOW
    rhoH: float = RHO_HIGH
    angle_smooth: float = ANGLE_SMOOTH
    rho_smooth: float = RHO_SMOOTH
    coh_th: float = COH_THRESHOLD
    vad_th: float = VAD_THRESHOLD


@dataclass
class LocalizationResult:
    """Result of a single localization frame"""
    timestamp: float
    angle_deg: float
    distance_m: float
    snr_db: float
    coherence: float
    diameter_m: float
    update_flag: bool
    error_plane_us: float
    error_near_us: float
    rho_hat: float
    valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'angle_deg': self.angle_deg,
            'distance_m': self.distance_m,
            'snr_db': self.snr_db,
            'coherence': self.coherence,
            'diameter_m': self.diameter_m,
            'update_flag': self.update_flag,
            'error_plane_us': self.error_plane_us,
            'error_near_us': self.error_near_us,
            'rho_hat': self.rho_hat,
            'valid': self.valid
        }


# =============================================================================
# Utility Functions
# =============================================================================

def hann_window(N: int) -> np.ndarray:
    """
    Generate Hann window of length N
    
    Args:
        N: Window length
        
    Returns:
        Hann window array of shape (N, 1)
    """
    n = np.arange(N, dtype=float)
    return 0.5 * (1 - np.cos(2 * np.pi * n / N))


def wrap_angle_180(deg: float) -> float:
    """
    Wrap angle to [-180, 180) range
    
    Args:
        deg: Angle in degrees
        
    Returns:
        Wrapped angle in degrees
    """
    return (deg + 180.0) % 360.0 - 180.0


def parabolic_peak(y: np.ndarray, i: int) -> float:
    """
    Interpolate peak position using parabolic fitting
    
    Args:
        y: Array of values
        i: Index of maximum element
        
    Returns:
        Fractional offset from i to true peak
    """
    N = y.size
    iL = (i - 1) % N
    i0 = i % N
    iR = (i + 1) % N
    denom = (y[iL] - 2 * y[i0] + y[iR] + np.finfo(float).eps)
    return (y[iL] - y[iR]) / (2 * denom)


def pair_differences(mic: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """
    Compute microphone pair position differences
    
    Args:
        mic: Microphone positions (M, 2)
        pairs: Pair indices (P, 2)
        
    Returns:
        Position difference vectors (P, 2)
    """
    return mic[pairs[:, 1]] - mic[pairs[:, 0]]


def clamp(value: float, lo: float, hi: float) -> float:
    """
    Clamp value to [lo, hi] range
    
    Args:
        value: Value to clamp
        lo: Lower bound
        hi: Upper bound
        
    Returns:
        Clamped value
    """
    return max(lo, min(hi, value))


def try_read_diameter_file(path: str) -> Optional[float]:
    """
    Try to read diameter value from file
    
    Args:
        path: Path to diameter file
        
    Returns:
        Diameter in meters if file exists and is valid, None otherwise
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            value = float(f.read().strip())
        return value
    except (FileNotFoundError, ValueError, IOError):
        return None


# =============================================================================
# Core Algorithm Functions
# =============================================================================

def estimate_direction_ls(mic: np.ndarray, pairs: np.ndarray, 
                          c: float, tdoa: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Estimate sound source direction using least squares (far-field approximation)
    
    Args:
        mic: Microphone positions (M, 2)
        pairs: Microphone pair indices (P, 2)
        c: Speed of sound
        tdoa: Time difference of arrival values (P,)
        
    Returns:
        Tuple of (unit direction vector, angle in degrees)
    """
    # Compute pair position differences
    A = pair_differences(mic, pairs)
    
    # Solve least squares problem
    b = c * tdoa.reshape(-1, 1)
    u_est, *_ = np.linalg.lstsq(A, b, rcond=None)
    
    # Normalize to unit vector
    u = u_est.flatten()
    u /= (np.linalg.norm(u) + 1e-12)
    
    # Convert to angle
    angle_deg = math.degrees(math.atan2(u[1], u[0]))
    
    return u, angle_deg


def gcc_phat_tdoa(xw: np.ndarray, pairs: np.ndarray, 
                  N: int, fs: float, tau_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute TDOA using GCC-PHAT (Generalized Cross-Correlation with Phase Transform)
    
    Args:
        xw: Windowed input signal (samples, channels)
        pairs: Microphone pair indices (P, 2)
        N: FFT size
        fs: Sample rate
        tau_max: Maximum TDOA to consider
        
    Returns:
        Tuple of (TDOA values, coherence weights)
    """
    P = pairs.shape[0]
    tdoa = np.zeros(P, dtype=float)
    weights = np.ones(P, dtype=float)
    
    for p in range(P):
        i, j = pairs[p]
        
        # Compute FFT of both signals
        Xi = np.fft.fft(xw[:, i], n=N)
        Xj = np.fft.fft(xw[:, j], n=N)
        
        # GCC-PHAT weighting
        R = Xi * np.conj(Xj)
        R /= (np.abs(R) + np.finfo(float).eps)
        
        # Cross-correlation
        cc = np.fft.ifft(R, n=N).real
        cc = np.fft.fftshift(cc)
        
        # Find peak with sub-sample interpolation
        idx = int(np.argmax(np.abs(cc)))
        d = parabolic_peak(cc, idx)
        lag = (idx - (N // 2)) + d
        
        # Compute weight based on peak sharpness
        peak_val = float(np.max(np.abs(cc)))
        med_val = float(np.median(np.abs(cc)) + np.finfo(float).eps)
        weights[p] = max(0.2, min(8.0, peak_val / med_val))
        
        # Convert to time and clamp
        t = lag / fs
        tdoa[p] = max(-tau_max, min(t, tau_max))
    
    return tdoa, weights


def scores_pair_1d(
    Xi: np.ndarray,
    fs: float,
    N: int,
    c: float,
    mic: np.ndarray,
    u: np.ndarray,
    rhos: np.ndarray,
    fmin: float,
    fmax: float,
    pairs_full: np.ndarray,
) -> np.ndarray:
    """SRP-PHAT-like 1D score along direction u for candidate ranges rhos."""
    fk = np.arange(N, dtype=float) * (fs / N)
    band = (fk >= fmin) & (fk <= fmax)
    fk = fk[band].reshape(-1, 1)

    X = Xi[band, :] / (np.abs(Xi[band, :]) + np.finfo(float).eps)

    P = pairs_full.shape[0]
    R = np.empty((fk.shape[0], P), dtype=complex)
    for p, (i, j) in enumerate(pairs_full):
        R[:, p] = X[:, i] * np.conj(X[:, j])

    scores = np.zeros(rhos.size, dtype=float)
    u_row = np.asarray(u, dtype=float).reshape(1, -1)

    for k, rho in enumerate(rhos):
        s = rho * u_row  # (1,2)
        D = mic - s
        dist = np.linalg.norm(D, axis=1)
        tau = dist / c
        tauij = tau[pairs_full[:, 1]] - tau[pairs_full[:, 0]]  # (P,)
        Ftau = fk @ tauij.reshape(1, -1)
        W = np.exp(-1j * 2 * math.pi * Ftau)
        scores[k] = float(np.real(np.sum(R * W)))

    return scores


def estimate_pos_tdoa_ls(
    mic: np.ndarray,
    pairs: np.ndarray,
    c: float,
    tdoa: np.ndarray,
    weights: np.ndarray,
    init_xy: np.ndarray,
    bounds_xy: float = 2.0,
):
    """Estimate 2D source position by weighted LS on near-field TDoA."""
    w = np.asarray(weights, dtype=float).copy()
    w = np.sqrt(np.clip(w / (np.median(w) + 1e-12), 0.3, 3.0))

    def residual(s):
        s = np.asarray(s, dtype=float).reshape(1, 2)
        dist = np.linalg.norm(mic - s, axis=1)
        tau = dist / c
        tauij = tau[pairs[:, 1]] - tau[pairs[:, 0]]
        return (tauij - tdoa) * w

    lb = np.array([-bounds_xy, -bounds_xy], dtype=float)
    ub = np.array([bounds_xy, bounds_xy], dtype=float)

    return least_squares(
        residual,
        x0=np.asarray(init_xy, dtype=float),
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1e-4,
        max_nfev=40,
    )


def nearfield_tdoa(mic: np.ndarray, pairs: np.ndarray, 
                   c: float, u: np.ndarray, rho: float) -> np.ndarray:
    """
    Compute expected TDOA for near-field source at given distance
    
    Args:
        mic: Microphone positions (M, 2)
        pairs: Microphone pair indices (P, 2)
        c: Speed of sound
        u: Unit direction vector (2,)
        rho: Distance in meters
        
    Returns:
        Expected TDOA values (P,)
    """
    # Source position
    s = rho * u
    
    # Distance from each microphone
    dist = np.linalg.norm(mic - s, axis=1)
    
    # Convert to time difference
    tau = dist / c
    
    return tau[pairs[:, 1]] - tau[pairs[:, 0]]


def nearfield_tdoa_batch(mic: np.ndarray, pairs: np.ndarray, 
                         c: float, u: np.ndarray, rho_values: np.ndarray) -> np.ndarray:
    """
    Vectorized TDOA computation for multiple distance values
    
    Args:
        mic: Microphone positions (M, 2)
        pairs: Microphone pair indices (P, 2)
        c: Speed of sound
        u: Unit direction vector (2,)
        rho_values: Array of distance values (N,)
        
    Returns:
        TDOA values for each distance (N, P)
    """
    # Source positions for all rho values
    s = (rho_values.reshape(-1, 1) * u.reshape(1, 2))
    
    # Distances from all microphones to all source positions
    dist = np.linalg.norm(mic[None, :, :] - s[:, None, :], axis=2)
    
    # Convert to time delays
    tau = dist / c
    
    # Compute TDOA for each pair
    return tau[:, pairs[:, 1]] - tau[:, pairs[:, 0]]


def median_abs_error_us(mic: np.ndarray, pairs: np.ndarray, 
                        c: float, u: np.ndarray, rho: float, tdoa: np.ndarray) -> float:
    """
    Compute median absolute error for given distance estimate
    
    Args:
        mic: Microphone positions (M, 2)
        pairs: Microphone pair indices (P, 2)
        c: Speed of sound
        u: Unit direction vector (2,)
        rho: Distance to evaluate
        tdoa: Measured TDOA values
        
    Returns:
        Median absolute error in microseconds
    """
    tau_pred = nearfield_tdoa(mic, pairs, c, u, rho)
    return float(np.median(np.abs(tau_pred - tdoa)) * 1e6)


def median_abs_error_batch(mic: np.ndarray, pairs: np.ndarray, 
                           c: float, u: np.ndarray, 
                           rho_values: np.ndarray, tdoa: np.ndarray) -> np.ndarray:
    """
    Vectorized error computation for multiple distance values
    
    Args:
        mic: Microphone positions (M, 2)
        pairs: Microphone pair indices (P, 2)
        c: Speed of sound
        u: Unit direction vector (2,)
        rho_values: Array of distance values (N,)
        tdoa: Measured TDOA values
        
    Returns:
        Median errors for each distance in microseconds (N,)
    """
    tdoa_pred = nearfield_tdoa_batch(mic, pairs, c, u, rho_values)
    abs_errors = np.abs(tdoa_pred - tdoa)
    return np.median(abs_errors, axis=1) * 1e6


def estimate_distance_refine(mic: np.ndarray, pairs: np.ndarray, 
                             c: float, u: np.ndarray, tdoa: np.ndarray,
                             weights: np.ndarray, rho0: float,
                             rho_lo: float, rho_hi: float) -> Tuple[float, float]:
    """
    Refine distance estimate using least squares optimization
    
    Args:
        mic: Microphone positions (M, 2)
        pairs: Microphone pair indices (P, 2)
        c: Speed of sound
        u: Unit direction vector (2,)
        tdoa: Measured TDOA values
        weights: TDOA weights
        rho0: Initial distance estimate
        rho_lo: Lower bound for search
        rho_hi: Upper bound for search
        
    Returns:
        Tuple of (refined distance, error in microseconds)
    """
    # Apply square root weighting
    w = np.sqrt(np.clip(weights / (np.median(weights) + 1e-12), 0.3, 3.0))
    
    # Normalize direction
    u = u / (np.linalg.norm(u) + 1e-12)
    
    def residual(r):
        """Residual function for optimization"""
        rho = float(r[0])
        tau_pred = nearfield_tdoa(mic, pairs, c, u, rho)
        return (tau_pred - tdoa) * w
    
    # Run optimization
    result = least_squares(
        residual,
        x0=np.array([rho0]),
        bounds=(np.array([rho_lo]), np.array([rho_hi])),
        loss='soft_l1',
        f_scale=1e-4,
        max_nfev=30
    )
    
    rho_hat = float(result.x[0])
    err_us = median_abs_error_us(mic, pairs, c, u, rho_hat, tdoa)
    
    return rho_hat, err_us


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """
    Apply moving average filter
    
    Args:
        x: Input array
        k: Window size
        
    Returns:
        Filtered array
    """
    if k <= 1:
        return x.copy()
    return np.convolve(x, np.ones(k) / k, mode='same')


# =============================================================================
# Audio Device Functions
# =============================================================================

def list_audio_devices():
    """List all available audio devices"""
    print("Available Audio Devices:")
    print(sd.query_devices())


def select_device(name_hint: str, min_channels: int) -> int:
    """
    Select an audio device based on name hint and channel requirement
    
    Args:
        name_hint: Device name substring to match
        min_channels: Minimum number of input channels required
        
    Returns:
        Device index
        
    Raises:
        RuntimeError: If no suitable device found
    """
    devices = sd.query_devices()
    hint = (name_hint or "").lower().strip()
    
    candidates = []
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", ""))
        max_inputs = int(dev.get("max_input_channels", 0))
        
        if max_inputs < min_channels:
            continue
        if hint and (hint not in name.lower()):
            continue
        
        default_sr = float(dev.get("default_samplerate", 0.0))
        candidates.append((idx, name, max_inputs, default_sr))
    
    if not candidates:
        raise RuntimeError(
            f"Cannot find device with hint='{name_hint}' and input_channels>={min_channels}."
        )
    
    # 原始逻辑：按通道数和采样率降序排列，选择第1个
    candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    idx, name, channels, sr = candidates[0]
    
    print(f"[DEVICE] Selected: index={idx}, name='{name}', "
          f"channels={channels}, sample_rate={sr}")
    
    return idx


# =============================================================================
# Main Localization Engine Class
# =============================================================================

class LocalizationEngine:
    """
    Real-time sound source localization engine
    
    This class integrates audio capture, TDOA estimation, and direction/distance
    estimation into a unified interface suitable for real-time applications.
    """
    
    def __init__(self, device_config: DeviceConfig, 
                 algo_config: Optional[AlgorithmConfig] = None):
        """
        Initialize the localization engine
        
        Args:
            device_config: Audio device configuration
            algo_config: Algorithm parameters (uses defaults if None)
        """
        self.device_config = device_config
        self.algo_config = algo_config or AlgorithmConfig()
        
        # Extract parameters
        self.c_sound = self.algo_config.c_sound
        self.rhoL = self.algo_config.rhoL
        self.rhoH = self.algo_config.rhoH
        self.angle_smooth = self.algo_config.angle_smooth
        self.rho_smooth = self.algo_config.rho_smooth
        self.cohTh = self.algo_config.coh_th
        self.vadTh = self.algo_config.vad_th
        
        # Initialize state
        self.running = False
        self.frame_count = 0
        self.start_time = 0.0
        self.realtime = 0.0
        
        # Audio stream and queue
        self.stream = None
        self.audio_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        
        # Device selection
        self.device_idx = select_device(
            device_config.device_name_hint,
            device_config.min_channels
        )
        device_info = sd.query_devices(self.device_idx, 'input')
        
        self.device_name = str(device_info.get('name', 'Unknown'))
        # Algorithm input channels are fixed to 8 (the microphone array)
        self.channels = 8
        # Stream channels: request full device channels to avoid PipeWire remap/downmix
        dev_max_ch = int(device_info.get('max_input_channels', self.channels) or self.channels)
        cfg_stream_ch = getattr(device_config, 'stream_channels', None)
        self.stream_channels = int(cfg_stream_ch) if cfg_stream_ch else dev_max_ch
        if self.stream_channels < self.channels:
            self.stream_channels = self.channels
        # Channel map: pick 8 channels from the stream (default 0..7)
        cfg_map = getattr(device_config, 'channel_map', None)
        self.channel_map = list(cfg_map) if cfg_map is not None else list(range(self.channels))
        if len(self.channel_map) != self.channels:
            raise ValueError(f"channel_map must have length {self.channels}, got {len(self.channel_map)}")
        if any((c < 0) for c in self.channel_map):
            raise ValueError("channel_map indices must be non-negative")

        self.samplerate = float(
            device_info.get('default_samplerate', 48000.0)
            if SAMPLERATE is None else SAMPLERATE
        )
        
        # Build microphone geometry
        self.diameter_m = clamp(algo_config.dia_init_meters, DIA_MIN_METERS, DIA_MAX_METERS)
        self.mic_positions, self.tau_max, self.mic_angles = self._build_geometry()
        
        # Generate pairs and window
        self.pairs = np.array(list(combinations(range(self.channels), 2)), dtype=int)
        self.blocksize = int(device_config.blocksize)
        self.window = hann_window(self.blocksize)[:, None]  # shape (N,1) for broadcasting over channels
        
        # Precompute pair differences
        self.A_pairs = pair_differences(self.mic_positions, self.pairs)
        
        # State variables
        self.noise_rms = None

        # 2D position state (matches the known-good realtime script)
        rho0 = clamp(0.30, self.rhoL, self.rhoH)
        self.pos_xy = np.array([rho0, 0.0], dtype=float)
        self.prev_rho = float(np.linalg.norm(self.pos_xy))
        # History buffers for plotting
        self.t_history: Deque[float] = deque(maxlen=MAX_HISTORY_POINTS)
        self.angle_history: Deque[float] = deque(maxlen=MAX_HISTORY_POINTS)
        self.rho_history: Deque[float] = deque(maxlen=MAX_HISTORY_POINTS)
        
        # Diameter file monitoring
        self.dia_last_poll = 0.0
        self.dia_last_value = None
        
        # Thread safety
        self.lock = Lock()
        
        # Cache for near-field TDOA computations
        self.tau_cache: Dict[Tuple[float, np.ndarray], np.ndarray] = {}
        
        print(f"[ENGINE] Initialized: {self.channels} channels @ {self.samplerate} Hz")
        print(f"[ENGINE] Array diameter: {self.diameter_m:.3f} m")
    
    def _build_geometry(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Build microphone array geometry based on current diameter
        
        Returns:
            Tuple of (positions, max_tdoa, angles)
        """
        r = self.diameter_m / 2.0
        angles = np.arange(self.channels) * (360.0 / self.channels)
        
        positions = np.column_stack((
            r * np.cos(np.deg2rad(angles)),
            r * np.sin(np.deg2rad(angles))
        ))
        
        tau_max = self.diameter_m / self.c_sound
        
        return positions, tau_max, angles
    
    def validate_diameter(self, dia_m: float) -> bool:
        """
        Validate diameter value
        
        Args:
            dia_m: Diameter in meters
            
        Returns:
            True if valid, False otherwise
        """
        return DIA_MIN_METERS < dia_m < DIA_MAX_METERS
    
    def set_diameter(self, dia_m: float):
        """
        Update array diameter and recompute geometry
        
        Args:
            dia_m: New diameter in meters
        """
        if not self.validate_diameter(dia_m):
            raise ValueError(f"Invalid diameter: {dia_m}")
        
        with self.lock:
            self.diameter_m = dia_m
            self.mic_positions, self.tau_max, self.mic_angles = self._build_geometry()
            self.A_pairs = pair_differences(self.mic_positions, self.pairs)
            
            # Clear TDOA cache
            self.tau_cache.clear()
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info, status):
        """
        Callback for audio input stream
        
        Notes:
            We request self.stream_channels from the backend, then select/reorder
            the 8 array microphones using self.channel_map. This avoids PipeWire/PortAudio
            downmix/remap that breaks TDOA.
        """
        if status:
            print(f"[AUDIO] Status: {status}", file=sys.stderr)
        
        # Ensure 2D
        if indata.ndim == 1:
            indata = indata[:, None]
        
        # Select/reorder 8 channels
        try:
            if indata.shape[1] >= max(self.channel_map) + 1:
                block8 = indata[:, self.channel_map]
            else:
                # Fallback: truncate/pad if backend delivered fewer channels than expected
                tmp = indata
                if tmp.shape[1] > self.channels:
                    tmp = tmp[:, :self.channels]
                elif tmp.shape[1] < self.channels:
                    import numpy as _np
                    tmp = _np.tile(tmp, (1, int(_np.ceil(self.channels / tmp.shape[1]))))[:, :self.channels]
                block8 = tmp
        except Exception:
            block8 = indata[:, :self.channels] if indata.shape[1] >= self.channels else indata

        try:
            self.audio_queue.put_nowait(block8.copy())
        except queue.Full:
            pass  # Drop if queue is full
    
    def start(self):
        """Start the audio stream and processing"""
        if self.running:
            return
        
        self.stream = sd.InputStream(
            device=self.device_idx,
            channels=self.stream_channels,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype='float32',
            callback=self._audio_callback,
            latency='low'
        )
        
        self.stream.start()
        self.running = True
        self.start_time = time.time()
        
        print(f"[ENGINE] Started audio stream")
    
    def stop(self):
        """Stop the audio stream"""
        if not self.running:
            return
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.running = False
        
        print(f"[ENGINE] Stopped audio stream")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
    
    def process_frame(self) -> Optional[LocalizationResult]:
        """
        Process a single audio frame
        
        Returns:
            LocalizationResult if successful, None otherwise
        """
        if not self.running:
            return None
        
        try:
            # Always get at least one block (blocking), then optionally drain to the newest.
            block = self.audio_queue.get(timeout=1.0)
            if DROP_OLD_BLOCKS:
                while True:
                    try:
                        block = self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
        except queue.Empty:
            return None
        
        self.frame_count += 1
        now = time.time()
        self.realtime = self.frame_count * self.blocksize / self.samplerate
        
        # Check for diameter file updates
        self._check_diameter_file()
        
        # Ensure correct shape
        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] != self.channels:
            if block.shape[1] > self.channels:
                block = block[:, :self.channels]
            else:
                block = np.tile(block, (1, int(np.ceil(self.channels / block.shape[1]))))[:, :self.channels]
        
        # Apply window and convert to double
        x = block.astype(np.float64, copy=False)
        xw = x * self.window
        
        # =================================================================
        # SNR estimation
        # =================================================================
        frm_rms = float(np.median(np.sqrt(np.mean(xw ** 2, axis=0))))
        frm_rms = max(frm_rms, 1e-6)
        
        if self.noise_rms is None:
            self.noise_rms = frm_rms
        self.noise_rms = max(float(self.noise_rms), NOISE_FLOOR_MIN)
        
        snr_db = 20.0 * math.log10(frm_rms / self.noise_rms)
        
        # Update noise floor
        if snr_db < self.vadTh:
            self.noise_rms = NOISE_ADAPTATION * self.noise_rms + (1.0 - NOISE_ADAPTATION) * frm_rms
            self.noise_rms = max(float(self.noise_rms), NOISE_FLOOR_MIN)
        
        # =================================================================
        # TDOA estimation using GCC-PHAT
        # =================================================================
        tdoa, weights = gcc_phat_tdoa(
            xw, self.pairs, self.blocksize, self.samplerate, self.tau_max
        )
        coherence = float(np.median(weights))
        
        # =================================================================
        # Angle + distance estimation (matches known-good LS-TDoA position solver)
        # =================================================================

        # Far-field anchor direction (used to resolve 180° ambiguity)
        u_ff, adeg_ff = estimate_direction_ls(
            self.mic_positions, self.pairs, self.c_sound, tdoa
        )

        A_mat = self.A_pairs  # (P,2)
        if float(np.sum((A_mat @ u_ff) * (self.c_sound * tdoa))) < 0.0:
            u_ff = -u_ff
            adeg_ff = wrap_angle_180(adeg_ff + 180.0)

        valid_detection = (snr_db >= self.vadTh) and (coherence >= self.cohTh)

        # Plane-model error (us)
        plane_pred = (A_mat @ u_ff) / self.c_sound
        error_plane = float(np.median(np.abs(plane_pred - tdoa)) * 1e6)

        update_flag = False
        rho_hat = float("nan")
        error_near = float("nan")

        # Initial guess along u_ff
        init_xy = u_ff * (self.prev_rho if np.isfinite(self.prev_rho) else 0.30)

        # Bounds for XY search (meters). Keep wide enough for your working range.
        bounds_xy = max(2.0, float(self.rhoH) * 2.0)

        try:
            res = estimate_pos_tdoa_ls(
                self.mic_positions,
                self.pairs,
                self.c_sound,
                tdoa,
                weights,
                init_xy,
                bounds_xy=bounds_xy,
            )
            xy_hat = np.asarray(res.x, dtype=float)
            rho_hat = float(np.linalg.norm(xy_hat))

            # 180° ambiguity resolution (KEY FIX): align LS direction with far-field anchor
            if rho_hat > 1e-6:
                u_ls = xy_hat / rho_hat
                if float(np.dot(u_ls, u_ff)) < 0.0:
                    xy_hat = -xy_hat
                    rho_hat = float(np.linalg.norm(xy_hat))

            # Optional SRP-PHAT refine along estimated direction (improves distance sensitivity)
            ENABLE_SRP_REFINE = True
            SRP_REFINE_INTERVAL = 4  # do not refine every frame to keep realtime stable
            FMIN, FMAX = 250.0, 3800.0

            if (
                ENABLE_SRP_REFINE
                and valid_detection
                and res.success
                and np.isfinite(rho_hat)
                and (self.rhoL <= rho_hat <= self.rhoH)
                and (self.frame_count % SRP_REFINE_INTERVAL == 0)
            ):
                Xi = np.fft.fft(xw, n=self.blocksize, axis=0)
                adeg_hat_tmp = math.degrees(math.atan2(xy_hat[1], xy_hat[0]))
                u_dir = np.array(
                    [math.cos(math.radians(adeg_hat_tmp)), math.sin(math.radians(adeg_hat_tmp))],
                    dtype=float,
                )

                r0 = max(self.rhoL, rho_hat - 0.15)
                r1 = min(self.rhoH, rho_hat + 0.15)
                rhos = np.linspace(r0, r1, 41)

                sc = scores_pair_1d(
                    Xi,
                    self.samplerate,
                    self.blocksize,
                    self.c_sound,
                    self.mic_positions,
                    u_dir,
                    rhos,
                    FMIN,
                    FMAX,
                    self.pairs,
                )
                rho_ref = float(rhos[int(np.argmax(sc))])
                xy_hat = u_dir * rho_ref
                rho_hat = float(np.linalg.norm(xy_hat))

                # Re-apply 180° alignment after refine
                if rho_hat > 1e-6:
                    u_ls = xy_hat / rho_hat
                    if float(np.dot(u_ls, u_ff)) < 0.0:
                        xy_hat = -xy_hat
                        rho_hat = float(np.linalg.norm(xy_hat))

            # Near-field error (us) at rho_hat (if available)
            if np.isfinite(rho_hat) and rho_hat > 1e-6:
                dist = np.linalg.norm(self.mic_positions - xy_hat.reshape(1, 2), axis=1)
                tau = dist / self.c_sound
                tauij = tau[self.pairs[:, 1]] - tau[self.pairs[:, 0]]
                error_near = float(np.median(np.abs(tauij - tdoa)) * 1e6)

            # State update (strong vs weak, same logic as known-good script)
            if valid_detection and res.success and np.isfinite(rho_hat) and (self.rhoL <= rho_hat <= self.rhoH):
                update_flag = True
                self.pos_xy = self.rho_smooth * self.pos_xy + (1.0 - self.rho_smooth) * xy_hat
            else:
                alpha_weak = 0.92
                if res.success and np.isfinite(rho_hat) and (self.rhoL * 0.5 <= rho_hat <= self.rhoH * 1.5):
                    self.pos_xy = alpha_weak * self.pos_xy + (1.0 - alpha_weak) * xy_hat

        except Exception:
            pass

        # Output from state
        distance_m = float(np.linalg.norm(self.pos_xy))
        angle_deg = wrap_angle_180(math.degrees(math.atan2(self.pos_xy[1], self.pos_xy[0])))

        # collapse guard (keep behavior consistent with known-good script)
        if distance_m < 0.02:
            distance_m = float(self.prev_rho)
            angle_deg = float(adeg_ff)
            self.pos_xy = np.array(
                [math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))], dtype=float
            ) * distance_m

        self.prev_rho = float(distance_m)
# =================================================================
        # Compute final output
        # =================================================================
        
        # Store in history
        self.t_history.append(self.realtime)
        self.angle_history.append(angle_deg)
        self.rho_history.append(distance_m)
        
        return LocalizationResult(
            timestamp=self.realtime,
            angle_deg=angle_deg,
            distance_m=distance_m,
            snr_db=snr_db,
            coherence=coherence,
            diameter_m=self.diameter_m,
            update_flag=update_flag,
            error_plane_us=error_plane,
            error_near_us=error_near,
            rho_hat=rho_hat,
            valid=valid_detection
        )
    
    def _check_diameter_file(self):
        """Check for diameter file updates"""
        now = time.time()
        
        if DIAMETER_FILE and (now - self.dia_last_poll) >= DIAMETER_FILE_POLL_SEC:
            self.dia_last_poll = now
            
            value = try_read_diameter_file(DIAMETER_FILE)
            if value is not None:
                value = clamp(float(value), DIA_MIN_METERS, DIA_MAX_METERS)
                
                if (self.dia_last_value is None or 
                    abs(value - self.dia_last_value) > 1e-6):
                    self.dia_last_value = value
                    
                    if abs(value - self.diameter_m) > 1e-9:
                        self.set_diameter(value)
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Get historical data for plotting
        
        Returns:
            Dictionary with 'time', 'angle', and 'distance' arrays
        """
        return {
            'time': np.array(self.t_history),
            'angle': np.array(self.angle_history),
            'distance': np.array(self.rho_history)
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# =============================================================================
# Validation Functions
# =============================================================================

def validate_parameters() -> bool:
    """
    Validate all configuration parameters
    
    Returns:
        True if all parameters valid
        
    Raises:
        ValueError: If any parameter is invalid
    """
    errors = []
    
    # Audio parameters
    if CHANNELS < 2:
        errors.append("CHANNELS must be at least 2")
    if SAMPLERATE is not None and SAMPLERATE <= 0:
        errors.append("SAMPLERATE must be positive")
    if DEFAULT_BLOCKSIZE < 32:
        errors.append("DEFAULT_BLOCKSIZE should be at least 32")
    
    # Diameter parameters
    if not (0 < DIA_INIT_METERS <= 10.0):
        errors.append(f"DIA_INIT_METERS should be in (0, 10.0]")
    if not (DIA_MIN_METERS < DIA_MAX_METERS):
        errors.append("DIA_MIN_METERS must be less than DIA_MAX_METERS")
    
    # Algorithm parameters
    if not (0 < C_SOUND <= 4000):
        errors.append("C_SOUND should be in (0, 4000] m/s")
    if not (0 < RHO_LOW < RHO_HIGH):
        errors.append("RHO_LOW must be positive and less than RHO_HIGH")
    
    # Smoothing parameters
    if not (0 < ANGLE_SMOOTH < 1):
        errors.append("ANGLE_SMOOTH must be in (0, 1)")
    if not (0 < RHO_SMOOTH < 1):
        errors.append("RHO_SMOOTH must be in (0, 1)")
    
    if errors:
        raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))
    
    return True


def list_devices():
    """List available audio devices"""
    list_audio_devices()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='8-Channel Microphone Array Localization'
    )
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices')
    parser.add_argument('--blocksize', type=int, default=DEFAULT_BLOCKSIZE,
                       help='Audio blocksize')
    parser.add_argument('--device', type=str, default=LOCK_DEVICE_NAME_HINT,
                       help='Device name hint')
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        validate_parameters()
        print("[OK] Configuration validated")
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    if args.list_devices:
        list_devices()
        sys.exit(0)
    
    # Create and run engine
    device_config = DeviceConfig(
        device_name_hint=args.device,
        blocksize=args.blocksize
    )
    
    with LocalizationEngine(device_config) as engine:
        print("\nRunning localization (Ctrl+C to stop)...")
        try:
            while True:
                result = engine.process_frame()
                if result is not None and engine.frame_count % 50 == 0:
                    print(f"[DATA] t={result.timestamp:5.2f}s  "
                          f"angle={result.angle_deg:7.2f}°  "
                          f"distance={result.distance_m:5.3f}m  "
                          f"SNR={result.snr_db:5.1f}dB  "
                          f"coh={result.coherence:.2f}")
        except KeyboardInterrupt:
            print("\n[STOP] Interrupted by user")