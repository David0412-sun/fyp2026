#!/usr/bin/env python3
"""
Local experiment runner for the 8-channel localization engine.

This script keeps the localization algorithm unchanged and only changes the
runtime behavior for local experiments:
1. blocksize is fixed at 1024 to match the current remote web deployment
2. every computed localization result is printed immediately
3. every result is saved in both JSONL and CSV using the same payload fields as
   the remote web program, with extra per-frame level fields
4. the reported loudness is the estimated loudness at the array geometric center
5. the received 8-channel audio stream is saved as a timestamped WAV file

Note:
    The experiment runner applies the user-provided frequency-response
    compensation together with an empirical +20 dB calibration offset from the
    sound level meter comparison so that reported loudness / SPL values are
    approximate dB SPL readings for local experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from localization import (  # noqa: E402
    AlgorithmConfig,
    CHANNELS,
    C_SOUND,
    DIA_INIT_METERS,
    DeviceConfig,
    DROP_OLD_BLOCKS,
    LOCK_DEVICE_NAME_HINT,
    LocalizationResult,
    LocalizationEngine,
    estimate_direction_ls,
    estimate_pos_tdoa_ls,
    gcc_phat_tdoa,
    list_audio_devices,
    scores_pair_1d,
    validate_parameters,
    wrap_angle_180,
)


DEFAULT_BLOCKSIZE = 1024
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiment_runs"

# Frequency-response compensation used only by the local experiment runner.
# This comes from the user's calibration notes:
# - around 100 Hz: microphone reads about 10-15 dB low -> use midpoint +12.5 dB
# - around 800-1000 Hz: about 20 dB low -> use +20 dB at 900 Hz
# - around 4000 Hz: about 30 dB low -> use +30 dB
# We keep the 20 Hz anchor at +12.5 dB and hold +30 dB above 4 kHz.
FREQ_COMPENSATION_POINTS_HZ = np.array([20.0, 100.0, 900.0, 4000.0], dtype=float)
FREQ_COMPENSATION_GAINS_DB = np.array([12.5, 12.5, 20.0, 30.0], dtype=float)
SPL_REFERENCE_PRESSURE_PA = 20e-6
SPL_CALIBRATION_OFFSET_DB = 10.0


def _safe_num(value: Any) -> Optional[float]:
    """Convert numpy scalars / NaN / Inf to JSON-safe float or None."""
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if value is None:
        return None

    try:
        numeric = float(value)
    except Exception:
        return None

    return numeric if math.isfinite(numeric) else None


def _safe_int(value: Any) -> Optional[int]:
    """Convert value to JSON-safe int or None."""
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def rms_to_spl_db(rms_value: float) -> float:
    """
    Convert an RMS pressure proxy to approximate dB SPL.

    This uses the standard 20 uPa SPL reference together with the empirical
    +20 dB offset reported from the microphone-vs-meter calibration.
    """
    rms_value = max(float(rms_value), 1e-12)
    return 20.0 * math.log10(rms_value / SPL_REFERENCE_PRESSURE_PA) + SPL_CALIBRATION_OFFSET_DB


def parse_channel_map(raw_value: Optional[str]) -> Optional[list[int]]:
    """Parse a comma-separated channel map."""
    if raw_value is None:
        return None

    text = raw_value.strip()
    if not text:
        return None

    try:
        values = [int(part.strip()) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid channel map: {raw_value!r}") from exc

    if len(values) != CHANNELS:
        raise ValueError(f"channel_map must contain exactly {CHANNELS} integers")

    return values


def _parse_optional_float(text: str) -> Optional[float]:
    """Parse text to float, returning None when parsing fails."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def sanitize_angle_tag(angle_text: str, angle_deg: Optional[float]) -> str:
    """Build a filesystem-safe angle tag for the run directory."""
    if angle_deg is not None and math.isfinite(angle_deg):
        sign = "neg_" if angle_deg < 0 else ""
        magnitude = f"{abs(angle_deg):.1f}".replace(".", "p")
        return f"angle_{sign}{magnitude}deg"

    cleaned = []
    for char in angle_text.strip():
        if char.isalnum():
            cleaned.append(char.lower())
        elif char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")

    tag = "".join(cleaned).strip("_") or "unknown"
    while "__" in tag:
        tag = tag.replace("__", "_")
    return f"angle_{tag}"


def map_angle_to_channel1_360(angle_deg: Any) -> Optional[float]:
    """
    Map the signed localization angle to [0, 360) using channel 1 as 0°.

    This preserves the exact reference used by the original localization.py:
    the first microphone position in the circular geometry is 0°, and angles
    increase counterclockwise. The only change is output convention:
    [-180, 180) -> [0, 360).
    """
    numeric = _safe_num(angle_deg)
    if numeric is None:
        return None
    return numeric % 360.0


def sanitize_diameter_tag(diameter_m: float) -> str:
    """Build a filesystem-safe diameter tag for the run directory."""
    magnitude = f"{float(diameter_m):.3f}".replace(".", "p")
    return f"dia_{magnitude}m"


def make_run_paths(
    output_dir: Path,
    diameter_tag: str,
    angle_tag: str,
) -> tuple[str, Path, Path, Path, Path, Path]:
    """Create a timestamped run directory and return output paths."""
    run_tag = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{run_tag}_{diameter_tag}_{angle_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    wav_path = run_dir / f"audio_{run_tag}.wav"
    results_path = run_dir / f"localization_{run_tag}.jsonl"
    csv_path = run_dir / f"localization_{run_tag}.csv"
    metadata_path = run_dir / f"session_{run_tag}.json"
    return run_tag, run_dir, wav_path, results_path, csv_path, metadata_path


def prompt_experiment_setup(
    default_diameter_m: float,
) -> tuple[float, str, str, Optional[float], str]:
    """
    Ask the user for the current experiment diameter and test angle.

    Returns:
        Tuple of (diameter_m, diameter_tag, angle_input, angle_deg, angle_tag)
    """
    if not sys.stdin or not sys.stdin.isatty():
        angle_input = "unknown"
        angle_deg = None
        diameter_m = float(default_diameter_m)
        return (
            diameter_m,
            sanitize_diameter_tag(diameter_m),
            angle_input,
            angle_deg,
            sanitize_angle_tag(angle_input, angle_deg),
        )

    print("[INPUT] Enter the current experiment setup before localization starts.")

    while True:
        try:
            response = input(f"Array diameter in meters [{default_diameter_m:.3f}]: ").strip()
        except EOFError:
            response = ""

        if not response:
            diameter_m = float(default_diameter_m)
            break

        parsed = _parse_optional_float(response)
        if parsed is not None and parsed > 0.0:
            diameter_m = float(parsed)
            break

        print("[INPUT] Please enter a positive number, for example 0.12")

    try:
        angle_input = input(
            "Current test angle in degrees or label (used in folder name) [unknown]: "
        ).strip()
    except EOFError:
        angle_input = ""

    if not angle_input:
        angle_input = "unknown"

    angle_deg = _parse_optional_float(angle_input)
    angle_tag = sanitize_angle_tag(angle_input, angle_deg)
    return diameter_m, sanitize_diameter_tag(diameter_m), angle_input, angle_deg, angle_tag


class ExperimentLocalizationEngine(LocalizationEngine):
    """
    Wrapper around the existing localization engine for local experiments.

    Localization math is left unchanged. This subclass only adds:
    - streaming WAV recording of the received 8-channel audio
    - a parallel metadata queue for per-frame level information
    """

    def __init__(
        self,
        device_config: DeviceConfig,
        algo_config: Optional[AlgorithmConfig],
        wav_path: Path,
    ):
        super().__init__(device_config, algo_config)
        self.wav_path = Path(wav_path)
        self._wav_file: Optional[wave.Wave_write] = None
        self._wav_lock = Lock()
        self.audio_meta_queue: queue.Queue = queue.Queue()
        self.saved_audio_blocks = 0
        self.saved_audio_frames = 0
        self.freq_compensation_points_hz = FREQ_COMPENSATION_POINTS_HZ.copy()
        self.freq_compensation_gains_db = FREQ_COMPENSATION_GAINS_DB.copy()
        self.freq_compensation_gain_db_fft = self._build_frequency_compensation_gain_db()
        self.freq_compensation_gain_linear_fft = np.power(
            10.0,
            self.freq_compensation_gain_db_fft / 20.0,
        )

    def _open_wav_file(self):
        self.wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_file = wave.open(str(self.wav_path), "wb")
        wav_file.setnchannels(self.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(round(self.samplerate)))
        self._wav_file = wav_file

    def _close_wav_file(self):
        with self._wav_lock:
            if self._wav_file is not None:
                self._wav_file.close()
                self._wav_file = None

    def _build_block8(self, indata: np.ndarray) -> np.ndarray:
        """Select and reorder the 8 array channels, matching the base engine."""
        if indata.ndim == 1:
            indata = indata[:, None]

        try:
            if indata.shape[1] >= max(self.channel_map) + 1:
                block8 = indata[:, self.channel_map]
            else:
                tmp = indata
                if tmp.shape[1] > self.channels:
                    tmp = tmp[:, :self.channels]
                elif tmp.shape[1] < self.channels:
                    tmp = np.tile(
                        tmp,
                        (1, int(np.ceil(self.channels / tmp.shape[1]))),
                    )[:, :self.channels]
                block8 = tmp
        except Exception:
            block8 = indata[:, :self.channels] if indata.shape[1] >= self.channels else indata

        return block8.copy()

    def _save_audio_block(self, block8: np.ndarray):
        """Append one 8-channel block to the WAV file as 16-bit PCM."""
        pcm = np.clip(block8, -1.0, 1.0)
        pcm_i16 = np.round(pcm * 32767.0).astype(np.int16)

        with self._wav_lock:
            if self._wav_file is None:
                return
            self._wav_file.writeframes(pcm_i16.tobytes())

        self.saved_audio_blocks += 1
        self.saved_audio_frames += int(pcm_i16.shape[0])

    def _make_audio_meta(self, block8: np.ndarray) -> Dict[str, float]:
        """
        Build per-frame audio level metadata.

        This intentionally does not touch the localization solver. It only saves
        extra experiment data for each accepted localization frame.
        """
        block64 = block8.astype(np.float64, copy=False)
        block64_comp, _ = self._apply_frequency_response_compensation(block64)
        frame_rms = float(np.median(np.sqrt(np.mean(block64_comp ** 2, axis=0))))
        frame_rms = max(frame_rms, 1e-12)

        return {
            "frame_rms": frame_rms,
            "sound_pressure_level_db": rms_to_spl_db(frame_rms),
            "peak_abs": float(np.max(np.abs(block64))),
        }

    def _build_frequency_compensation_gain_db(self) -> np.ndarray:
        """
        Build a per-bin compensation curve for the current FFT size.

        The curve is interpolated in log-frequency because the calibration
        points are sparse and span multiple octaves.
        """
        freq_hz = np.abs(np.fft.fftfreq(self.blocksize, d=1.0 / self.samplerate))
        safe_freq_hz = np.maximum(freq_hz, self.freq_compensation_points_hz[0])
        gain_db = np.interp(
            np.log10(safe_freq_hz),
            np.log10(self.freq_compensation_points_hz),
            self.freq_compensation_gains_db,
            left=float(self.freq_compensation_gains_db[0]),
            right=float(self.freq_compensation_gains_db[-1]),
        )
        gain_db[freq_hz == 0.0] = float(self.freq_compensation_gains_db[0])
        return gain_db

    def _apply_frequency_response_compensation(
        self,
        xw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply the experiment-only frequency-response compensation.

        Returns:
            Tuple of (time-domain compensated frame, compensated FFT)
        """
        Xi = np.fft.fft(xw, n=self.blocksize, axis=0)
        Xi_comp = Xi * self.freq_compensation_gain_linear_fft[:, None]
        xw_comp = np.fft.ifft(Xi_comp, n=self.blocksize, axis=0).real
        return xw_comp, Xi_comp

    def _estimate_center_signal(
        self,
        Xi: np.ndarray,
        source_xy: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the signal that would be received at the array center.

        The current source position estimate is used to phase-align all channels
        back to the geometric center, then the aligned channels are averaged.
        """
        fk = np.fft.fftfreq(self.blocksize, d=1.0 / self.samplerate).reshape(-1, 1)
        source_xy = np.asarray(source_xy, dtype=float).reshape(1, 2)

        if not np.all(np.isfinite(source_xy)):
            return np.fft.ifft(np.mean(Xi, axis=1), n=self.blocksize).real

        center_distance = float(np.linalg.norm(source_xy))
        tau_center = center_distance / self.c_sound

        mic_distance = np.linalg.norm(self.mic_positions - source_xy, axis=1)
        tau_mic = mic_distance / self.c_sound
        delta_tau = tau_mic - tau_center

        phase_to_center = np.exp(1j * 2 * math.pi * fk * delta_tau.reshape(1, -1))
        Xi_center = Xi * phase_to_center
        X_center = np.mean(Xi_center, axis=1)
        return np.fft.ifft(X_center, n=self.blocksize).real

    def _compute_center_loudness(
        self,
        Xi: np.ndarray,
        source_xy: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Compute loudness at the geometric center from the current frame.

        Returns:
            Tuple of (loudness_db, center_rms, center_peak_abs)
        """
        center_signal = self._estimate_center_signal(Xi, source_xy)
        center_rms = float(np.sqrt(np.mean(center_signal ** 2)))
        center_rms = max(center_rms, 1e-12)
        center_peak_abs = float(np.max(np.abs(center_signal)))
        loudness_db = rms_to_spl_db(center_rms)
        return loudness_db, center_rms, center_peak_abs

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """
        Callback for audio input stream.

        The localization path stays the same as the base engine. We only add
        WAV recording and per-frame level metadata collection.
        """
        if status:
            print(f"[AUDIO] Status: {status}", file=sys.stderr)

        block8 = self._build_block8(indata)
        self._save_audio_block(block8)

        try:
            self.audio_queue.put_nowait(block8)
            self.audio_meta_queue.put_nowait(self._make_audio_meta(block8))
        except queue.Full:
            pass

    def pop_latest_audio_meta(self) -> Optional[Dict[str, float]]:
        """
        Pop the newest accepted frame metadata.

        This mirrors the base engine behavior where old queued audio blocks may
        be skipped to keep the newest frame.
        """
        latest = None

        try:
            latest = self.audio_meta_queue.get_nowait()
            while True:
                try:
                    latest = self.audio_meta_queue.get_nowait()
                except queue.Empty:
                    break
        except queue.Empty:
            return None

        return latest

    def start(self):
        """Open the WAV file before the audio stream starts."""
        if self.running:
            return

        self._open_wav_file()
        try:
            super().start()
        except Exception:
            self._close_wav_file()
            raise

    def stop(self):
        """Stop audio capture and close the WAV file."""
        try:
            super().stop()
        finally:
            self._close_wav_file()

    def process_frame(self) -> Optional[LocalizationResult]:
        """
        Process one frame using the base localization pipeline plus
        experiment-only frequency-response compensation before GCC-PHAT.
        """
        if not self.running:
            return None

        try:
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
        self.realtime = self.frame_count * self.blocksize / self.samplerate

        self._check_diameter_file()

        if block.ndim == 1:
            block = block[:, None]
        if block.shape[1] != self.channels:
            if block.shape[1] > self.channels:
                block = block[:, :self.channels]
            else:
                block = np.tile(
                    block,
                    (1, int(np.ceil(self.channels / block.shape[1]))),
                )[:, :self.channels]

        x = block.astype(np.float64, copy=False)
        _, Xi_comp_raw = self._apply_frequency_response_compensation(x)
        xw_raw = x * self.window

        frm_rms = float(np.median(np.sqrt(np.mean(xw_raw ** 2, axis=0))))
        frm_rms = max(frm_rms, 1e-9)
        snr_db = 20.0 * math.log10(frm_rms)

        xw_comp, Xi_comp = self._apply_frequency_response_compensation(xw_raw)

        tdoa, weights = gcc_phat_tdoa(
            xw_comp,
            self.pairs,
            self.blocksize,
            self.samplerate,
            self.tau_max,
        )
        coherence = float(np.median(weights))

        u_ff, adeg_ff = estimate_direction_ls(
            self.mic_positions,
            self.pairs,
            self.c_sound,
            tdoa,
        )

        A_mat = self.A_pairs
        if float(np.sum((A_mat @ u_ff) * (self.c_sound * tdoa))) < 0.0:
            u_ff = -u_ff
            adeg_ff = wrap_angle_180(adeg_ff + 180.0)

        valid_detection = coherence >= self.cohTh

        plane_pred = (A_mat @ u_ff) / self.c_sound
        error_plane = float(np.median(np.abs(plane_pred - tdoa)) * 1e6)

        update_flag = False
        rho_hat = float("nan")
        error_near = float("nan")

        init_xy = u_ff * (self.prev_rho if np.isfinite(self.prev_rho) else 0.30)
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

            if rho_hat > 1e-6:
                u_ls = xy_hat / rho_hat
                if float(np.dot(u_ls, u_ff)) < 0.0:
                    xy_hat = -xy_hat
                    rho_hat = float(np.linalg.norm(xy_hat))

            enable_srp_refine = True
            srp_refine_interval = 4
            fmin, fmax = 0.0, min(10000.0, 0.5 * self.samplerate - 1.0)

            if (
                enable_srp_refine
                and valid_detection
                and res.success
                and np.isfinite(rho_hat)
                and (self.rhoL <= rho_hat <= self.rhoH)
                and (self.frame_count % srp_refine_interval == 0)
            ):
                adeg_hat_tmp = math.degrees(math.atan2(xy_hat[1], xy_hat[0]))
                u_dir = np.array(
                    [
                        math.cos(math.radians(adeg_hat_tmp)),
                        math.sin(math.radians(adeg_hat_tmp)),
                    ],
                    dtype=float,
                )

                r0 = max(self.rhoL, rho_hat - 0.15)
                r1 = min(self.rhoH, rho_hat + 0.15)
                rhos = np.linspace(r0, r1, 41)

                sc = scores_pair_1d(
                    Xi_comp,
                    self.samplerate,
                    self.blocksize,
                    self.c_sound,
                    self.mic_positions,
                    u_dir,
                    rhos,
                    fmin,
                    fmax,
                    self.pairs,
                )
                rho_ref = float(rhos[int(np.argmax(sc))])
                xy_hat = u_dir * rho_ref
                rho_hat = float(np.linalg.norm(xy_hat))

                if rho_hat > 1e-6:
                    u_ls = xy_hat / rho_hat
                    if float(np.dot(u_ls, u_ff)) < 0.0:
                        xy_hat = -xy_hat
                        rho_hat = float(np.linalg.norm(xy_hat))

            if np.isfinite(rho_hat) and rho_hat > 1e-6:
                dist = np.linalg.norm(
                    self.mic_positions - xy_hat.reshape(1, 2),
                    axis=1,
                )
                tau = dist / self.c_sound
                tauij = tau[self.pairs[:, 1]] - tau[self.pairs[:, 0]]
                error_near = float(np.median(np.abs(tauij - tdoa)) * 1e6)

            if (
                valid_detection
                and res.success
                and np.isfinite(rho_hat)
                and (self.rhoL <= rho_hat <= self.rhoH)
            ):
                update_flag = True
                self.pos_xy = self.rho_smooth * self.pos_xy + (1.0 - self.rho_smooth) * xy_hat
            else:
                alpha_weak = 0.92
                if (
                    res.success
                    and np.isfinite(rho_hat)
                    and (self.rhoL * 0.5 <= rho_hat <= self.rhoH * 1.5)
                ):
                    self.pos_xy = alpha_weak * self.pos_xy + (1.0 - alpha_weak) * xy_hat

        except Exception:
            pass

        distance_m = float(np.linalg.norm(self.pos_xy))
        angle_deg = wrap_angle_180(
            math.degrees(math.atan2(self.pos_xy[1], self.pos_xy[0]))
        )

        if distance_m < 0.02:
            distance_m = float(self.prev_rho)
            angle_deg = float(adeg_ff)
            self.pos_xy = np.array(
                [
                    math.cos(math.radians(angle_deg)),
                    math.sin(math.radians(angle_deg)),
                ],
                dtype=float,
            ) * distance_m

        self.prev_rho = float(distance_m)
        self.t_history.append(self.realtime)
        self.angle_history.append(angle_deg)
        self.rho_history.append(distance_m)

        loudness_db, loudness_rms, loudness_peak_abs = self._compute_center_loudness(
            Xi_comp_raw,
            self.pos_xy.copy(),
        )

        result = LocalizationResult(
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
            valid=valid_detection,
        )
        result.loudness_db = loudness_db
        result.loudness_rms = loudness_rms
        result.loudness_peak_abs = loudness_peak_abs
        return result


def build_result_payload(
    result: Any,
    audio_meta: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """Build one saved record using the remote web payload plus extra fields."""
    now_wall = time.time()
    wall_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")
    loudness_db = _safe_num(getattr(result, "loudness_db", None))
    if loudness_db is None:
        loudness_db = _safe_num(getattr(result, "snr_db", None))
    mapped_angle_deg = map_angle_to_channel1_360(getattr(result, "angle_deg", None))

    payload: Dict[str, Any] = {
        "type": "data",
        "timestamp": _safe_num(getattr(result, "timestamp", None)),
        "wall_ts": _safe_num(now_wall),
        "wall_time_iso": wall_iso,
        "angle": mapped_angle_deg,
        "distance": _safe_num(getattr(result, "distance_m", None)),
        "snr_db": _safe_num(getattr(result, "snr_db", None)),
        "coherence": _safe_num(getattr(result, "coherence", None)),
        "diameter_m": _safe_num(getattr(result, "diameter_m", None)),
        "update_flag": _safe_int(getattr(result, "update_flag", None)),
        "error_plane_us": _safe_num(getattr(result, "error_plane_us", None)),
        "error_near_us": _safe_num(getattr(result, "error_near_us", None)),
        "rho_hat": _safe_num(getattr(result, "rho_hat", None)),
        "valid": bool(getattr(result, "valid", False)),
        "loudness_db": loudness_db,
        "loudness_rms": _safe_num(getattr(result, "loudness_rms", None)),
        "loudness_peak_abs": _safe_num(getattr(result, "loudness_peak_abs", None)),
        "sound_pressure_level_db": None,
        "frame_rms": None,
        "peak_abs": None,
    }

    if audio_meta is not None:
        payload["sound_pressure_level_db"] = _safe_num(audio_meta.get("sound_pressure_level_db"))
        payload["frame_rms"] = _safe_num(audio_meta.get("frame_rms"))
        payload["peak_abs"] = _safe_num(audio_meta.get("peak_abs"))

    return payload


def print_payload(payload: Dict[str, Any]):
    """Print every localization result immediately."""
    angle = payload.get("angle")
    distance = payload.get("distance")
    snr_db = payload.get("snr_db")
    loudness_db = payload.get("loudness_db")
    spl_db = payload.get("sound_pressure_level_db")
    coh = payload.get("coherence")
    valid = payload.get("valid")
    timestamp = payload.get("timestamp")

    angle_text = f"{angle:7.2f}deg" if isinstance(angle, (int, float)) else "   n/a "
    dist_text = f"{distance:6.3f}m" if isinstance(distance, (int, float)) else "  n/a "
    snr_text = f"{snr_db:6.1f}dB" if isinstance(snr_db, (int, float)) else "  n/a "
    loud_text = f"{loudness_db:6.1f}dB" if isinstance(loudness_db, (int, float)) else "  n/a "
    spl_text = f"{spl_db:6.1f}dB" if isinstance(spl_db, (int, float)) else "  n/a "
    coh_text = f"{coh:4.2f}" if isinstance(coh, (int, float)) else "n/a "
    time_text = f"{timestamp:7.3f}s" if isinstance(timestamp, (int, float)) else "   n/a "

    print(
        f"[DATA] {payload['wall_time_iso']}  "
        f"t={time_text}  "
        f"angle={angle_text}  "
        f"distance={dist_text}  "
        f"snr={snr_text}  "
        f"loud={loud_text}  "
        f"spl={spl_text}  "
        f"coh={coh_text}  "
        f"valid={valid}"
    )


def write_session_metadata(path: Path, metadata: Dict[str, Any]):
    """Write run metadata as JSON."""
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the localization engine locally and save every result.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help=(
            "Audio device name hint. Leave empty to auto-pick an input device "
            f"with at least {CHANNELS} channels."
        ),
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=DIA_INIT_METERS,
        help="Array diameter in meters.",
    )
    parser.add_argument(
        "--c-sound",
        type=float,
        default=C_SOUND,
        dest="c_sound",
        help="Speed of sound in m/s.",
    )
    parser.add_argument(
        "--stream-channels",
        type=int,
        default=None,
        help="Channels requested from the backend before applying channel_map.",
    )
    parser.add_argument(
        "--channel-map",
        type=str,
        default=None,
        help="Comma-separated 8-channel map, for example: 0,1,2,3,4,5,6,7",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory used to store experiment outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return 0

    try:
        validate_parameters()
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        channel_map = parse_channel_map(args.channel_map)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    run_diameter_m, diameter_tag, test_angle_input, test_angle_deg, angle_tag = prompt_experiment_setup(
        args.diameter
    )

    output_dir = args.output_dir.expanduser().resolve()
    run_tag, run_dir, wav_path, results_path, csv_path, metadata_path = make_run_paths(
        output_dir,
        diameter_tag,
        angle_tag,
    )
    started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")

    device_config = DeviceConfig(
        device_name_hint=args.device,
        min_channels=CHANNELS,
        blocksize=DEFAULT_BLOCKSIZE,
        stream_channels=args.stream_channels,
        channel_map=channel_map,
    )
    algo_config = AlgorithmConfig(
        c_sound=args.c_sound,
        dia_init_meters=run_diameter_m,
    )

    result_count = 0
    engine: Optional[ExperimentLocalizationEngine] = None

    try:
        engine = ExperimentLocalizationEngine(
            device_config=device_config,
            algo_config=algo_config,
            wav_path=wav_path,
        )

        initial_metadata = {
            "run_tag": run_tag,
            "started_at": started_at,
            "blocksize": DEFAULT_BLOCKSIZE,
            "device_name_hint": args.device or None,
            "device_name_default_hint": LOCK_DEVICE_NAME_HINT,
            "stream_channels": args.stream_channels,
            "channel_map": channel_map,
            "diameter_m": run_diameter_m,
            "diameter_tag": diameter_tag,
            "c_sound": args.c_sound,
            "test_angle_input": test_angle_input,
            "test_angle_deg": test_angle_deg,
            "angle_tag": angle_tag,
            "angle_output_convention": "channel_1_as_0deg_counterclockwise_0_to_360",
            "run_directory": str(run_dir),
            "results_file": str(results_path),
            "csv_file": str(csv_path),
            "audio_file": str(wav_path),
            "device_name": engine.device_name,
            "samplerate": engine.samplerate,
            "channels": engine.channels,
            "frequency_response_compensation": {
                "enabled": True,
                "points_hz": engine.freq_compensation_points_hz.tolist(),
                "gain_db": engine.freq_compensation_gains_db.tolist(),
            },
            "spl_calibration": {
                "reference_pressure_pa": SPL_REFERENCE_PRESSURE_PA,
                "offset_db": SPL_CALIBRATION_OFFSET_DB,
                "method": "frequency_compensated_rms_plus_empirical_meter_offset",
            },
            "status": "running",
        }
        write_session_metadata(metadata_path, initial_metadata)

        print("[OK] Configuration validated")
        print(f"[RUN] blocksize fixed at {DEFAULT_BLOCKSIZE}")
        print(f"[RUN] device={engine.device_name}")
        print(f"[RUN] samplerate={engine.samplerate}")
        print(f"[RUN] input diameter={run_diameter_m:.3f} m")
        print(f"[RUN] engine diameter={engine.diameter_m:.3f} m")
        print(f"[RUN] test angle={test_angle_input}")
        print(
            "[RUN] freq compensation="
            + ", ".join(
                f"{freq:.0f}Hz:+{gain:.1f}dB"
                for freq, gain in zip(
                    engine.freq_compensation_points_hz,
                    engine.freq_compensation_gains_db,
                )
            )
        )
        print(f"[RUN] folder  -> {run_dir}")
        print(f"[RUN] results -> {results_path}")
        print(f"[RUN] csv     -> {csv_path}")
        print(f"[RUN] audio   -> {wav_path}")
        print("[RUN] Running localization (Ctrl+C to stop)...")

        with results_path.open("w", encoding="utf-8", buffering=1) as results_file:
            with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                csv_writer: Optional[csv.DictWriter] = None
                with engine:
                    while True:
                        result = engine.process_frame()
                        if result is None:
                            continue

                        audio_meta = engine.pop_latest_audio_meta()
                        payload = build_result_payload(result, audio_meta)
                        results_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(csv_file, fieldnames=list(payload.keys()))
                            csv_writer.writeheader()
                        csv_writer.writerow(payload)
                        print_payload(payload)
                        result_count += 1

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    finally:
        ended_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
        metadata = {
            "run_tag": run_tag,
            "started_at": started_at,
            "ended_at": ended_at,
            "blocksize": DEFAULT_BLOCKSIZE,
            "device_name_hint": args.device or None,
            "device_name_default_hint": LOCK_DEVICE_NAME_HINT,
            "stream_channels": args.stream_channels,
            "channel_map": channel_map,
            "diameter_m": run_diameter_m,
            "diameter_tag": diameter_tag,
            "c_sound": args.c_sound,
            "test_angle_input": test_angle_input,
            "test_angle_deg": test_angle_deg,
            "angle_tag": angle_tag,
            "angle_output_convention": "channel_1_as_0deg_counterclockwise_0_to_360",
            "run_directory": str(run_dir),
            "results_file": str(results_path),
            "csv_file": str(csv_path),
            "audio_file": str(wav_path),
            "device_name": getattr(engine, "device_name", None),
            "samplerate": getattr(engine, "samplerate", None),
            "channels": getattr(engine, "channels", None),
            "frequency_response_compensation": {
                "enabled": True,
                "points_hz": getattr(engine, "freq_compensation_points_hz", np.array([])).tolist(),
                "gain_db": getattr(engine, "freq_compensation_gains_db", np.array([])).tolist(),
            },
            "spl_calibration": {
                "reference_pressure_pa": SPL_REFERENCE_PRESSURE_PA,
                "offset_db": SPL_CALIBRATION_OFFSET_DB,
                "method": "frequency_compensated_rms_plus_empirical_meter_offset",
            },
            "saved_result_count": result_count,
            "saved_audio_blocks": getattr(engine, "saved_audio_blocks", 0),
            "saved_audio_frames": getattr(engine, "saved_audio_frames", 0),
            "status": "stopped",
        }
        write_session_metadata(metadata_path, metadata)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
