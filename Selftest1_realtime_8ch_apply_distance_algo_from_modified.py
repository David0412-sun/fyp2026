#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime 8-ch Microphone Array Localization (Py3.9)

This realtime version *directly applies the same distance algorithm* used in
Selftest1_compact_modified.py:
- Distance via near-field multilateration (LS-TDoA).
- 180° ambiguity resolved by aligning LS direction with far-field GCC direction (u_ff).
- Optional SRP-PHAT distance refine around rho_hat along the estimated direction.

Extra realtime engineering fixes:
- Robust device selection (by index or by name hint, with 8 input channels).
- Always opens InputStream with channels=8 (no accidental 2ch).
- TkAgg-safe plotting (frozen snapshots + length guard).
- Optional per-channel RMS(dBFS) monitor.

Run (CLI):
  python Selftest1_realtime_8ch_apply_distance_algo_from_modified.py --list-devices
  python Selftest1_realtime_8ch_apply_distance_algo_from_modified.py --device 78 --samplerate 48000 --blocksize 512

Run (PyCharm "Run" button):
  Edit the USER CONFIG section below (DEVICE_INDEX / NAME_HINT / etc.), then Run.
"""

import sys
import math
import argparse
import queue
import time
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sounddevice as sd
from itertools import combinations
from scipy.optimize import least_squares


# =========================
# USER CONFIG (PyCharm-friendly)
# =========================
DEVICE_INDEX: Optional[int] = None      # set to 78 if you want to lock to that index; None -> auto-pick
DEVICE_NAME_HINT: str = "YDM8MIC"       # substring match in device name (case-insensitive); "" disables
CHANNELS: int = 8                       # MUST be 8 for your array
SAMPLERATE: Optional[float] = None      # None -> use device default_samplerate (more robust)
BLOCKSIZE: int = 512                    # should match your known-good offline N=512
DIA_METERS: float = 0.22                # IMPORTANT: set to your real array diameter (meters)
MIC_CHANNEL_MAP = list(range(8))        # device channel -> mic order around the circle (edit if needed)

# Optional input monitor
PRINT_INPUT_EVERY_SEC: float = 1.0      # 0 to disable


# =========================
# Algorithm utils (copied/kept consistent with Selftest1_compact_modified.py)
# =========================
def hann(N: int) -> np.ndarray:
    n = np.arange(N, dtype=float)
    return 0.5 * (1 - np.cos(2 * np.pi * n / N))


def wrap180(a_deg: float) -> float:
    return (a_deg + 180.0) % 360.0 - 180.0


def parab(y: np.ndarray, i: int) -> float:
    N = y.size
    iL = (i - 1) % N
    i0 = i % N
    iR = (i + 1) % N
    denom = (y[iL] - 2 * y[i0] + y[iR] + np.finfo(float).eps)
    return (y[iL] - y[iR]) / (2 * denom)


def pairA(mic: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    P = pairs.shape[0]
    A = np.zeros((P, 2), dtype=float)
    for p in range(P):
        i, j = pairs[p]
        A[p, :] = mic[j, :] - mic[i, :]
    return A


def dir_est_ls_farfield(mic: np.ndarray, pairs: np.ndarray, c: float, tdoa: np.ndarray):
    A = pairA(mic, pairs)
    b = c * tdoa.reshape(-1, 1)
    u_est, *_ = np.linalg.lstsq(A, b, rcond=None)
    u = u_est.flatten()
    u /= (np.linalg.norm(u) + 1e-12)
    adeg = math.degrees(math.atan2(u[1], u[0]))
    return u, adeg


def gcc_phat_tdoa(xw: np.ndarray, pairs: np.ndarray, N: int, fs: float, tauMax: float):
    P = pairs.shape[0]
    tdoa = np.zeros(P, dtype=float)
    w = np.ones(P, dtype=float)

    for p in range(P):
        i, j = pairs[p]
        Xi = np.fft.fft(xw[:, i], n=N)
        Xj = np.fft.fft(xw[:, j], n=N)

        R = Xi * np.conj(Xj)
        R /= (np.abs(R) + np.finfo(float).eps)
        cc = np.fft.ifft(R, n=N).real
        cc = np.fft.fftshift(cc)

        idx = int(np.argmax(np.abs(cc)))
        d = parab(cc, idx)
        lag = (idx - (N // 2)) + d

        pk = float(np.max(np.abs(cc)))
        med = float(np.median(np.abs(cc)) + np.finfo(float).eps)
        w[p] = max(0.2, min(8.0, pk / med))

        t = lag / fs
        tdoa[p] = max(-tauMax, min(t, tauMax))

    return tdoa, w


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
        s = rho * u_row
        D = mic - s
        dist = np.linalg.norm(D, axis=1)
        tau = dist / c
        tauij = tau[pairs_full[:, 1]] - tau[pairs_full[:, 0]]
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


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    return np.convolve(x, np.ones(k) / k, mode="same")


def rms_dbfs(x: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2, axis=0) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    return db


# =========================
# Device selection (realtime robustness)
# =========================
def list_devices():
    print(sd.query_devices())


def pick_device(index: Optional[int], name_hint: str, min_in_ch: int) -> Tuple[int, dict]:
    devs = sd.query_devices()
    if index is not None:
        if index < 0 or index >= len(devs):
            raise ValueError(f"--device {index} out of range 0..{len(devs)-1}")
        info = devs[index]
        if int(info.get("max_input_channels", 0)) < min_in_ch:
            raise ValueError(f"Device {index} has only {info.get('max_input_channels',0)} input ch, need {min_in_ch}.")
        return int(index), info

    hint = (name_hint or "").strip().lower()
    if hint:
        for i, d in enumerate(devs):
            if int(d.get("max_input_channels", 0)) >= min_in_ch and hint in str(d.get("name", "")).lower():
                return int(i), d

    # fallback: first device with enough channels
    for i, d in enumerate(devs):
        if int(d.get("max_input_channels", 0)) >= min_in_ch:
            return int(i), d

    raise RuntimeError(f"No input device found with >= {min_in_ch} channels.")


# =========================
# Realtime runner
# =========================
def run_realtime(
    device_index: Optional[int],
    name_hint: str,
    samplerate: Optional[float],
    blocksize: int,
    diameter: float,
    ch_map: list,
    print_input_every: float,
):
    N = int(blocksize)
    c = 343.0
    Dia = float(diameter)
    r = Dia / 2.0
    M = 8

    # Parameters (same as modified/offline)
    rhoL, rhoH = 0.05, 1.50
    bounds_xy = 2.0

    Ksmooth = 5
    pos_smooth = 0.75

    cohTh = 1.15
    vadTh = 3.0
    noiseA = 0.985

    enable_srp_refine = True
    fmin, fmax = 250.0, 3800.0

    # mic geometry
    ang = np.arange(M) * 45.0
    mic = np.column_stack((r * np.cos(np.deg2rad(ang)), r * np.sin(np.deg2rad(ang))))
    pairs = np.array(list(combinations(range(M), 2)), dtype=int)

    # stream
    dev_i, dev_info = pick_device(device_index, name_hint, min_in_ch=8)
    fs = float(dev_info.get("default_samplerate", 48000.0)) if samplerate is None else float(samplerate)

    # Attempt stream open; if fails, retry with device default_samplerate
    def open_stream(sr: float) -> sd.InputStream:
        return sd.InputStream(
            device=dev_i,
            channels=8,              # ALWAYS 8
            samplerate=sr,
            blocksize=N,
            dtype="float32",
            callback=audio_callback,
        )

    # plot
    plt.close("all")
    fig = plt.figure(num="Angle + Distance (Realtime 8ch Array | LS-TDoA)", figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1], wspace=0.25, hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Angle vs Time")
    ax1.set_ylim(-180, 180)
    ax1.grid()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("Distance vs Time")
    ax2.set_ylim(0, max(1.0, rhoH))
    ax2.grid()

    axp = fig.add_subplot(gs[:, 1], projection="polar")
    axp.set_title("Angle (needle)")
    axp.set_theta_zero_location("E")
    axp.set_theta_direction(1)
    axp.set_thetagrids(np.arange(0, 360, 30))
    axp.set_rlim(0, r * 1.2)

    h_ang_raw, = ax1.plot([], [], linestyle="--")
    h_ang_s,   = ax1.plot([], [], linestyle="-")
    h_rho_raw, = ax2.plot([], [], linestyle="--")
    h_rho_s,   = ax2.plot([], [], linestyle="-")

    for m in range(M):
        axp.plot([math.radians(ang[m])], [r], marker="^")

    h_needle, = axp.plot([], [], "-")
    h_tip,    = axp.plot([], [], "o")

    status_text = fig.text(0.01, 0.01, "", fontsize=9)
    t_window = 30.0

    # state
    tauMax = Dia / c
    noiseRMS = None

    treal = 0.0
    t_list, Adeg, Rho = [], [], []

    pos_xy = np.array([0.30, 0.0], dtype=float)
    prev_rho = float(np.linalg.norm(pos_xy))

    win = hann(N).reshape(-1, 1)

    # audio queue
    q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=80)

    # input monitor
    last_print_t = 0.0
    last_db = None
    last_pk = None

    # validate channel map
    if len(ch_map) != 8:
        raise ValueError(f"MIC_CHANNEL_MAP must have 8 entries, got {len(ch_map)}")

    def audio_callback(indata, frames, time_info, status):
        nonlocal last_db, last_pk
        if status:
            print(status, file=sys.stderr)
        try:
            x = indata.copy()  # float32
            # monitor snapshot
            last_pk = np.max(np.abs(x), axis=0)
            last_db = rms_dbfs(x)
            q.put_nowait(x)
        except queue.Full:
            try:
                _ = q.get_nowait()
                q.put_nowait(indata.copy())
            except queue.Empty:
                pass

    # open stream with fallback
    try:
        stream = sd.InputStream(
            device=dev_i,
            channels=8,
            samplerate=fs,
            blocksize=N,
            dtype="float32",
            callback=audio_callback,
        )
    except Exception as e:
        fs2 = float(dev_info.get("default_samplerate", fs))
        print(f"[WARN] Open stream failed at fs={fs}. Retrying at default fs={fs2}. Reason: {e}")
        fs = fs2
        stream = sd.InputStream(
            device=dev_i,
            channels=8,
            samplerate=fs,
            blocksize=N,
            dtype="float32",
            callback=audio_callback,
        )

    print(f"[Realtime] device={dev_i} name='{dev_info.get('name','')}'")
    print(f"[Realtime] fs={fs}Hz, blocksize={N}, channels=8")
    print(f"[Geometry] DIA_METERS={Dia}  MIC_CHANNEL_MAP={ch_map}")
    print("Clap/speak to test. Close the plot window to stop.\n")

    with stream:
        while plt.fignum_exists(fig.number):
            try:
                block = q.get(timeout=1.0)
            except queue.Empty:
                continue

            x_in = block
            if x_in.ndim == 1:
                x_in = x_in.reshape(-1, 1)

            # ensure exactly N frames
            if x_in.shape[0] < N:
                pad = np.zeros((N - x_in.shape[0], x_in.shape[1]), dtype=x_in.dtype)
                x_in = np.vstack((x_in, pad))
            elif x_in.shape[0] > N:
                x_in = x_in[:N, :]

            # ensure 8 channels then map to mic order
            if x_in.shape[1] < 8:
                x_in = np.pad(x_in, ((0, 0), (0, 8 - x_in.shape[1])), mode="constant")
            elif x_in.shape[1] > 8:
                x_in = x_in[:, :8]
            x_in = x_in[:, ch_map]

            # periodic input print (proof of streaming)
            treal += N / fs
            if print_input_every > 0 and (treal - last_print_t) >= print_input_every and last_db is not None and last_pk is not None:
                last_print_t = treal
                db = np.asarray(last_db).flatten()
                pk = np.asarray(last_pk).flatten()
                s_db = " ".join([f"ch{i}:{db[i]:6.1f}dB" for i in range(8)])
                s_pk = " ".join([f"ch{i}:{pk[i]:.4f}" for i in range(8)])
                print(f"[IN] dBFS  {s_db}")
                print(f"[IN] peak  {s_pk}\n")

            # ===== Apply the SAME distance algorithm as the uploaded modified version =====
            x = x_in.astype(np.float64, copy=False)
            xw = x * win

            frmRMS = float(np.median(np.sqrt(np.mean(xw ** 2, axis=0))))
            if noiseRMS is None:
                noiseRMS = frmRMS + 1e-12

            snrDb = 20 * math.log10((frmRMS + 1e-12) / (noiseRMS + 1e-12))
            if snrDb < vadTh:
                noiseRMS = noiseA * noiseRMS + (1 - noiseA) * frmRMS

            tdoa, w = gcc_phat_tdoa(xw, pairs, N, fs, tauMax)
            coh = float(np.median(w))

            # Far-field anchor direction (for 180° disambiguation)
            u_ff, adeg_ff = dir_est_ls_farfield(mic, pairs, c, tdoa)

            A_mat = pairA(mic, pairs)
            if float(np.sum((A_mat @ u_ff) * (c * tdoa))) < 0:
                u_ff = -u_ff
                adeg_ff = wrap180(adeg_ff + 180.0)

            good = (snrDb >= vadTh) and (coh >= cohTh)

            init_xy = u_ff * (prev_rho if np.isfinite(prev_rho) else 0.30)

            try:
                res = estimate_pos_tdoa_ls(mic, pairs, c, tdoa, w, init_xy, bounds_xy=bounds_xy)
                xy_hat = res.x
                rho_hat = float(np.linalg.norm(xy_hat))

                # 180° ambiguity resolution (KEY FIX)
                u_ls = xy_hat / (rho_hat + 1e-12)
                if float(np.dot(u_ls, u_ff)) < 0.0:
                    xy_hat = -xy_hat
                    rho_hat = float(np.linalg.norm(xy_hat))
                adeg_hat = wrap180(math.degrees(math.atan2(xy_hat[1], xy_hat[0])))

                # Optional SRP distance refine around rho_hat
                if enable_srp_refine and (rhoL <= rho_hat <= rhoH):
                    Xi = np.fft.fft(xw, n=N, axis=0)
                    u_dir = np.array([math.cos(math.radians(adeg_hat)), math.sin(math.radians(adeg_hat))], dtype=float)

                    r0 = max(rhoL, rho_hat - 0.15)
                    r1 = min(rhoH, rho_hat + 0.15)
                    rhos = np.linspace(r0, r1, 41)

                    sc = scores_pair_1d(Xi, fs, N, c, mic, u_dir, rhos, fmin, fmax, pairs)
                    rho_ref = float(rhos[int(np.argmax(sc))])

                    xy_hat = u_dir * rho_ref
                    rho_hat = float(np.linalg.norm(xy_hat))

                    # re-apply 180° alignment after refine
                    u_ls = xy_hat / (rho_hat + 1e-12)
                    if float(np.dot(u_ls, u_ff)) < 0.0:
                        xy_hat = -xy_hat
                        rho_hat = float(np.linalg.norm(xy_hat))
                    adeg_hat = wrap180(math.degrees(math.atan2(xy_hat[1], xy_hat[0])))

                # state update (same as modified)
                if good and res.success and (rhoL <= rho_hat <= rhoH):
                    pos_xy = pos_smooth * pos_xy + (1 - pos_smooth) * xy_hat
                else:
                    alpha_weak = 0.92
                    if res.success and np.isfinite(rho_hat) and (rhoL * 0.5 <= rho_hat <= rhoH * 1.5):
                        pos_xy = alpha_weak * pos_xy + (1 - alpha_weak) * xy_hat

            except Exception:
                pass

            rho_out = float(np.linalg.norm(pos_xy))
            adeg_out = wrap180(math.degrees(math.atan2(pos_xy[1], pos_xy[0])))

            # collapse guard (same behavior)
            if rho_out < 0.02:
                rho_out = prev_rho
                adeg_out = float(adeg_ff)
                pos_xy = np.array([math.cos(math.radians(adeg_out)), math.sin(math.radians(adeg_out))]) * rho_out

            prev_rho = rho_out

            # history
            t_list.append(treal)
            Adeg.append(float(adeg_out))
            Rho.append(float(rho_out))

            A1 = moving_average(np.array(Adeg, dtype=float), Ksmooth)
            R1 = moving_average(np.array(Rho, dtype=float), Ksmooth)

            # ===== Plot update (TkAgg-safe) =====
            h_needle.set_data([math.radians(adeg_out), math.radians(adeg_out)], [0, r])
            h_tip.set_data([math.radians(adeg_out)], [r])

            nt = len(t_list)
            na = len(Adeg)
            nr = len(Rho)
            nA1 = int(np.size(A1))
            nR1 = int(np.size(R1))
            nmin = min(nt, na, nr, nA1, nR1)
            if nmin >= 2:
                tt = np.asarray(t_list[:nmin], dtype=float).copy()
                aa = np.asarray(Adeg[:nmin], dtype=float).copy()
                rr = np.asarray(Rho[:nmin], dtype=float).copy()
                aa1 = np.asarray(A1[:nmin], dtype=float).copy()
                rr1 = np.asarray(R1[:nmin], dtype=float).copy()

                h_ang_raw.set_data(tt, aa)
                h_ang_s.set_data(tt, aa1)
                h_rho_raw.set_data(tt, rr)
                h_rho_s.set_data(tt, rr1)

            tmin = max(0.0, treal - t_window)
            tmax = max(t_window, treal)
            ax1.set_xlim(tmin, tmax)
            ax2.set_xlim(tmin, tmax)
            ax2.set_ylim(0, max(1.0, min(rhoH, float(np.nanmax(R1) + 0.2))))

            status_text.set_text(
                f"dev={dev_i}  fs={fs:.0f}Hz  SNR={snrDb:.1f}dB  coh={coh:.2f}  "
                f"angle={adeg_out:+.1f}°  rho={rho_out:.3f}m"
            )

            plt.pause(0.001)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--device", type=int, default=None, help="Input device index for sounddevice.")
    parser.add_argument("--name-hint", type=str, default=None, help="Substring to match device name.")
    parser.add_argument("--samplerate", type=float, default=None, help="Sampling rate (Hz). None -> device default.")
    parser.add_argument("--blocksize", type=int, default=None, help="Block size per frame (default uses BLOCKSIZE).")
    parser.add_argument("--diameter", type=float, default=None, help="Array diameter in meters (default uses DIA_METERS).")
    parser.add_argument("--no-input-print", action="store_true", help="Disable periodic input dBFS/peak printing.")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    dev = DEVICE_INDEX if args.device is None else args.device
    hint = DEVICE_NAME_HINT if args.name_hint is None else args.name_hint
    sr = SAMPLERATE if args.samplerate is None else args.samplerate
    bs = BLOCKSIZE if args.blocksize is None else args.blocksize
    dia = DIA_METERS if args.diameter is None else args.diameter
    pr = 0.0 if args.no_input_print else PRINT_INPUT_EVERY_SEC

    run_realtime(
        device_index=dev,
        name_hint=hint,
        samplerate=sr,
        blocksize=int(bs),
        diameter=float(dia),
        ch_map=MIC_CHANNEL_MAP,
        print_input_every=float(pr),
    )


if __name__ == "__main__":
    main()
