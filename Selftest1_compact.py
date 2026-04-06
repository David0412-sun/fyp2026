#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python port of Selftest1_compact.m

Features:
- GCC-PHAT TDoA per mic pair
- Least-squares DoA (2D) and 180° disambiguation
- SRP-PHAT distance grid search around estimated angle ±6° at 2° steps
- VAD by SNR + pairwise coherence gate
- EMA smoothing for noise RMS and SRP accumulator
- Live plots: angle/time, distance/time, and polar angle needle

Dependencies: numpy, scipy, matplotlib, (soundfile optional)
Run: python Selftest1_compact.py [path/to/test0929.wav]
"""

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Try to import soundfile (for streaming) and fall back to scipy.io.wavfile
try:
    import soundfile as sf
    HAVE_SF = True
except ImportError:
    HAVE_SF = False
    from scipy.io import wavfile

def hann(N: int) -> np.ndarray:
    n = np.arange(N, dtype=float)
    return 0.5 * (1 - np.cos(2 * np.pi * n / N))

def wrap180(a_deg: float) -> float:
    return (a_deg + 180.0) % 360.0 - 180.0

def tern(cond, a, b):
    return a if cond else b

def parab(y: np.ndarray, i: int) -> float:
    N = y.size
    iL = (i - 1 - 1) % N
    i0 = (i - 1) % N
    iR = (i) % N
    denom = (y[iL] - 2*y[i0] + y[iR] + np.finfo(float).eps)
    return (y[iL] - y[iR]) / (2 * denom)

def pairA(mic: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    P = pairs.shape[0]
    A = np.zeros((P, 2), dtype=float)
    for p in range(P):
        i, j = pairs[p]
        A[p, :] = mic[j, :] - mic[i, :]
    return A

def dir_est(mic: np.ndarray, pairs: np.ndarray, c: float, tdoa: np.ndarray) -> Tuple[np.ndarray, float]:
    A = pairA(mic, pairs)
    b = c * tdoa.reshape(-1, 1)
    u_est, *_ = np.linalg.lstsq(A, b, rcond=None)
    u = u_est.flatten()
    u /= (np.linalg.norm(u) + 1e-12)
    adeg = math.degrees(math.atan2(u[1], u[0]))
    return u, adeg

def gcc_sub(xw: np.ndarray, pairs: np.ndarray, N: int, fs: float, tauMax: float) -> Tuple[np.ndarray, np.ndarray]:
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
        idx = np.argmax(np.abs(cc))
        d = parab(cc, idx)
        lag = (idx - (N//2)) + d
        pk = np.max(np.abs(cc))
        med = np.median(np.abs(cc)) + np.finfo(float).eps
        w[p] = max(0.1, min(5.0, pk/med))
        tdoa[p] = max(-tauMax, min(lag/fs, tauMax))
    return tdoa, w

def scores_pair(Xi: np.ndarray, fs: float, N: int, c: float, mic: np.ndarray,
                u: np.ndarray, rhos: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    M = mic.shape[0]
    fk = np.arange(N, dtype=float)*(fs/N)
    band = (fk>=fmin)&(fk<=fmax)
    fk = fk[band].reshape(-1,1)
    X = Xi[band,:]/(np.abs(Xi[band,:])+np.finfo(float).eps)
    from itertools import combinations
    pairs = np.array(list(combinations(range(M),2)),dtype=int)
    P = pairs.shape[0]
    R = np.empty((fk.shape[0],P),dtype=complex)
    for p,(i,j) in enumerate(pairs):
        R[:,p] = X[:,i]*np.conj(X[:,j])
    scores = np.zeros(rhos.size,dtype=float)
    u_row = u.reshape(1,-1)
    for k,rho in enumerate(rhos):
        s = rho*u_row
        D = mic - s
        dist = np.linalg.norm(D,axis=1)
        tau = dist/c
        tauij = tau[pairs[:,1]] - tau[pairs[:,0]]
        Ftau = fk @ tauij.reshape(1,-1)
        W = np.exp(-1j*2*math.pi*Ftau)
        scores[k] = np.real(np.sum(R*W))
    return scores

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k<=1:
        return x.copy()
    return np.convolve(x, np.ones(k)/k, mode='same')

def Selftest1_compact_py(file_path: str='test0929.wav'):
    # === parameters ===
    N=512; c=343.0; Dia=0.08; r=Dia/2; M=8
    rhoL=0.01; rhoH=0.5; Ksmooth=5
    cohTh=1.7; vadTh=6.0; noiseA=0.98; srpA=0.85
    fmin=300.0; fmax=3400.0; nR=81
    rhoGrid=np.linspace(rhoL,rhoH,nR)
    dTheta=np.arange(-6,8,2)
    # file read setup
    if HAVE_SF:
        info=sf.info(file_path)
        fs=float(info.samplerate); total=info.frames; T=total/fs
        sf_reader=sf.SoundFile(file_path,'r')
    else:
        fs_i,data=wavfile.read(file_path)
        fs=float(fs_i)
        if data.ndim==1:
            data=data[:,None]
        if np.issubdtype(data.dtype,np.integer):
            data=data.astype(np.float32)/np.iinfo(data.dtype).max
        total=data.shape[0]; T=total/fs
        sf_reader=None
    ang=np.arange(M)*45.0
    mic=np.column_stack((r*np.cos(np.deg2rad(ang)),r*np.sin(np.deg2rad(ang))))
    from itertools import combinations
    Pairs=np.array(list(combinations(range(M),2)),dtype=int)
    # plotting
    plt.close('all')
    fig=plt.figure(num='GCC angle + SRP distance',figsize=(10,7))
    gs=fig.add_gridspec(2,2,width_ratios=[3,2],height_ratios=[1,1],wspace=0.25,hspace=0.25)
    ax1=fig.add_subplot(gs[0,0]); ax1.set_title('Angle vs Time'); ax1.set_xlim(0,T); ax1.set_ylim(-180,180); ax1.grid()
    ax2=fig.add_subplot(gs[1,0]); ax2.set_title('Distance vs Time'); ax2.set_xlim(0,T); ax2.set_ylim(0,1); ax2.grid()
    axp=fig.add_subplot(gs[:,1],projection='polar'); axp.set_title('Angle (fixed radius)')
    axp.set_theta_zero_location('E'); axp.set_theta_direction(1); axp.set_thetagrids(np.arange(0,360,30)); axp.set_rlim(0,r*1.2)
    h1,=ax1.plot([],[],linestyle='--'); h2,=ax1.plot([],[],linestyle='-')
    h3,=ax2.plot([],[],linestyle='--'); h4,=ax2.plot([],[],linestyle='-')
    for m in range(M):
        axp.plot([math.radians(ang[m])],[r],marker='^')
    hl,=axp.plot([],[],'-'); hm,=axp.plot([],[],'o')
    # buffers
    t_list=[]; Adeg=[]; Rho=[]; treal=0.0; prevRho=np.nan; tauMax=Dia/c
    srpAcc=np.zeros(nR); noiseRMS=None; win= hann(N).reshape(-1,1)
    
    # frame reader
    def read_frame():
        if HAVE_SF:
            frame = sf_reader.read(N, dtype='float64', always_2d=True)
            if frame.size == 0: return None
        else:
            if read_frame.idx >= total: return None
            i0 = read_frame.idx
            i1 = min(i0 + N, total)
            frame = data[i0:i1]
            read_frame.idx = i1
            if frame.shape[0] < N:
                pad = np.zeros((N - frame.shape[0], frame.shape[1]))
                frame = np.vstack((frame, pad))
    # 共同处理逻辑
        ch = frame.shape[1]
        if ch < M:
            frame = np.tile(frame, (1, int(np.ceil(M / ch))))[:, :M]
        elif ch > M:
            frame = frame[:, :M]
        return frame

# 初始化 idx（必须在函数定义后、循环前添加此行）
    if not HAVE_SF: read_frame.idx = 0
    # main loop
    while True:
        x=read_frame()
        if x is None: break
        xw=x*win; treal+=N/fs
        frmRMS=np.median(np.sqrt(np.mean(xw**2,axis=0)))
        if noiseRMS is None: noiseRMS=frmRMS
        snrDb=20*math.log10((frmRMS+1e-12)/(noiseRMS+1e-12))
        if snrDb<vadTh: noiseRMS=noiseA*noiseRMS+(1-noiseA)*frmRMS
        tdoa,w= gcc_sub(xw,Pairs,N,fs,tauMax)
        udir,adeg= dir_est(mic,Pairs,c,tdoa)
        if np.sum((pairA(mic,Pairs)@udir)*(c*tdoa))<0:
            udir=-udir; adeg=wrap180(adeg+180)
        Xi=np.fft.fft(xw,n=N,axis=0)
        good=(snrDb>=vadTh) and (np.median(w)>=cohTh)
        if not good:
            rho=0.20 if np.isnan(prevRho) else prevRho
        else:
            bestIdx=0; bestU=udir.copy(); bestA=adeg
            for th in adeg + dTheta:
                u=[math.cos(math.radians(th)),math.sin(math.radians(th))]
                sc=scores_pair(Xi,fs,N,c,mic,u,rhoGrid,fmin,fmax)
                tmp=srpA*srpAcc+(1-srpA)*sc
                idx=int(np.argmax(tmp))
                if tmp[idx]>srpAcc[bestIdx]:
                    bestIdx=idx; bestU=u; bestA=th
            sc=scores_pair(Xi,fs,N,c,mic,bestU,rhoGrid,fmin,fmax)
            srpAcc=srpA*srpAcc+(1-srpA)*sc
            bestIdx=int(np.argmax(srpAcc)); rho=float(rhoGrid[bestIdx])
            udir=bestU; adeg=bestA
        t_list.append(treal); Adeg.append(adeg); Rho.append(rho)
        A1=moving_average(np.array(Adeg),Ksmooth)
        R1=moving_average(np.array(Rho),Ksmooth)
        # update plots
        hl.set_data([math.radians(adeg),math.radians(adeg)],[0,r])
        hm.set_data([math.radians(adeg)],[r])
        h1.set_data(t_list,Adeg); h2.set_data(t_list,A1)
        h3.set_data(t_list,Rho); h4.set_data(t_list,R1)
        ax1.set_xlim(0,max(T,t_list[-1])); ax2.set_xlim(0,max(T,t_list[-1]))
        plt.pause(N/fs*0.2)
        prevRho=rho
    if HAVE_SF: sf_reader.close()
    plt.show()

if __name__=="__main__":
    path=sys.argv[1] if len(sys.argv)>1 else "test0929.wav"
    Selftest1_compact_py(path)
