# normalize_keypoints.py
import numpy as np
from scipy.signal import savgol_filter

IDX_POSE = 0
IDX_LEFT_HAND = 6
IDX_RIGHT_HAND = 27
def unflatten_seq(seq, V=48, C=3):
    if seq.ndim == 2 and seq.shape[1] == V * C:
        return seq.reshape(seq.shape[0], V, C)
    return seq

def ensure_3d(seq):
    if seq.shape[-1] == 2:
        z = np.zeros((*seq.shape[:2], 1), dtype=seq.dtype)
        seq = np.concatenate([seq, z], axis=-1)
    return seq.astype(np.float32)

def root_center_mid_shoulder(seq, l=IDX_POSE, r=IDX_POSE+1):
    root = (seq[:, l] + seq[:, r]) / 2.0
    return seq - root[:, None, :]

def compute_shoulder_distance_mean(seq, l=IDX_POSE, r=IDX_POSE+1):
    d = np.linalg.norm(seq[:, l] - seq[:, r], axis=1)
    d = d[d > 1e-6]
    return d.mean() if len(d) > 0 else 1.0

def savgol_smooth(seq, win=5, poly=2):
    T, V, C = seq.shape
    if T < win:
        return seq
    out = np.zeros_like(seq)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = savgol_filter(seq[:, v, c], win, poly)
    return out

def resample_sequence(seq, target_T):
    T, V, C = seq.shape
    if T == target_T:
        return seq
    src = np.linspace(0, T - 1, T)
    dst = np.linspace(0, T - 1, target_T)
    out = np.zeros((target_T, V, C), dtype=seq.dtype)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(dst, src, seq[:, v, c])
    return out

def normalize_sequence(
    seq,
    target_T=64,
    smooth=True
):
    seq = unflatten_seq(seq)
    seq = ensure_3d(seq)
    seq = root_center_mid_shoulder(seq)
    scale = compute_shoulder_distance_mean(seq)
    seq = seq / (scale + 1e-8)

    if smooth:
        seq = savgol_smooth(seq)

    seq = resample_sequence(seq, target_T)
    return seq
