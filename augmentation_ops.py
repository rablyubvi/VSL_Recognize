import numpy as np

def temporal_scale(seq, scale_range=(0.9, 1.1)):
    T, V, C = seq.shape
    scale = np.random.uniform(*scale_range)
    new_T = int(T * scale)

    src_idx = np.linspace(0, T-1, T)
    dst_idx = np.linspace(0, T-1, new_T)

    out = np.zeros((new_T, V, C), dtype=seq.dtype)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(dst_idx, src_idx, seq[:, v, c])

    return out


def gaussian_jitter(seq, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise


def random_rotation_z(seq, max_deg=5):
    theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    return seq @ R.T


def motion_dropout(seq, drop_prob=0.05):
    T = seq.shape[0]
    for t in range(1, T):
        if np.random.rand() < drop_prob:
            seq[t] = seq[t-1]
    return seq
