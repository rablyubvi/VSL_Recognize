# build_streams.py
import numpy as np

MP_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

IDX_POSE = 0
IDX_LEFT_HAND = 6
IDX_RIGHT_HAND = 27

HAND_LEFT = list(range(IDX_LEFT_HAND, IDX_LEFT_HAND + 21))
HAND_RIGHT = list(range(IDX_RIGHT_HAND, IDX_RIGHT_HAND + 21))
HAND_ALL = HAND_LEFT + HAND_RIGHT


def build_edges():
    edges = []
    for a,b in MP_HAND_CONNECTIONS:
        edges.append((IDX_LEFT_HAND+a, IDX_LEFT_HAND+b))
        edges.append((IDX_RIGHT_HAND+a, IDX_RIGHT_HAND+b))
    edges += [
        (0,2),(2,4),
        (1,3),(3,5),
        (4, IDX_LEFT_HAND),
        (5, IDX_RIGHT_HAND)
    ]
    return edges

EDGES = build_edges()

def compute_hand_scale(seq, hand_indices, edges):
    """
    Scale = mean bone length của các bone thuộc bàn tay
    """
    lens = []
    for (a, b) in edges:
        if a in hand_indices and b in hand_indices:
            d = np.linalg.norm(seq[:, b] - seq[:, a], axis=1)
            d = d[d > 1e-6]
            if len(d) > 0:
                lens.append(d.mean())

    if len(lens) == 0:
        return 1.0
    return float(np.mean(lens))

def compute_motion(seq, hand_gain=1.3):
    motion = np.diff(seq, axis=0)
    pad = np.zeros((1, seq.shape[1], seq.shape[2]), dtype=seq.dtype)
    motion = np.concatenate([motion, pad], axis=0)

    motion[:, HAND_ALL] *= hand_gain
    return motion


def build_bone(seq, edges):
    bone = np.zeros_like(seq)
    hand_scale = compute_hand_scale(seq, HAND_ALL, edges)

    for (a, b) in edges:
        v = seq[:, b] - seq[:, a]

        if a in HAND_ALL and b in HAND_ALL:
            bone[:, b] = v / (hand_scale + 1e-8)
        else:
            bone[:, b] = v 

    return bone

def build_tcn_streams(seq):
    joint = seq
    bone = build_bone(seq, EDGES)
    motion = compute_motion(seq)
    bone_motion = compute_motion(bone)

    return {
        "joint": joint,
        "bone": bone,
        "motion": motion,
        "bone_motion": bone_motion
    }

def build_stgcn_streams(seq):
    joint = seq
    bone = build_bone(seq, EDGES)
    jm = compute_motion(joint)
    bm = compute_motion(bone)

    def to_stgcn(x):
        return x.transpose(2,0,1)

    return {
        "joint": to_stgcn(joint),
        "bone": to_stgcn(bone),
        "joint_motion": to_stgcn(jm),
        "bone_motion": to_stgcn(bm)
    }