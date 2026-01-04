import numpy as np
from build_streams import *

print("EDGES:", EDGES)
def build_bone(seq, edges=EDGES):
    T, V, C = seq.shape
    bone = np.zeros_like(seq)
    for i, j in edges:
        bone[:, j] = seq[:, j] - seq[:, i]
    return bone


def compute_motion(seq):
    motion = np.zeros_like(seq)
    motion[1:] = seq[1:] - seq[:-1]
    return motion
