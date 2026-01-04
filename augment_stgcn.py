from augmentation_ops import *
from stream_builders import *
import numpy as np
import os

SRC = "D:\\sign_dictionary\\keypoints_processed\\stgcn"
DST = "D:\\sign_dictionary\\preprocess\\datasets\\stgcn"
N_AUG = 2

os.makedirs(DST, exist_ok=True)

for f in os.listdir(SRC):
    if not f.endswith("_joint.npy"):
        continue

    base = f.replace("_joint.npy", "")
    joint = np.load(os.path.join(SRC, f))   # (C, T, V)
    joint = joint.transpose(1, 2, 0)        # → (T, V, C)

    for k in range(N_AUG):
        x = joint.copy()

        x = temporal_scale(x)
        x = gaussian_jitter(x, sigma=0.005)
        x = motion_dropout(x)

        bone = build_bone(x)
        jm = compute_motion(x)
        bm = compute_motion(bone)

        def to_stgcn(y):
            return y.transpose(2, 0, 1)

        np.save(os.path.join(DST, f"{base}_aug{k}_joint.npy"), to_stgcn(x))
        np.save(os.path.join(DST, f"{base}_aug{k}_bone.npy"), to_stgcn(bone))
        np.save(os.path.join(DST, f"{base}_aug{k}_joint_motion.npy"), to_stgcn(jm))
        np.save(os.path.join(DST, f"{base}_aug{k}_bone_motion.npy"), to_stgcn(bm))
