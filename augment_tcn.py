from augmentation_ops import *
from stream_builders import *
import numpy as np
import os

SRC = "D:\\sign_dictionary\\keypoints_processed\\tcn"
DST = "D:\\sign_dictionary\\preprocess\\datasets\\tcn"
N_AUG = 2

os.makedirs(DST, exist_ok=True)

for f in os.listdir(SRC):
    if not f.endswith("_joint.npy"):
        continue

    base = f.replace("_joint.npy", "")
    joint = np.load(os.path.join(SRC, f))

    for k in range(N_AUG):
        x = joint.copy()

        x = temporal_scale(x)
        x = gaussian_jitter(x, sigma=0.01)
        x = motion_dropout(x)

        bone = build_bone(x)
        motion = compute_motion(x)

        np.save(os.path.join(DST, f"{base}_aug{k}_joint.npy"), x)
        np.save(os.path.join(DST, f"{base}_aug{k}_bone.npy"), bone)
        np.save(os.path.join(DST, f"{base}_aug{k}_motion.npy"), motion)

