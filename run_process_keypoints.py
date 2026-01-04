# run_process_keypoints.py
from build_streams import build_tcn_streams, build_stgcn_streams
from normalize_keypoints import normalize_sequence
from tqdm import tqdm
import numpy as np
import os

SRC = "D:\\sign_dictionary\\keypoints"
DST = "D:\\sign_dictionary\\keypoints_processed"
TARGET_T = 64

os.makedirs(DST, exist_ok=True)

files = [f for f in os.listdir(SRC) if f.endswith(".npy")]

for fname in tqdm(files):
    seq = np.load(os.path.join(SRC, fname))

    norm_seq = normalize_sequence(seq, target_T=TARGET_T)

    tcn = build_tcn_streams(norm_seq)
    tcn_dir = os.path.join(DST, "tcn")
    os.makedirs(tcn_dir, exist_ok=True)
    for k,v in tcn.items():
        np.save(os.path.join(tcn_dir, f"{fname[:-4]}_{k}.npy"), v)

    st = build_stgcn_streams(norm_seq)
    st_dir = os.path.join(DST, "stgcn")
    os.makedirs(st_dir, exist_ok=True)
    for k,v in st.items():
        np.save(os.path.join(st_dir, f"{fname[:-4]}_{k}.npy"), v)
