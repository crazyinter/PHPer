# -*- coding: utf-8 -*-
"""Build the standard 118-class DeepHost processed file (2,845 phages, 108 classes)."""
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data/processed"

std = {}
for line in (ROOT / "data/raw/DeepHost-master/data/species.txt").read_text().strip().splitlines():
    i, name = line.split("\t")
    std[name.strip()] = int(i)
std_names = [None] * len(std)
for name, i in std.items():
    std_names[i] = name

d = np.load(PROC / "deephost_processed.npz", allow_pickle=True)
# materialize arrays once (NpzFile re-extracts on every key access)
raw_mat = d["matrices"]
raw_labels = d["labels"]
raw_accs = d["accs"]
raw_names = [str(x) for x in d["host_names"]]
name_to_raw = {n: i for i, n in enumerate(raw_names)}

keep_mat, keep_lab, keep_acc = [], [], []
for raw_name, std_i in std.items():
    if raw_name not in name_to_raw:
        continue
    ri = name_to_raw[raw_name]
    idx = np.where(raw_labels == ri)[0]
    for k in idx:
        keep_mat.append(raw_mat[k])
        keep_lab.append(std_i)
        keep_acc.append(str(raw_accs[k]))

matrices = np.stack(keep_mat).astype(np.float32)
labels = np.array(keep_lab, dtype=np.int64)
accs = np.array(keep_acc, dtype=object)
print("phages:", len(labels), "classes covered:", len(set(labels.tolist())))
np.savez_compressed(PROC / "deephost_std_processed.npz",
                    matrices=matrices, labels=labels,
                    host_names=np.array(std_names), accs=accs)
print("saved deephost_std_processed.npz")
