# -*- coding: utf-8 -*-
"""Incrementally build host protein features with resumable cache."""
import csv
import pickle
from pathlib import Path

import numpy as np

from protein_features import extract_features, read_gz

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
C = ROOT / "data" / "raw" / "CHERRY-main"
PROC = ROOT / "data" / "processed"
CACHE = PROC / "host_protein_cache.pkl"

hf = np.load(PROC / "cherry_host_fcgr.npz", allow_pickle=True)
host_names = list(hf["host_names"])

acc2sp = {}
with open(C / "dataset" / "prokaryote.csv", encoding="utf-8", newline="") as fh:
    for r in csv.DictReader(fh):
        acc2sp[r["Accession"]] = r["Species"]
# gz accession grouped by species, first available used first
sp2gz = {}
for p in sorted((C / "prokaryote").glob("*.fa.gz")):
    acc = p.name.replace(".fa.gz", "")
    sp = acc2sp[acc]
    sp2gz.setdefault(sp, []).append(p)

cache = {}
if CACHE.exists():
    cache = pickle.loads(CACHE.read_bytes())

done = 0
for sp in host_names:
    if sp in cache:
        done += 1
        continue
    cache[sp] = extract_features(read_gz(sp2gz[sp][0]))
    done += 1
    if done % 5 == 0:
        CACHE.write_bytes(pickle.dumps(cache))
        print(f"  {done}/{len(host_names)} saved", flush=True)

CACHE.write_bytes(pickle.dumps(cache))
hpf = np.stack([cache[sp] for sp in host_names]).astype(np.float32)
np.savez_compressed(PROC / "cherry_host_protein_features.npz",
                    features=hpf, host_names=np.array(host_names))
print("final host protein features:", hpf.shape)
