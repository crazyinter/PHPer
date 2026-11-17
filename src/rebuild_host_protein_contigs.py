# -*- coding: utf-8 -*-
"""Rebuild only the multi-contig DeepHost host protein features.

Single-contig hosts keep their stored values; only hosts whose FASTA contains
multiple contigs are recomputed with per-contig gene calling.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from protein_features import extract_features_contigs, _finder  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
D = ROOT / "data" / "fasta" / "deephost_hosts"


def read_contigs(i):
    out, cur = [], []
    for ln in (D / f"{i:03d}.fasta").read_text(
            encoding="utf-8", errors="replace").splitlines():
        if ln.startswith(">"):
            if cur:
                out.append("".join(cur)); cur = []
        else:
            cur.append(ln.strip())
    if cur:
        out.append("".join(cur))
    return [s for s in out if s]


present = sorted(set(np.load(PROC / "deephost_std_processed.npz",
                             allow_pickle=True)["labels"].tolist()))
old = np.load(PROC / "deephost_host_protein_features.npz", allow_pickle=True)
feats = old["features"].copy(); names = [str(x) for x in old["host_names"]]
finder = _finder()
changed = []
for k, i in enumerate(present):
    contigs = read_contigs(i)
    if len(contigs) > 1:
        feats[k] = extract_features_contigs(contigs, finder)
        changed.append((names[k], len(contigs)))
np.savez_compressed(PROC / "deephost_host_protein_features.npz",
                    features=feats.astype(np.float32),
                    host_names=np.array(names))
print("recomputed multi-contig hosts:", changed)
print("shape", feats.shape)
