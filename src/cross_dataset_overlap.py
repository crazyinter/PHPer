# -*- coding: utf-8 -*-
"""Cross-dataset overlap: accession/species intersection and FCGR nearest-neighbor checks."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"

ch = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
dh = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
ch_acc = set(base for base in ch["accs"]); dh_acc = set(dh["accs"])
ch_sp = set(s.strip().lower() for s in ch["host_names"])
dh_sp = set(s.strip().lower() for s in dh["host_names"])

# L2-normalize flattened FCGR, then cross-dataset cosine nearest neighbors
A = ch["matrices"].reshape(len(ch["accs"]), -1)
B = dh["matrices"].reshape(len(dh["accs"]), -1)
A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)

# block-wise nearest cosine similarity to limit memory
nn = np.empty(len(A))
for i in range(0, len(A), 256):
    block = A[i:i + 256] @ B.T
    nn[i:i + 256] = block.max(axis=1)

thresholds = {str(t): int((nn >= t).sum()) for t in (0.999, 0.99, 0.95, 0.9)}
out = {
    "phage_accession_intersection": len(ch_acc & dh_acc),
    "host_species_intersection": len(ch_sp & dh_sp),
    "cherry_host_species": len(ch_sp), "deephost_host_species": len(dh_sp),
    "cross_fcgr_nearest_cosine": {"mean": float(nn.mean()), "median": float(np.median(nn)),
                                  "max": float(nn.max())},
    "cherry_phages_with_close_deephost_neighbor": thresholds,
    "note": "Both resources draw on public phage sequence repositories; treat replication as "
            "cross-benchmark convergent evidence, not independent external validation.",
}
(RES / "cross_dataset_overlap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
