# -*- coding: utf-8 -*-
"""Multi-seed Mash threshold sweep (replaces single-seed reporting with mean +/- sample SD)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from splits import build_sketches, mash_distance_matrix, connected_components, group_split
from train import load_dataset
from threshold_sensitivity import run_one

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
SEEDS = [123, 2024, 42]


def msd(vals):
    a = np.array(vals)
    return round(float(a.mean()), 4), round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0


X, y, H, names = load_dataset("cherry")
sketches = build_sketches("cherry")
dmat = mash_distance_matrix(sketches)
out = {}
for thr in (0.05, 0.1, 0.2):
    comp, nc = connected_components(dmat, thr)
    t1, t5, rr = [], [], []
    for seed in SEEDS:
        tr, va, te = group_split(comp, len(sketches), seed)
        a, b, c = run_one(X, y, H, tr, va, te, seed)
        t1.append(a); t5.append(b); rr.append(c)
        print(f"thr={thr} seed={seed} top1={a:.4f}", flush=True)
    out[str(thr)] = {
        "n_components": int(nc), "seeds": SEEDS,
        "top1_mean": msd(t1)[0], "top1_sd": msd(t1)[1],
        "top5_mean": msd(t5)[0], "top5_sd": msd(t5)[1],
        "mrr_mean": msd(rr)[0], "mrr_sd": msd(rr)[1],
    }
(RES / "threshold_sensitivity_multi.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("saved threshold_sensitivity_multi.json")
for k, v in out.items():
    print(k, "top1", v["top1_mean"], "+/-", v["top1_sd"],
          "top5", v["top5_mean"], "+/-", v["top5_sd"],
          "mrr", v["mrr_mean"], "+/-", v["mrr_sd"])
