# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; FIG = ROOT / "figures"
SEEDS = [123, 42, 2024, 7, 99]
VARIANTS = [("fcgr", "FCGR (sequence)"), ("protein", "Protein-level"),
            ("fusion", "Fusion")]


def vals(split, variant):
    out = []
    for s in SEEDS:
        r = json.loads((RES / f"cherry_{split}_{variant}_fusion_seed{s}.json").read_text())
        out.append(r["test_top1"] * 100)
    return out


fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(3); w = 0.36
for j, split in enumerate(["random", "similarity"]):
    means, errs = [], []
    for v, _ in VARIANTS:
        d = vals(split, v)
        means.append(np.mean(d)); errs.append(np.std(d, ddof=1))
    ax.bar(x + (j - 0.5) * w, means, w, yerr=errs, capsize=4,
           label=f"{split} split", color=["#4C72B0", "#C44E52"][j])
ax.set_xticks(x); ax.set_xticklabels([v[1] for v in VARIANTS])
ax.set_ylabel("Test top-1 accuracy (%)"); ax.set_ylim(0, 90)
ax.set_title("Sequence vs protein-level vs fusion (CHERRY, mean +/- SD, n=5)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "fig_fusion_comparison.png", dpi=300)
print("saved fig_fusion_comparison.png")
