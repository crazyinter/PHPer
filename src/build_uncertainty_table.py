# -*- coding: utf-8 -*-
"""Aggregate every per-run JSON into mean +/- sample SD for all four metrics across seeds."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"

METRICS = [("test_top1", "top1"), ("test_top5", "top5"),
           ("test_mrr", "mrr"), ("macro_f1", "macro_f1")]
groups = defaultdict(list)
for f in sorted(RES.glob("*.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if "test_top1" not in d:
        continue
    family = re.sub(r"_seed\d+$", "", f.stem)
    groups[family].append(d)

rows = []
for family in sorted(groups):
    runs = groups[family]
    n = len(runs)
    entry = {"run_family": family, "n_seeds": n,
             "dataset": runs[0].get("dataset", ""), "split": runs[0].get("split_type", "")}
    for key, label in METRICS:
        vals = np.array([r[key] for r in runs], dtype=float)
        entry[f"{label}_mean"] = round(float(vals.mean()), 4)
        entry[f"{label}_sd"] = round(float(vals.std(ddof=1)), 4) if n > 1 else 0.0
    rows.append(entry)

cols = ["run_family", "n_seeds", "dataset", "split"]
for _, label in METRICS:
    cols += [f"{label}_mean", f"{label}_sd"]
with open(RES / "uncertainty_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(rows)
(RES / "uncertainty_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("families:", len(rows))
for r in rows[:8]:
    print(r["run_family"], "n", r["n_seeds"],
          "top1", r["top1_mean"], r["top1_sd"], "top5", r["top5_mean"], r["top5_sd"],
          "mrr", r["mrr_mean"], r["mrr_sd"], "f1", r["macro_f1_mean"], r["macro_f1_sd"])
