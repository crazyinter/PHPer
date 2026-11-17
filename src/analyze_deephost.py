# -*- coding: utf-8 -*-
"""Summarize DeepHost parallel experiments and run paired t-tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
SEEDS = [123, 2024, 42, 7, 99]
VARIANTS = ["fcgr", "protein", "fusion"]


def load(split, variant, seed):
    return json.loads((RES / f"deephost_{split}_{variant}_seed{seed}.json").read_text(
        encoding="utf-8"))


rows = []
for split in ("random", "similarity"):
    for variant in VARIANTS:
        rs = [load(split, variant, s) for s in SEEDS]
        rows.append({
            "split": split, "variant": variant,
            "top1_mean": np.mean([r["test_top1"] for r in rs]),
            "top1_std": np.std([r["test_top1"] for r in rs], ddof=1),
            "top5_mean": np.mean([r["test_top5"] for r in rs]),
            "mrr_mean": np.mean([r["test_mrr"] for r in rs]),
            "f1_mean": np.mean([r["macro_f1"] for r in rs]),
        })
df = pd.DataFrame(rows)
pd.set_option("display.width", 150)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
df.to_csv(RES / "deephost_summary.csv", index=False)


def paired(split, a, b, metric="test_top1"):
    va = [load(split, a, s)[metric] for s in SEEDS]
    vb = [load(split, b, s)[metric] for s in SEEDS]
    t, p = ttest_rel(va, vb)
    return {"comparison": f"{a} vs {b}", "split": split,
            "diff_pp": (np.mean(va) - np.mean(vb)) * 100, "t": t, "p": p}


tests = []
for split in ("random", "similarity"):
    tests.append(paired(split, "fusion", "fcgr"))
    tests.append(paired(split, "fusion", "protein"))
    tests.append(paired(split, "protein", "fcgr"))
print(f"\nPaired t-tests (top-1, n={len(SEEDS)}):")
for t in tests:
    print(f"  [{t['split']:10s}] {t['comparison']:18s} diff={t['diff_pp']:+.2f}pp "
          f"t={t['t']:+.3f} p={t['p']:.4f}")
(RES / "deephost_paired_tests.json").write_text(json.dumps(tests, indent=2),
                                                encoding="utf-8")
