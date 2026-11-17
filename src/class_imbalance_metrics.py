# -*- coding: utf-8 -*-
"""Class-imbalance metrics: balanced accuracy, frequency baselines, stratified F1."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
d = json.loads((RES / "detailed_eval_seed123.json").read_text(encoding="utf-8"))
pc = d["per_class"]

supports = np.array([v["support"] for v in pc.values()])
recalls = np.array([v["recall"] for v in pc.values()])
f1s = np.array([v["f1-score"] for v in pc.values()])
total = supports.sum()

balanced_acc = float(recalls.mean())
majority_baseline = float(supports.max() / total)
uniform_baseline = 1 / 206


def strat(mask):
    return {"classes": int(mask.sum()),
            "mean_f1": float(f1s[mask].mean()) if mask.sum() else 0.0,
            "zero_f1": int(((f1s == 0) & mask).sum())}


out = {
    "seed": 123, "split": "random", "test_classes_with_support": len(pc),
    "balanced_accuracy": round(balanced_acc, 4),
    "top1_accuracy": round(d["top1"], 4),
    "baselines": {"uniform_random": round(uniform_baseline, 4),
                  "majority_class": round(majority_baseline, 4)},
    "stratified": {
        "singleton_n1": strat(supports == 1),
        "n2_to_n4": strat((supports >= 2) & (supports <= 4)),
        "n5_plus": strat(supports >= 5)},
    "zero_f1_classes": int((f1s == 0).sum()),
}
(RES / "class_imbalance_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
