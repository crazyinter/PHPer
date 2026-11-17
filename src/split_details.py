# -*- coding: utf-8 -*-
"""Component-level split detail: counts, class coverage, missing classes (comment 17)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from splits import build_sketches, mash_distance_matrix, connected_components, group_split  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def details(name, threshold=0.05, seed=123):
    if name == "cherry":
        ph = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
        sketches = build_sketches("cherry")
    else:
        ph = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
        sketches = list(np.load(PROC / "deephost_std_sketches.npz",
                                allow_pickle=True)["sketches"])
    y = np.array(ph["labels"]); n = len(y)
    dist = mash_distance_matrix(sketches)
    comp, nc = connected_components(dist, threshold)
    tr, va, te = group_split(comp, n, seed)
    sizes = np.bincount(comp)
    out = {"dataset": name, "threshold": threshold, "seed": seed,
           "total_components": int(nc),
           "component_size": {"largest": int(sizes.max()),
                              "singletons": int((sizes == 1).sum()),
                              "median": float(np.median(sizes))},
           "splits": {}}
    for label, idx in (("train", tr), ("val", va), ("test", te)):
        comps = set(int(comp[i]) for i in idx)
        classes = set(int(v) for v in y[idx])
        out["splits"][label] = {"samples": len(idx), "components": len(comps),
                                "classes_present": len(classes)}
    train_cls = set(int(v) for v in y[tr]); test_cls = set(int(v) for v in y[te])
    out["test_classes_absent_from_train"] = sorted(test_cls - train_cls)
    return out


report = [details("cherry"), details("deephost")]
(RES / "split_component_details.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
