"""Aggregate experiment results into summary statistics and paired tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
METRICS = ["test_top1", "test_top5", "test_mrr", "macro_f1"]


def load_all(dataset):
    rows = []
    for f in sorted(RES.glob(f"{dataset}_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "split_type" in r:
            rows.append(r)
    return rows


def config_key(r):
    return (r["split_type"], r.get("encoder", "cnn_mamba"),
            r.get("representation", "fcgr"), r.get("loss", "contrastive"))


def summarize(dataset):
    rows = load_all(dataset)
    groups = {}
    for r in rows:
        groups.setdefault(config_key(r), []).append(r)
    summary = []
    for key, items in sorted(groups.items()):
        entry = dict(zip(("split", "encoder", "representation", "loss"), key))
        entry["n_runs"] = len(items)
        for m in METRICS:
            vals = np.array([i[m] for i in items])
            entry[f"{m}_mean"] = vals.mean()
            entry[f"{m}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary.append(entry)
    df = pd.DataFrame(summary)
    return df, groups


def paired_test(groups, key_a, key_b, metric="test_top1"):
    a = {r["seed"]: r[metric] for r in groups.get(key_a, [])}
    b = {r["seed"]: r[metric] for r in groups.get(key_b, [])}
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    va = np.array([a[s] for s in common])
    vb = np.array([b[s] for s in common])
    t, p = stats.ttest_rel(va, vb)
    return {"seeds": common, "mean_a": va.mean(), "mean_b": vb.mean(),
            "diff": (va - vb).mean(), "t": t, "p": p}


def main(dataset="cherry"):
    df, groups = summarize(dataset)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    df.to_csv(RES / f"{dataset}_summary.csv", index=False)

    full = ("random", "cnn_mamba", "fcgr", "contrastive")
    tests = {
        "full random vs similarity": (full, ("similarity", "cnn_mamba", "fcgr", "contrastive")),
        "full vs CNN-only (random)": (full, ("random", "cnn", "fcgr", "contrastive")),
        "FCGR vs random rep (random)": (full, ("random", "cnn_mamba", "random", "contrastive")),
        "contrastive vs CE (random)": (full, ("random", "cnn_mamba", "fcgr", "ce")),
    }
    out = {}
    for name, (ka, kb) in tests.items():
        r = paired_test(groups, ka, kb)
        out[name] = r
        if r:
            print(f"\n{name}: diff={r['diff']*100:.2f}pp t={r['t']:.3f} p={r['p']:.4f} n={len(r['seeds'])}")
    (RES / f"{dataset}_paired_tests.json").write_text(json.dumps(out, indent=2, default=str),
                                                     encoding="utf-8")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "cherry")
