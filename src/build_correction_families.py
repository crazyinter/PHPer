# -*- coding: utf-8 -*-
"""Recompute and audit Holm/BH adjusted p with explicit family definitions and full-precision raw p."""
import csv
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"

raw = json.loads((RES / "multiple_comparison_corrected.json").read_text(encoding="utf-8"))


def holm_adjusted(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        val = min(1.0, (m - rank + 1) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj, order


def bh_adjusted(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = min(1.0, m * pvals[idx] / rank)
        running = min(running, val)
        adj[idx] = running
    return adj


# One global family: all 13 tests (12 representation contrasts + 1 leakage partition).
pvals = [float(r["t_p"]) for r in raw]
m = len(pvals)
holm, order = holm_adjusted(pvals)
bh = bh_adjusted(pvals)

# rank lookup
rank_of = {idx: r for r, idx in enumerate(order, start=1)}

rows = []
mismatch = 0
for i, r in enumerate(raw):
    existing_h = float(r["holm_adj_p"])
    existing_b = float(r["bh_adj_p"])
    ok_h = abs(existing_h - holm[i]) < 5e-4
    ok_b = abs(existing_b - bh[i]) < 5e-4
    if not (ok_h and ok_b):
        mismatch += 1
    rows.append({
        "family": "F1 (global)",
        "rank": rank_of[i],
        "context": r["context"],
        "contrast": r["contrast"],
        "n_seeds": r["n_seeds"],
        "raw_p_full": f"{pvals[i]:.6f}",
        "holm_recomputed": f"{holm[i]:.6f}",
        "bh_recomputed": f"{bh[i]:.6f}",
        "reported_holm": f"{existing_h:.4f}",
        "reported_bh": f"{existing_b:.4f}",
        "match": "yes" if (ok_h and ok_b) else "NO",
    })
rows.sort(key=lambda d: d["rank"])

with open(RES / "correction_full_list.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

summary = {
    "family_definition": {
        "family_id": "F1",
        "scope": "one global family containing every inferential test in the study",
        "m_tests": m,
        "members": "12 representation contrasts (2 datasets x 2 splits x 3 pairwise contrasts) plus 1 full-model random-vs-similarity partition test",
        "test_statistic": "two-sided paired t-test on seed-level top-1; raw p carried to 6 decimals",
        "holm": "step-down family-wise error control, multiplier (m-rank+1)",
        "benjamini_hochberg": "step-up false-discovery control, m*p/rank",
    },
    "n_mismatches_vs_reported": mismatch,
    "full_precision_rows": rows,
}
(RES / "correction_families.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("family m =", m, "mismatches =", mismatch)
for d in rows:
    print(d["rank"], d["context"], d["contrast"], "raw", d["raw_p_full"],
          "holm", d["holm_recomputed"], "bh", d["bh_recomputed"], d["match"])
