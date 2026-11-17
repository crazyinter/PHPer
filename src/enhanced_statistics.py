# -*- coding: utf-8 -*-
"""Enhanced paired statistics: t test, Wilcoxon, 95% CI, and Cohen's d."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; SEEDS = [123, 2024, 42, 7, 99]


def load_top1(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["test_top1"]


def series(template, variant):
    return np.array([load_top1(RES / template.format(v=variant, s=s)) for s in SEEDS])


def paired(a_name, a, b_name, b, context):
    d = (a - b) * 100.0  # percentage points
    n = len(d); mean = d.mean(); sd = d.std(ddof=1)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci_half = tcrit * sd / np.sqrt(n)
    tstat, tp = stats.ttest_rel(a, b)
    try:
        wstat, wp = stats.wilcoxon(a, b, alternative="two-sided", method="exact")
    except ValueError:
        wstat, wp = float("nan"), float("nan")
    return {
        "context": context, "contrast": f"{a_name} vs {b_name}", "n_seeds": n,
        "diff_pp": round(float(mean), 3),
        "ci95_low": round(float(mean - ci_half), 3),
        "ci95_high": round(float(mean + ci_half), 3),
        "paired_t": round(float(tstat), 3), "t_p": round(float(tp), 5),
        "wilcoxon_W": round(float(wstat), 2), "wilcoxon_p": round(float(wp), 5),
        "cohen_d": round(float(mean / sd), 3) if sd > 0 else None,
    }


out = []
# DeepHost controlled comparisons
for split in ("random", "similarity"):
    tmpl = f"deephost_{split}_{{v}}_seed{{s}}.json"
    fcgr = series(tmpl, "fcgr"); protein = series(tmpl, "protein"); fusion = series(tmpl, "fusion")
    out.append(paired("Protein", protein, "FCGR", fcgr, f"DeepHost, {split}"))
    out.append(paired("Fusion", fusion, "FCGR", fcgr, f"DeepHost, {split}"))
    out.append(paired("Fusion", fusion, "Protein", protein, f"DeepHost, {split}"))

# CHERRY fusion controlled comparisons
for split in ("random", "similarity"):
    tmpl = f"cherry_{split}_{{v}}_fusion_seed{{s}}.json"
    fcgr = series(tmpl, "fcgr"); protein = series(tmpl, "protein"); fusion = series(tmpl, "fusion")
    out.append(paired("Protein", protein, "FCGR", fcgr, f"CHERRY, {split}"))
    out.append(paired("Fusion", fusion, "FCGR", fcgr, f"CHERRY, {split}"))
    out.append(paired("Fusion", fusion, "Protein", protein, f"CHERRY, {split}"))

# CHERRY leakage: full model random vs similarity, same seeds
rnd = series("cherry_random_{v}_contrastive_seed{s}.json", "cnn_mamba_fcgr")
sim = series("cherry_similarity_{v}_contrastive_seed{s}.json", "cnn_mamba_fcgr")
out.append(paired("Random split", rnd, "Similarity split", sim, "CHERRY, full model leakage"))

(RES / "enhanced_statistics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
for r in out:
    print(f"{r['context']:28s} {r['contrast']:34s} d={r['diff_pp']:7.2f}pp "
          f"CI[{r['ci95_low']:6.2f},{r['ci95_high']:6.2f}] t_p={r['t_p']:.4f} "
          f"Wilcoxon_p={r['wilcoxon_p']:.4f} d={r['cohen_d']}")
print("saved enhanced_statistics.json")
