# -*- coding: utf-8 -*-
"""Full-class error summary and figure (reviewer #21, #30).

Builds the complete confusion matrix over every candidate host, writes a
per-class error table (support/TP/FP/FN/top misclassification) and plots
per-class recall for every class that has test support.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model_bmk import build_model  # noqa
from train import load_dataset, random_split, recalibrate_bn  # noqa
from plot_style import apply_style, PALETTE  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; CKPT = ROOT / "models"; FIG = ROOT / "figures"
device = "cuda" if torch.cuda.is_available() else "cpu"

CFGS = [
    ("cherry", "cherry_random_cnn_mamba_fcgr_contrastive_seed123.pt"),
    ("deephost", "deephost_random_fcgr_seed123.pt"),
]
summary = {}


def analyze(dataset, ckpt):
    X, y, H, names = load_dataset(dataset)
    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    tr, va, te = random_split(len(X), 123)
    model = build_model(6).to(device)
    model.load_state_dict(torch.load(CKPT / ckpt))
    model = recalibrate_bn(model, Xt, Ht, tr, 32)
    model.eval()
    with torch.no_grad():
        scores = torch.cdist(model(Xt[te]), model(Ht)).cpu().numpy()
    yte = y[te]; pred = scores.argmin(1)
    n_cls = len(names)
    C = np.zeros((n_cls, n_cls), int)
    for t, p in zip(yte, pred):
        C[t, p] += 1

    rows = []
    for c in range(n_cls):
        support = int(C[c].sum())
        if support == 0:
            continue
        tp = int(C[c, c]); fp = int(C[:, c].sum() - tp); fn = support - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / support; f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        mis = C[c].copy(); mis[c] = 0
        if mis.sum():
            tgt = int(mis.argmax()); top_mis = names[tgt]; top_mis_n = int(mis[tgt])
        else:
            top_mis = ""; top_mis_n = 0
        rows.append({"class": names[c], "support": support, "TP": tp, "FP": fp,
                     "FN": fn, "precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "top_misclassification": top_mis,
                     "top_misclassification_count": top_mis_n})
    rows.sort(key=lambda r: -r["support"])
    csv_path = RES / f"full_class_errors_{dataset}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # full-class recall figure, ordered by support
    apply_style()
    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(rows))))
    labels = [r["class"] for r in rows]; recs = [r["recall"] for r in rows]
    colors = [PALETTE["green"] if v >= 0.5 else PALETTE["red"] for v in recs]
    ax.barh(range(len(rows))[::-1], recs, color=colors)
    ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Per-class recall"); ax.set_xlim(0, 1.02)
    ax.set_title(f"Per-class recall across all supported classes ({dataset})")
    fig_path = FIG / f"fig_full_class_recall_{dataset}.png"
    fig.savefig(fig_path); plt.close(fig)

    summary[dataset] = {"classes_with_support": len(rows),
                        "zero_recall_classes": sum(r["recall"] == 0 for r in rows),
                        "zero_f1_classes": sum(r["f1"] == 0 for r in rows),
                        "mean_recall": round(float(np.mean([r["recall"] for r in rows])), 4),
                        "csv": str(csv_path.name), "figure": str(fig_path.name)}


for cfg in CFGS:
    analyze(*cfg)
(RES / "full_class_errors_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
