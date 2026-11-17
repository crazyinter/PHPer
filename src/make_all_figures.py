# -*- coding: utf-8 -*-
"""Redraw all result figures in the shared publication style."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_style import apply_style, PALETTE, SPLIT_COLORS, REPR_COLORS, add_err

apply_style()
ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES, FIG = ROOT / "results", ROOT / "figures"


def short_name(name):
    parts = str(name).split()
    return f"{parts[0][0]}. {parts[1]}" if len(parts) >= 2 else str(name)


# 1. Leakage across deep models on CHERRY -----------------------------------
def fig_leakage():
    df = pd.read_csv(RES / "cherry_summary.csv")
    keys = [("cnn", "fcgr", "contrastive", "CNN-only"),
            ("cnn_mamba", "fcgr", "contrastive", "CNN+BiMamba2"),
            ("cnn_mamba", "random", "contrastive", "Permuted FCGR"),
            ("cnn_mamba", "fcgr", "ce", "Cross-entropy")]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, split in enumerate(("random", "similarity")):
        means, errs = [], []
        for enc, rep, loss, _ in keys:
            r = df[(df.split == split) & (df.encoder == enc) &
                   (df.representation == rep) & (df.loss == loss)].iloc[0]
            means.append(r.test_top1_mean * 100); errs.append(r.test_top1_std * 100)
        xx = x + (j - 0.5) * w
        ax.bar(xx, means, w, color=SPLIT_COLORS[split], label=f"{split} split")
        add_err(ax, xx, means, errs)
    ax.set_xticks(x); ax.set_xticklabels([k[3] for k in keys])
    ax.set_ylabel("Test top-1 accuracy (%)"); ax.set_ylim(0, 90)
    ax.set_title("Neighbor leakage across model variants (CHERRY)")
    ax.legend()
    fig.savefig(FIG / "fig_leakage.png"); plt.close(fig)


# 2. Threshold dose-response -------------------------------------------------
def fig_threshold():
    d = json.loads((RES / "threshold_sensitivity.json").read_text())
    thr = [0.05, 0.10, 0.20]
    comp = [d["0.05"]["n_components"], d["0.1"]["n_components"], d["0.2"]["n_components"]]
    series = [("top1", "Top-1", PALETTE["blue"]), ("top5", "Top-5", PALETTE["green"]),
              ("mrr", "MRR", PALETTE["orange"])]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, lab, c in series:
        y = [d[str(t)][key] * 100 for t in thr]
        ax.plot(thr, y, "-o", color=c, label=lab, lw=1.8, ms=7)
    for t, n in zip(thr, comp):
        ax.annotate(f"{n} components", (t, 24), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color="#555555")
    ax.set_xlabel("Mash distance threshold"); ax.set_ylabel("Test metric (%)")
    ax.set_xticks(thr); ax.set_ylim(20, 70)
    ax.set_title("Dose-response of neighbor leakage (CHERRY)")
    ax.legend()
    fig.savefig(FIG / "fig_threshold.png"); plt.close(fig)


# 3. Protein vs FCGR vs Fusion across datasets -------------------------------
def fig_protein():
    cherry = pd.read_csv(RES / "fusion_summary.csv")
    deep = pd.read_csv(RES / "deephost_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    variants = [("fcgr", "FCGR"), ("protein", "Protein"), ("fusion", "Fusion")]
    x = np.arange(3); w = 0.38
    for ax, df, title in ((axes[0], cherry, "CHERRY (1,875 / 206)"),
                          (axes[1], deep, "DeepHost (2,845 / 108)")):
        for j, split in enumerate(("random", "similarity")):
            means, errs, cols = [], [], []
            for v, _ in variants:
                r = df[(df.split == split) & (df.variant == v)].iloc[0]
                means.append(r.top1_mean * 100); errs.append(r.top1_std * 100)
                cols.append(REPR_COLORS[{"fcgr": "FCGR", "protein": "Protein",
                                         "fusion": "Fusion"}[v]])
            xx = x + (j - 0.5) * w
            ax.bar(xx, means, w, color=cols, alpha=1 if j == 0 else 0.45)
            add_err(ax, xx, means, errs)
        ax.set_xticks(x); ax.set_xticklabels([v[1] for v in variants])
        ax.set_title(title)
    axes[0].set_ylabel("Test top-1 accuracy (%)"); axes[0].set_ylim(0, 95)
    handles = [Patch(facecolor="#666666", label="random split"),
               Patch(facecolor="#666666", alpha=0.45, label="similarity split")]
    axes[1].legend(handles=handles)
    fig.suptitle("Sequence-only, protein-only, and fused representations", y=1.02)
    fig.savefig(FIG / "fig_protein.png"); plt.close(fig)


# 4. Protein feature ablation ------------------------------------------------
def fig_ablation():
    d = json.loads((RES / "protein_ablation.json").read_text())
    groups = [("structure", "Structural"), ("codon", "Codon usage"),
              ("aa_phys", "AA / physico-chem."), ("codon_aa", "Codon + AA")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    x = np.arange(4); w = 0.6
    for ax, split, title in ((axes[0], "random", "Random split"),
                             (axes[1], "similarity", "Similarity split")):
        means, errs = [], []
        for g, _ in groups:
            vals = np.array(d[split][g]["values"]) * 100
            means.append(vals.mean()); errs.append(vals.std(ddof=1))
        ax.bar(x, means, w, color=[PALETTE["gray"], PALETTE["green"],
               PALETTE["teal"], PALETTE["purple"]])
        add_err(ax, x, means, errs)
        ax.set_xticks(x); ax.set_xticklabels([g[1] for g in groups], rotation=12)
        ax.set_title(title)
    axes[0].set_ylabel("Test top-1 accuracy (%)"); axes[0].set_ylim(0, 90)
    fig.suptitle("Protein feature-group ablation (CHERRY, n=3)", y=1.02)
    fig.savefig(FIG / "fig_ablation.png"); plt.close(fig)


# 5. Calibration --------------------------------------------------------------
def fig_calibration():
    t = json.loads((RES / "temperature_scaling_seed123.json").read_text())
    metrics = [("ECE", t["ece_before"], t["ece_after"]),
               ("NLL", t["nll_before"], t["nll_after"])]
    x = np.arange(2); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for j, (lab, col) in enumerate((("Before", PALETTE["red"]),
                                     ("After", PALETTE["green"]))):
        y = [m[j + 1] for m in metrics]
        xx = x + (j - 0.5) * w
        ax.bar(xx, y, w, color=col, label=f"{lab} scaling")
        for xi, yi in zip(xx, y):
            ax.text(xi, yi + 0.08, f"{yi:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["Expected calibration error", "Negative log-likelihood"])
    ax.set_ylabel("Error"); ax.set_title(f"Confidence calibration (T={t['temperature']:.3f})")
    ax.legend()
    fig.savefig(FIG / "fig_calibration.png"); plt.close(fig)


# 6. Confusion matrix ---------------------------------------------------------
def fig_confusion():
    cm = np.load(FIG / "confusion_seed123.npy")
    ph = np.load(ROOT / "data" / "processed" / "cherry_processed.npz", allow_pickle=True)
    names = [short_name(n) for n in ph["host_names"]]
    idx = np.argsort(cm.sum(1))[::-1][:15]
    sub = cm[np.ix_(idx, idx)]; labels = [names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(sub, cmap="BuGn")
    ax.set_xticks(range(15)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(15)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted host"); ax.set_ylabel("True host")
    ax.set_title("Confusion matrix, 15 best-supported host classes")
    ax.grid(False)
    for i in range(15):
        for j in range(15):
            if sub[i, j] > 0:
                ax.text(j, i, int(sub[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if sub[i, j] > sub.max() * 0.5 else "#333333")
    fig.colorbar(im, fraction=0.046)
    fig.savefig(FIG / "fig_confusion.png"); plt.close(fig)


# 7. Classical ML baselines vs deep models -----------------------------------
def fig_ml():
    ml = pd.read_csv(RES / "ml_baselines_summary.csv")
    deep = pd.read_csv(RES / "cherry_summary.csv")
    clfs = [("Logistic regression", "LR"), ("Random forest", "RF"),
            ("K-nearest neighbors", "KNN"), ("RBF support vector machine", "SVM")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    x = np.arange(4); w = 0.38
    for ax, ds, dref in ((axes[0], "cherry", "CHERRY"), (axes[1], "deephost", "DeepHost")):
        for j, split in enumerate(("random", "similarity")):
            means, errs = [], []
            for c, _ in clfs:
                r = ml[(ml.dataset == ds) & (ml.split == split) &
                       (ml.feature == "FCGR") & (ml.classifier == c)].iloc[0]
                means.append(r.top1_mean * 100); errs.append(r.top1_std * 100)
            xx = x + (j - 0.5) * w
            ax.bar(xx, means, w, color=SPLIT_COLORS[split])
            add_err(ax, xx, means, errs)
        # deep reference lines (full model CHERRY; fcgr deep DeepHost)
        if ds == "cherry":
            rnd = deep[(deep.split == "random") & (deep.encoder == "cnn_mamba") &
                       (deep.representation == "fcgr") & (deep.loss == "contrastive")].iloc[0]
            sim = deep[(deep.split == "similarity") & (deep.encoder == "cnn_mamba") &
                       (deep.representation == "fcgr") & (deep.loss == "contrastive")].iloc[0]
        else:
            dh = pd.read_csv(RES / "deephost_summary.csv")
            rnd = dh[(dh.split == "random") & (dh.variant == "fcgr")].iloc[0]
            sim = dh[(dh.split == "similarity") & (dh.variant == "fcgr")].iloc[0]
        ax.axhline(rnd.test_top1_mean * 100 if ds == "cherry" else rnd.top1_mean * 100,
                   color=SPLIT_COLORS["random"], ls="--", lw=1.3)
        ax.axhline(sim.test_top1_mean * 100 if ds == "cherry" else sim.top1_mean * 100,
                   color=SPLIT_COLORS["similarity"], ls="--", lw=1.3)
        ax.set_xticks(x); ax.set_xticklabels([c[1] for c in clfs])
        ax.set_title(dref)
    axes[0].set_ylabel("Test top-1 accuracy on FCGR (%)"); axes[0].set_ylim(0, 95)
    handles = [Patch(facecolor=SPLIT_COLORS["random"], label="random split"),
               Patch(facecolor=SPLIT_COLORS["similarity"], label="similarity split"),
               Patch(facecolor="none", label="dashed = deep model")]
    axes[1].legend(handles=handles)
    fig.suptitle("Classical ML classifiers versus deep models", y=1.02)
    fig.savefig(FIG / "fig_ml.png"); plt.close(fig)


if __name__ == "__main__":
    fig_leakage(); fig_threshold(); fig_protein(); fig_ablation()
    fig_calibration(); fig_confusion(); fig_ml()
    print("all figures redrawn in", FIG)
