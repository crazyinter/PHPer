"""Generate publication figures from real results."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

LABELS = [
    ("cnn", "fcgr", "contrastive", "CNN-only"),
    ("cnn_mamba", "fcgr", "contrastive", "CNN+BiMamba2 (full)"),
    ("cnn_mamba", "random", "contrastive", "Random permutation"),
    ("cnn_mamba", "fcgr", "ce", "Cross-entropy"),
]


def collect():
    rows = []
    for f in RES.glob("cherry_*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "split_type" in r:
            rows.append(r)
    return rows


def fig_main(rows):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LABELS)); w = 0.36
    for j, split in enumerate(["random", "similarity"]):
        means, errs = [], []
        for enc, rep, loss, _ in LABELS:
            vals = [r["test_top1"] for r in rows if r["split_type"] == split
                    and r.get("encoder", "cnn_mamba") == enc
                    and r.get("representation", "fcgr") == rep
                    and r.get("loss", "contrastive") == loss]
            means.append(np.mean(vals) * 100)
            errs.append(np.std(vals, ddof=1) * 100 if len(vals) > 1 else 0)
        ax.bar(x + (j - 0.5) * w, means, w, yerr=errs, capsize=4,
               label=f"{split} split", color=["#4C72B0", "#C44E52"][j])
    ax.set_xticks(x); ax.set_xticklabels([l[3] for l in LABELS])
    ax.set_ylabel("Test top-1 accuracy (%)"); ax.set_ylim(0, 90)
    ax.set_title("Random vs similarity-aware split (CHERRY, mean +/- SD)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig_split_comparison.png", dpi=300)
    plt.close(fig)


def fig_calibration():
    ts = json.loads((RES / "temperature_scaling_seed123.json").read_text())
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Before", "After temp. scaling"], [ts["ece_before"], ts["ece_after"]],
           color=["#8172B3", "#55A868"])
    for i, v in enumerate([ts["ece_before"], ts["ece_after"]]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_ylabel("Expected calibration error (ECE)"); ax.set_ylim(0, 0.85)
    ax.set_title(f"Confidence calibration (T={ts['temperature']:.3f})")
    fig.tight_layout(); fig.savefig(FIG / "fig_calibration.png", dpi=300)
    plt.close(fig)


def short_name(name):
    parts = str(name).split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[1]}"
    return str(name)


def fig_confusion():
    cm = np.load(FIG / "confusion_seed123.npy")
    ph = np.load(ROOT / "data" / "processed" / "cherry_processed.npz", allow_pickle=True)
    names = [short_name(n) for n in ph["host_names"]]
    # keep the 15 classes with most support for readability
    support = cm.sum(1); idx = np.argsort(support)[::-1][:15]
    sub = cm[np.ix_(idx, idx)]
    labels = [names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(sub, cmap="Blues")
    ax.set_xticks(range(15)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(15)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted host"); ax.set_ylabel("True host")
    ax.set_title("Confusion matrix (top 15 host classes)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG / "fig_confusion.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    rows = collect()
    fig_main(rows); fig_calibration(); fig_confusion()
    print("figures saved to", FIG)
