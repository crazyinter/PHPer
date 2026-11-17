"""Detailed evaluation on the full model: confusion matrix, per-class, errors, ECE."""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from model_bmk import build_model
from train import load_dataset, random_split, recalibrate_bn
from train import rank_from_scores

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def expected_calibration_error(dist, labels, n_bins=10):
    """Convert distances to confidence via softmax of negative distance, then ECE."""
    sim = torch.softmax(-dist, dim=1)
    conf, pred = sim.max(1)
    acc = pred.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece, total = 0.0, len(labels)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += m.sum().item() / total * abs(acc[m].float().mean().item() - conf[m].mean().item())
    return ece


def main(seed=123):
    X, y, H, host_names = load_dataset("cherry")
    tr, va, te = random_split(len(X), seed)
    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    yt = torch.tensor(y).to(device)

    model = build_model(6).to(device)
    tag = f"cherry_random_cnn_mamba_fcgr_contrastive_seed{seed}"
    model.load_state_dict(torch.load(ROOT / "models" / f"{tag}.pt"))
    model = recalibrate_bn(model, Xt, Ht, tr, 32)
    with torch.no_grad():
        dist = torch.cdist(model(Xt[te]), model(Ht))
        top1, top5, mrr, pred = rank_from_scores(dist, yt[te], True)
    yte = y[te]

    cm = confusion_matrix(yte, pred, labels=range(len(host_names)))
    np.save(FIG / f"confusion_seed{seed}.npy", cm)
    report = classification_report(yte, pred, labels=range(len(host_names)),
                                   target_names=host_names, zero_division=0, output_dict=True)
    # top confusion pairs (excluding diagonal)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:10], off.shape))[0]
    top_errors = [{"true": host_names[t], "predicted": host_names[p], "count": int(off[t, p])}
                  for t, p in pairs if off[t, p] > 0]
    ece = expected_calibration_error(dist, yt[te])

    out = {"seed": seed, "top1": top1, "top5": top5, "mrr": mrr,
           "ece": ece, "top_confusions": top_errors,
           "per_class": {h: {k: report[h][k] for k in
                          ("precision", "recall", "f1-score", "support")}
                         for h in host_names if h in report and report[h]["support"] > 0}}
    (RES / f"detailed_eval_seed{seed}.json").write_text(json.dumps(out, indent=2),
                                                       encoding="utf-8")
    print(f"top1={top1:.4f} ECE={ece:.4f}")
    print("top errors:")
    for e in top_errors[:6]:
        print(" ", e)


if __name__ == "__main__":
    main()
