# -*- coding: utf-8 -*-
"""Phage-level (test-sample) bootstrap confidence intervals (reviewer #18, #20).

Resamples individual test phages with replacement rather than seeds, giving
sample-level CIs for top-1/top-5/MRR and per-class recall/precision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_bmk import build_model  # noqa
from train import load_dataset, random_split, recalibrate_bn  # noqa
from splits import get_similarity_split  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; CKPT = ROOT / "models"
device = "cuda" if torch.cuda.is_available() else "cpu"
B = 2000
SEED = 123

CONFIGS = [
    ("cherry", "random", "cherry_random_cnn_mamba_fcgr_contrastive_seed123.pt"),
    ("cherry", "similarity", "cherry_similarity_cnn_mamba_fcgr_contrastive_seed123.pt"),
    ("deephost", "random", "deephost_random_fcgr_seed123.pt"),
    ("deephost", "similarity", "deephost_similarity_fcgr_seed123.pt"),
]


def pct_ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def analyze(dataset, split_type, ckpt):
    X, y, H, names = load_dataset(dataset)
    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    if split_type == "random":
        tr, va, te = random_split(len(X), SEED)
    else:
        sname = "deephost_std" if dataset == "deephost" else "cherry"
        tr, va, te = get_similarity_split(sname, SEED, 0.05)
    model = build_model(6).to(device)
    model.load_state_dict(torch.load(CKPT / ckpt))
    model = recalibrate_bn(model, Xt, Ht, tr, 32)
    model.eval()
    with torch.no_grad():
        scores = torch.cdist(model(Xt[te]), model(Ht)).cpu().numpy()
    ytrue = y[te]
    order = scores.argsort(1)
    rank = np.empty(len(te), int)
    for i, o in enumerate(order):
        rank[i] = np.where(o == ytrue[i])[0][0]
    pred = order[:, 0]
    correct1 = (rank == 0); correct5 = (rank < 5); rr = 1.0 / (rank + 1)
    n_te = len(te)

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_te, size=(B, n_te))
    top1b = correct1[draws].mean(1); top5b = correct5[draws].mean(1)
    mrrb = rr[draws].mean(1)

    per_class = {}
    for c in np.unique(ytrue):
        idx = np.where(ytrue == c)[0]
        rc = correct1[idx]
        rdraws = rng.integers(0, len(idx), size=(B, len(idx)))
        recall_b = rc[rdraws].mean(1)
        entry = {"support": int(len(idx)), "recall": float(rc.mean()),
                 "recall_ci": pct_ci(recall_b)}
        pidx = np.where(pred == c)[0]
        if len(pidx):
            pc = (ytrue[pidx] == c)
            pdraws = rng.integers(0, len(pidx), size=(B, len(pidx)))
            prec_b = pc[pdraws].mean(1)
            entry["precision"] = float(pc.mean()); entry["precision_ci"] = pct_ci(prec_b)
        per_class[str(names[c])] = entry

    return {"dataset": dataset, "split": split_type, "n_test": n_te,
            "n_classes_in_test": len(np.unique(ytrue)),
            "top1": float(correct1.mean()), "top1_ci": pct_ci(top1b),
            "top5": float(correct5.mean()), "top5_ci": pct_ci(top5b),
            "mrr": float(rr.mean()), "mrr_ci": pct_ci(mrrb),
            "bootstrap": "2000 resamples of test phages, percentile intervals",
            "per_class": per_class}


report = [analyze(*cfg) for cfg in CONFIGS]
(RES / "phage_level_bootstrap.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
for r in report:
    print(r["dataset"], r["split"], "top1", round(r["top1"], 4), r["top1_ci"],
          "mrr", round(r["mrr"], 4), r["mrr_ci"])
print("saved phage_level_bootstrap.json")
