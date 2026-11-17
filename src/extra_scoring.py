# -*- coding: utf-8 -*-
"""Extra scoring (reviewer #9, #22, #23): Brier, reliability, conformal, energy,
and DeepHost frequency baselines."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model_bmk import build_model  # noqa
from train import load_dataset, random_split, recalibrate_bn  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; CKPT = ROOT / "models"
device = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 0.1459
NBIN = 10
ALPHA = 0.1

CFGS = [
    ("cherry", "cherry_random_cnn_mamba_fcgr_contrastive_seed123.pt"),
    ("deephost", "deephost_random_fcgr_seed123.pt"),
]


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


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
        d_te = torch.cdist(model(Xt[te]), model(Ht)).cpu().numpy()
        d_va = torch.cdist(model(Xt[va]), model(Ht)).cpu().numpy()
    yte, yva = y[te], y[va]
    probs = softmax(-d_te / TEMP)
    p_va = softmax(-d_va / TEMP)

    # Brier (multiclass) and a raw, uncalibrated comparison
    onehot = np.zeros_like(probs); onehot[np.arange(len(yte)), yte] = 1
    brier_cal = float(np.mean(np.sum((probs - onehot) ** 2, 1)))
    raw = softmax(-d_te)  # T=1 reference
    brier_raw = float(np.mean(np.sum((raw - onehot) ** 2, 1)))

    # reliability diagram (10 bins, sample-weighted)
    conf = probs.max(1); pred = probs.argmax(1); correct = pred == yte
    bins = []
    for b in range(NBIN):
        m = (conf >= b / NBIN) & (conf < (b + 1) / NBIN)
        if b == NBIN - 1:
            m = (conf >= b / NBIN)
        bins.append({"bin": b + 1, "n": int(m.sum()),
                     "mean_confidence": round(float(conf[m].mean()), 4) if m.sum() else None,
                     "accuracy": round(float(correct[m].mean()), 4) if m.sum() else None})

    # split conformal prediction sets (finite-sample quantile)
    cal_score = 1 - p_va[np.arange(len(yva)), yva]
    ql = math.ceil((len(yva) + 1) * (1 - ALPHA)) / len(yva)
    q = float(np.quantile(cal_score, min(ql, 1.0), method="higher"))
    nc = 1 - probs
    sets = nc <= q
    coverage = float(np.mean(sets[np.arange(len(yte)), yte]))
    mean_size = float(sets.sum(1).mean())

    # energy OOD detection under the withheld-host setup
    logits = -d_te / TEMP
    energy_id = -TEMP * np.log(np.exp(logits / TEMP).sum(1))
    ood_log = logits.copy()
    for i, c in enumerate(yte):
        ood_log[i, c] = -1e9
    energy_ood = -TEMP * np.log(np.exp(ood_log / TEMP).sum(1))
    # lower energy = more in-distribution; negate for higher=ID
    e_score = np.concatenate([-energy_id, -energy_ood])
    is_id = np.concatenate([np.ones(len(yte), bool), np.zeros(len(yte), bool)])
    energy_auroc = float(roc_auc_score(is_id, e_score))

    # frequency baselines (test composition)
    _, counts = np.unique(yte, return_counts=True)
    n_cls = probs.shape[1]
    baselines = {"uniform_random": round(1 / n_cls, 4),
                 "majority_class": round(counts.max() / len(yte), 4),
                 "candidate_host_classes": int(n_cls)}

    return {"dataset": dataset, "n_test": len(te), "n_calibration": len(va),
            "brier_raw": round(brier_raw, 4), "brier_calibrated": round(brier_cal, 4),
            "reliability_diagram": bins,
            "conformal": {"alpha": ALPHA, "quantile": round(q, 4),
                          "empirical_coverage": round(coverage, 4),
                          "mean_prediction_set_size": round(mean_size, 3)},
            "energy_ood_auroc": round(energy_auroc, 4),
            "frequency_baselines": baselines}


report = [analyze(*c) for c in CFGS]
(RES / "extra_scoring.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
for r in report:
    print(r["dataset"], "brier", r["brier_raw"], "->", r["brier_calibrated"],
          "conformal cov", r["conformal"]["empirical_coverage"],
          "size", r["conformal"]["mean_prediction_set_size"],
          "energy AUROC", r["energy_ood_auroc"], "baselines", r["frequency_baselines"])
print("saved extra_scoring.json")
