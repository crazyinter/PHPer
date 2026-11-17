# -*- coding: utf-8 -*-
"""Multi-seed confidence calibration on CHERRY full model (random + similarity)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_bmk import build_model  # noqa
from train import load_dataset, random_split, recalibrate_bn  # noqa
from splits import (build_sketches, mash_distance_matrix,  # noqa
                    connected_components, group_split)

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; CKPT = ROOT / "models"
SEEDS = [123, 2024, 42, 7, 99]
device = "cuda" if torch.cuda.is_available() else "cpu"


def ece(logits, labels, n_bins=10):
    p = torch.softmax(logits, 1); conf, pred = p.max(1); acc = pred.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1); out = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            out += m.sum().item() / len(labels) * abs(
                acc[m].float().mean() - conf[m].mean()).item()
    return float(out)


X, y, H, names = load_dataset("cherry")
Xt = torch.tensor(X).unsqueeze(1).to(device)
Ht = torch.tensor(H).unsqueeze(1).to(device)
yt = torch.tensor(y).to(device)
comp, _ = connected_components(mash_distance_matrix(build_sketches("cherry")), 0.05)

records = []
for split_type in ("random", "similarity"):
    for seed in SEEDS:
        if split_type == "random":
            tr, va, te = random_split(len(X), seed)
        else:
            tr, va, te = group_split(comp, len(X), seed)
        model = build_model(6).to(device)
        model.load_state_dict(torch.load(
            CKPT / f"cherry_{split_type}_cnn_mamba_fcgr_contrastive_seed{seed}.pt"))
        model = recalibrate_bn(model, Xt, Ht, tr, 32)
        with torch.no_grad():
            va_l = -torch.cdist(model(Xt[va]), model(Ht))
            te_l = -torch.cdist(model(Xt[te]), model(Ht))
        temp = torch.nn.Parameter(torch.ones(1, device=device))
        opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=200)

        def closure():
            opt.zero_grad(); loss = torch.nn.functional.cross_entropy(va_l / temp, yt[va])
            loss.backward(); return loss
        opt.step(closure)
        rec = {"split": split_type, "seed": seed, "temperature": round(temp.item(), 4),
               "ece_before": round(ece(te_l, yt[te]), 4),
               "ece_after": round(ece(te_l / temp.detach(), yt[te]), 4),
               "nll_before": round(torch.nn.functional.cross_entropy(te_l, yt[te]).item(), 4),
               "nll_after": round(torch.nn.functional.cross_entropy(
                   te_l / temp.detach(), yt[te]).item(), 4)}
        records.append(rec); print(rec, flush=True)


def summarize(rows, key):
    vals = np.array([r[key] for r in rows])
    boots = [np.random.default_rng(i).choice(vals, len(vals), True).mean() for i in range(2000)]
    return {"mean": round(vals.mean(), 4), "sd": round(vals.std(), 4),
            "bootstrap_ci95": [round(np.percentile(boots, 2.5), 4),
                               round(np.percentile(boots, 97.5), 4)]}


summary = {st: {k: summarize([r for r in records if r["split"] == st], k)
                for k in ("temperature", "ece_before", "ece_after", "nll_before", "nll_after")}
           for st in ("random", "similarity")}
out = {"n_bins": 10, "softmax_axis": "candidate hosts", "ece_weighting": "sample-weighted",
       "records": records, "summary": summary}
(RES / "multi_seed_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
