# -*- coding: utf-8 -*-
"""Multiple component-split robustness (reviewer #17).

Repeats the similarity-aware component split with several independent seeds,
retraining the full model each time, and reports mean +/- SD instead of a
single split.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models_ext import build_variant  # noqa
from train import (load_dataset, matrix_contrastive_loss, rank_from_scores,
                   recalibrate_bn, set_seed)  # noqa
from splits import get_similarity_split  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
device = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [123, 2024, 42, 7, 99]
EPOCHS = 100
BS = 32


def run_dataset(dataset):
    X, y, H, names = load_dataset(dataset)
    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    yt = torch.tensor(y).to(device)
    records = []
    for seed in SEEDS:
        set_seed(seed)
        split_name = "deephost_std" if dataset == "deephost" else dataset
        tr, va, te = get_similarity_split(split_name, seed, 0.05)
        model = build_variant("cnn_mamba").to(device)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        best_va, best_state = -1, None
        for ep in range(EPOCHS):
            model.train(); perm = np.random.permutation(tr)
            for s0 in range(0, len(perm), BS):
                bi = perm[s0:s0 + BS]
                loss = matrix_contrastive_loss(model(Xt[bi]), model(Ht), yt[bi], 1.0)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va1, _, _, _ = rank_from_scores(
                    torch.cdist(model(Xt[va]), model(Ht)), yt[va], True)
            if va1 > best_va:
                best_va = va1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model = recalibrate_bn(model, Xt, Ht, tr, BS)
        model.eval()
        with torch.no_grad():
            top1, top5, mrr, _ = rank_from_scores(
                torch.cdist(model(Xt[te]), model(Ht)), yt[te], True)
        records.append({"seed": seed, "split": [len(tr), len(va), len(te)],
                        "top1": round(top1, 4), "top5": round(top5, 4),
                        "mrr": round(mrr, 4)})
        print(dataset, "seed", seed, "done", records[-1], flush=True)

    def ms(key):
        v = np.array([r[key] for r in records])
        return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4)}

    return {"dataset": dataset, "mash_threshold": 0.05, "n_splits": len(SEEDS),
            "per_split": records,
            "top1": ms("top1"), "top5": ms("top5"), "mrr": ms("mrr")}


report = [run_dataset("deephost")]
(RES / "component_split_robustness.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
