# -*- coding: utf-8 -*-
"""Paired phage bootstrap CIs for open-set AUROC and FPR@95TPR."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model_bmk import build_model
from train import load_dataset, random_split, recalibrate_bn

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; CKPT = ROOT / "models"
device = "cuda" if torch.cuda.is_available() else "cpu"
TEMP = 0.1459
B = 2000
rng = np.random.default_rng(2024)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fpr_at_tpr(score, is_id, target=0.95):
    order = np.sort(score)[::-1]; n_id = is_id.sum()
    for thr in order:
        tpr = ((score >= thr) & is_id).sum() / n_id
        if tpr <= target:
            return ((score >= thr) & ~is_id).sum() / (~is_id).sum()
    return 1.0


X, y, H, names = load_dataset("cherry")
Xt = torch.tensor(X).unsqueeze(1).to(device)
Ht = torch.tensor(H).unsqueeze(1).to(device)
tr, va, te = random_split(len(X), 123)
model = build_model(6).to(device)
model.load_state_dict(torch.load(CKPT / "cherry_random_cnn_mamba_fcgr_contrastive_seed123.pt"))
model = recalibrate_bn(model, Xt, Ht, tr, 32); model.eval()
with torch.no_grad():
    emb = model(Xt[te]); host_emb = model(Ht)
    dist = torch.cdist(emb, host_emb)
logits = (-dist / TEMP).cpu().numpy(); ytrue = y[te]; n = len(te)

id_conf = np.max(softmax(logits), 1)
ood_logits = logits.copy()
for i, c in enumerate(ytrue):
    ood_logits[i, c] = -1e9
ood_conf = np.max(softmax(ood_logits), 1)
id_dist = (-dist.min(1).values).cpu().numpy()
ood_dist = np.array([-dist[i, np.delete(np.arange(len(names)), ytrue[i])].min().item()
                    for i in range(n)])

# per-phage paired arrays: each phage has one ID and one OOD score
ph_sm = np.stack([id_conf, ood_conf], 1)       # (n,2) col0=ID col1=OOD
ph_ds = np.stack([id_dist, ood_dist], 1)


def metrics(ph):
    idv = ph[:, 0]; oov = ph[:, 1]
    score = np.concatenate([idv, oov])
    is_id = np.concatenate([np.ones(len(idv), bool), np.zeros(len(oov), bool)])
    return roc_auc_score(is_id, score), fpr_at_tpr(score, is_id)


def boot(ph):
    au = []; fp = []
    for _ in range(B):
        ix = rng.integers(0, len(ph), len(ph))
        a, f = metrics(ph[ix]); au.append(a); fp.append(f)
    return au, fp


out = {"n_test_phages": n, "bootstrap_resamples": B,
       "temperature_used": TEMP, "temperature_source": "multi-seed validation-fit mean, fixed before test"}
for key, ph in [("calibrated_softmax", ph_sm), ("distance_score", ph_ds)]:
    pt_au, pt_fp = metrics(ph)
    au, fp = boot(ph)
    au = np.array(au); fp = np.array(fp)
    out[key] = {
        "auroc_point": round(pt_au, 4),
        "auroc_95ci": [round(float(np.percentile(au, 2.5)), 4), round(float(np.percentile(au, 97.5)), 4)],
        "fpr95_point": round(pt_fp, 4),
        "fpr95_95ci": [round(float(np.percentile(fp, 2.5)), 4), round(float(np.percentile(fp, 97.5)), 4)],
    }
(RES / "open_set_bootstrap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
