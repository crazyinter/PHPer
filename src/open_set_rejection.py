# -*- coding: utf-8 -*-
"""Open-set rejection: withheld-true-host detection via calibrated confidence and distance."""
import json
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
TEMP = 0.1459  # multi-seed mean temperature


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fpr_at_tpr(score, is_id, target=0.95):
    # higher score = more in-distribution; sweep thresholds to hit TPR on ID samples
    order = np.sort(score)[::-1]; n_id = is_id.sum()
    for thr in order:
        tpr = ((score >= thr) & is_id).sum() / n_id
        if tpr <= target:
            fpr = ((score >= thr) & ~is_id).sum() / (~is_id).sum()
            return float(fpr)
    return 1.0


X, y, H, names = load_dataset("cherry")
Xt = torch.tensor(X).unsqueeze(1).to(device)
Ht = torch.tensor(H).unsqueeze(1).to(device)
yt = torch.tensor(y).to(device)
tr, va, te = random_split(len(X), 123)
model = build_model(6).to(device)
model.load_state_dict(torch.load(
    CKPT / "cherry_random_cnn_mamba_fcgr_contrastive_seed123.pt"))
model = recalibrate_bn(model, Xt, Ht, tr, 32)
model.eval()

with torch.no_grad():
    emb = model(Xt[te]); host_emb = model(Ht)
    dist = torch.cdist(emb, host_emb)            # smaller = more likely
logits = (-dist / TEMP).cpu().numpy()
ytrue = y[te]

# ID: full candidate set. OOD: remove the true host column, then renormalize softmax
id_conf = np.max(_softmax(logits), axis=1)
ood_logits = logits.copy()
for i, c in enumerate(ytrue):
    ood_logits[i, c] = -1e9
ood_conf = np.max(_softmax(ood_logits), axis=1)
# distance-based score: negative minimum distance
id_dist = (-dist.min(1).values).cpu().numpy()
ood_dist = np.array([-dist[i, np.delete(np.arange(len(names)), ytrue[i])].min().item()
                    for i in range(len(te))])

scores = np.concatenate([id_conf, ood_conf])
is_id = np.concatenate([np.ones_like(id_conf, bool), np.zeros_like(ood_conf, bool)])
dscores = np.concatenate([id_dist, ood_dist])

# coverage-accuracy: at confidence thresholds, fraction accepted and accuracy among accepted
probs = _softmax(logits); pred = probs.argmax(1); correct = pred == ytrue
cov_acc = []
for q in (0.5, 0.7, 0.9):
    keep = id_conf >= q
    cov_acc.append({"threshold": q, "coverage": round(keep.mean(), 4),
                    "accuracy_when_accepted": round(correct[keep].mean(), 4) if keep.any() else None})

out = {"setting": "true host withheld from candidate set", "n_test": len(te),
       "calibrated_softmax": {"auroc": round(roc_auc_score(is_id, scores), 4),
                              "fpr_at_95tpr": round(fpr_at_tpr(scores, is_id), 4)},
       "distance_score": {"auroc": round(roc_auc_score(is_id, dscores), 4),
                          "fpr_at_95tpr": round(fpr_at_tpr(dscores, is_id), 4)},
       "coverage_accuracy": cov_acc}
(RES / "open_set_rejection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
