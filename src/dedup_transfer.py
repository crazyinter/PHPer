# -*- coding: utf-8 -*-
"""Cross-dataset de-duplicated transfer (reviewer comment #6).

Trains the full deep model on one resource and evaluates it on the other after
removing phages that have a close cross-dataset neighbor, so the reported
cross-benchmark result does not rely on near-identical sequences.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_bmk import build_model  # noqa
from train import matrix_contrastive_loss  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
EPOCHS = 100
BS = 32
LR = 1e-3
SEEDS = [123]
DEDUP_T = [0.999, 0.99, 0.95]


def load_side(tag):
    if tag == "cherry":
        ph = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
        hf = np.load(PROC / "cherry_host_fcgr.npz", allow_pickle=True)
    else:
        ph = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
        hf = np.load(PROC / "deephost_host_fcgr.npz", allow_pickle=True)
    sp = [str(x).strip().lower() for x in ph["host_names"]]
    hsp = [str(x).strip().lower() for x in hf["host_names"]]
    hpos = {s: i for i, s in enumerate(hsp)}
    return ph, hf, sp, hpos


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def cross_neighbors(srcX, tgtX):
    A = srcX.reshape(len(srcX), -1); B = tgtX.reshape(len(tgtX), -1)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    out = np.empty(len(B))
    for i in range(0, len(B), 256):
        out[i:i + 256] = (B[i:i + 256] @ A.T).max(1)
    return out


def run_direction(src, tgt, seed):
    sp_ph, sp_hf, sp_names, sp_hpos = load_side(src)
    tg_ph, tg_hf, tg_names, tg_hpos = load_side(tgt)
    # shared species with host FCGR on both sides
    common = sorted(set(sp_names) & set(tg_names)
                    & set(s for s in sp_names if s in sp_hpos)
                    & set(s for s in tg_names if s in tg_hpos))
    local = {s: k for k, s in enumerate(common)}

    def build(ph, names, hf, hpos):
        yl = np.array([names[int(t)] for t in ph["labels"]])
        keep = np.array([i for i, s in enumerate(yl) if s in local])
        X = ph["matrices"][keep].astype(np.float32)
        y = np.array([local[s] for s in yl[keep]])
        H = np.stack([hf["matrices"][hpos[s]] for s in common]).astype(np.float32)
        accs = np.array(ph["accs"])[keep]
        return X, y, H, accs

    Xs, ys, Hs, _ = build(sp_ph, sp_names, sp_hf, sp_hpos)
    Xt, yt, Ht, tgt_acc = build(tg_ph, tg_names, tg_hf, tg_hpos)
    nn = cross_neighbors(Xs, Xt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    model = build_model(6).to(device)
    opt = torch.optim.Adam(model.parameters(), LR)
    Xst = torch.tensor(Xs).unsqueeze(1).to(device)
    Hst = torch.tensor(Hs).unsqueeze(1).to(device)
    yst = torch.tensor(ys).to(device)
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(len(Xs))
        for s0 in range(0, len(perm), BS):
            bi = perm[s0:s0 + BS]
            loss = matrix_contrastive_loss(model(Xst[bi]), model(Hst), yst[bi], 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Xtt = torch.tensor(Xt).unsqueeze(1).to(device)
        Htt = torch.tensor(Ht).unsqueeze(1).to(device)
        scores = torch.cdist(model(Xtt), model(Htt)).cpu().numpy()
    order = scores.argsort(1)
    pred = order[:, 0]
    correct = (pred == yt)

    def acc(mask):
        return float(correct[mask].mean()) if mask.sum() else None

    res = {"direction": f"{src} -> {tgt}", "seed": seed,
           "n_candidate_hosts": len(common),
           "n_source_phages": len(Xs), "n_target_phages": len(Xt),
           "top1_all_target": acc(np.ones(len(yt), bool))}
    for t in DEDUP_T:
        m = nn < t
        res[f"top1_dedup_cos_lt_{t}"] = acc(m)
        res[f"n_retained_cos_lt_{t}"] = int(m.sum())
    return res


report = []
for seed in SEEDS:
    report.append(run_direction("cherry", "deephost", seed))
    report.append(run_direction("deephost", "cherry", seed))
(RES / "dedup_transfer.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
