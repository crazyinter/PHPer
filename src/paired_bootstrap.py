# -*- coding: utf-8 -*-
"""Phage-level paired bootstrap and permutation for architecture contrasts (sample-level, not seed-level)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_bmk import build_model
from models_ext import permute_representation, build_variant
from train import load_dataset, random_split, recalibrate_bn
from splits import get_similarity_split

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"; CKPT = ROOT / "models"
device = "cuda" if torch.cuda.is_available() else "cpu"
B = 2000; SEED = 123

# (label, split, ckpt A, encoder A, ckpt B, encoder B, permute B inputs)
CONTRASTS = [
    ("Full vs CNN-only", "random",
     "cherry_random_cnn_mamba_fcgr_contrastive_seed123.pt", "cnn_mamba",
     "cherry_random_cnn_fcgr_contrastive_seed123.pt", "cnn", False),
    ("Full vs CNN-only", "similarity",
     "cherry_similarity_cnn_mamba_fcgr_contrastive_seed123.pt", "cnn_mamba",
     "cherry_similarity_cnn_fcgr_contrastive_seed123.pt", "cnn", False),
    ("FCGR vs permuted", "random",
     "cherry_random_cnn_mamba_fcgr_contrastive_seed123.pt", "cnn_mamba",
     "cherry_random_cnn_mamba_random_contrastive_seed123.pt", "cnn_mamba", True),
    ("FCGR vs permuted", "similarity",
     "cherry_similarity_cnn_mamba_fcgr_contrastive_seed123.pt", "cnn_mamba",
     "cherry_similarity_cnn_mamba_random_contrastive_seed123.pt", "cnn_mamba", True),
]


def correct_vector(ckpt, encoder, X, H, y, tr, te, permute):
    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    if permute:
        Xp = permute_representation(X, SEED)
        Hp = permute_representation(H, SEED)
        Xt = torch.tensor(Xp).unsqueeze(1).to(device)
        Ht = torch.tensor(Hp).unsqueeze(1).to(device)
    model = build_variant(encoder).to(device)
    model.load_state_dict(torch.load(CKPT / ckpt, map_location=device))
    model = recalibrate_bn(model, Xt, Ht, tr, 32)
    model.eval()
    with torch.no_grad():
        scores = torch.cdist(model(Xt[te]), model(Ht)).cpu().numpy()
    order = scores.argsort(1)
    rank = np.array([np.where(o == y[te][i])[0][0] for i, o in enumerate(order)])
    return (rank == 0)


def pct_ci(v):
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


out = []
rng = np.random.default_rng(SEED)
for label, split, cka, ena, ckb, enb, permb in CONTRASTS:
    X, y, H, names = load_dataset("cherry")
    if split == "random":
        tr, va, te = random_split(len(X), SEED)
    else:
        tr, va, te = get_similarity_split("cherry", SEED, 0.05)
    ca = correct_vector(cka, ena, X, H, y, tr, te, False)
    cb = correct_vector(ckb, enb, X, H, y, tr, te, permb)
    n = len(te)
    obs = float(ca.mean() - cb.mean())

    draws = rng.integers(0, n, size=(B, n))
    diff_boot = (ca[draws].mean(1) - cb[draws].mean(1))

    # paired sign-flip permutation: swap A/B correctness within each phage
    flips = rng.random((B, n)) < 0.5
    da = np.where(flips, cb, ca)
    db = np.where(flips, ca, cb)
    perm_diff = da.mean(1) - db.mean(1)
    p_perm = (np.sum(np.abs(perm_diff) >= abs(obs) - 1e-12) + 1) / (B + 1)

    out.append({
        "contrast": label, "split": split, "n_test": n,
        "top1_A": round(float(ca.mean()), 4), "top1_B": round(float(cb.mean()), 4),
        "observed_delta": round(obs, 4),
        "paired_bootstrap_delta_ci": pct_ci(diff_boot),
        "permutation_p": round(float(p_perm), 4),
        "interpretation": "sample-level paired inference over test phages; not a seed-level t-test",
    })
    print(label, split, "delta", round(obs, 4), "CI", pct_ci(diff_boot), "perm p", round(float(p_perm), 4))

(RES / "paired_bootstrap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("saved paired_bootstrap.json")
