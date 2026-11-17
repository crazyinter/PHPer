# -*- coding: utf-8 -*-
"""Unified full-host softmax-CE training for FCGR/protein/fusion on DeepHost."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from train import set_seed, random_split, rank_from_scores
from models_ext import ProteinOnlyNet, FusionModel
from splits import get_similarity_split

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"; RES = ROOT / "results"
CKPT = ROOT / "models"; SEEDS = [123, 2024, 42, 7, 99]


def load_data():
    from train import load_dataset
    X, y, H, shared = load_dataset("deephost")
    pp = np.load(PROC / "deephost_protein_features.npz", allow_pickle=True)
    P = pp["features"].astype(np.float32)
    return X, y, P, shared


def build_encoder(variant):
    from model_bmk import build_model
    if variant == "fcgr":
        return build_model(6), "fcgr"
    if variant == "protein":
        return ProteinOnlyNet(96), "protein"
    return FusionModel(build_model(6), 96), "fusion"


def encode(enc, kind, fcgr, prot):
    if kind == "fcgr":
        return enc(fcgr)
    if kind == "protein":
        return enc(prot)
    return enc(fcgr, prot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--split", default="random", choices=["random", "similarity"])
    ap.add_argument("--variant", default="protein",
                    choices=["fcgr", "protein", "fusion"])
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, P, shared = load_data()
    if args.split == "similarity":
        tr, va, te = get_similarity_split("deephost_std", args.seed, 0.05)
    else:
        tr, va, te = random_split(len(X), args.seed)
    mu, sd = P[tr].mean(0), P[tr].std(0) + 1e-6
    P = (P - mu) / sd

    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Pt = torch.tensor(P).to(device); yt = torch.tensor(y).to(device)
    enc, kind = build_encoder(args.variant)
    enc = enc.to(device)
    head = torch.nn.Linear(512, len(shared)).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, 1e-3)
    best_va, best_state = -1, None
    t0 = time.time()
    for ep in range(args.epochs):
        enc.train(); head.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), args.batch_size):
            bi = perm[s:s + args.batch_size]
            logits = head(encode(enc, kind, Xt[bi], Pt[bi]))
            loss = torch.nn.functional.cross_entropy(logits, yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            va1, _, _, _ = rank_from_scores(
                head(encode(enc, kind, Xt[va], Pt[va])), yt[va], False)
        if va1 > best_va:
            best_va = va1
            best_state = ({k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} va {va1:.4f} best {best_va:.4f}", flush=True)
    enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    enc.eval(); head.eval()
    with torch.no_grad():
        top1, top5, mrr, pred = rank_from_scores(
            head(encode(enc, kind, Xt[te], Pt[te])), yt[te], False)
    result = {
        "dataset": "deephost", "objective": "ce", "seed": args.seed,
        "split_type": args.split, "variant": args.variant, "N": len(X),
        "hosts": len(shared), "split": [len(tr), len(va), len(te)],
        "test_top1": top1, "test_top5": top5, "test_mrr": mrr,
        "macro_precision": precision_score(y[te], pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y[te], pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
        "seconds": round(time.time() - t0, 1),
    }
    tag = f"deephost_{args.split}_{args.variant}_ce_seed{args.seed}"
    (RES / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save((enc.state_dict(), head.state_dict()), CKPT / f"{tag}.pt")
    print("RESULT", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
