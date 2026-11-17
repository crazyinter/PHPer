# -*- coding: utf-8 -*-
"""Train FCGR / protein / fusion models on DeepHost with the identical protocol."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from models_ext import ProteinOnlyNet, FusionModel
from train import (set_seed, load_dataset, random_split, matrix_contrastive_loss,
                   rank_from_scores)
from splits import get_similarity_split

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
CKPT = ROOT / "models"


def load_deephost_fusion():
    X, y, H, shared = load_dataset("deephost")
    ph = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
    pp = np.load(PROC / "deephost_protein_features.npz", allow_pickle=True)
    assert list(pp["accs"]) == list(ph["accs"])
    P = pp["features"].astype(np.float32)  # same row order as load_dataset keep (all kept)
    hpp = np.load(PROC / "deephost_host_protein_features.npz", allow_pickle=True)
    hmap = {n: i for i, n in enumerate(hpp["host_names"])}
    HP = np.stack([hpp["features"][hmap[n]] for n in shared]).astype(np.float32)
    return X, y, H, P, HP, shared


def build_model(variant):
    from model_bmk import build_model
    if variant == "fcgr":
        return build_model(6)
    if variant == "protein":
        return ProteinOnlyNet(96)
    return FusionModel(build_model(6), 96)


def embed(model, variant, fcgr, prot):
    if variant == "fcgr":
        return model(fcgr)
    if variant == "protein":
        return model(prot)
    return model(fcgr, prot)


def recalibrate(model, variant, Xt, Pt, Ht, HPt, tr, bs, passes=3):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            order = np.random.permutation(tr)
            for s in range(0, len(order), bs):
                bi = order[s:s + bs]
                embed(model, variant, Xt[bi], Pt[bi])
            embed(model, variant, Ht, HPt)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split", default="random", choices=["random", "similarity"])
    ap.add_argument("--variant", default="fusion",
                    choices=["fcgr", "protein", "fusion"])
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, H, P, HP, shared = load_deephost_fusion()
    if args.split == "similarity":
        tr, va, te = get_similarity_split("deephost_std", args.seed, threshold=0.05)
    else:
        tr, va, te = random_split(len(X), args.seed)

    mu, sd = P[tr].mean(0), P[tr].std(0) + 1e-6
    P = (P - mu) / sd
    hmu, hsd = HP.mean(0), HP.std(0) + 1e-6
    HP = (HP - hmu) / hsd

    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    Pt = torch.tensor(P).to(device)
    HPt = torch.tensor(HP).to(device)
    yt = torch.tensor(y).to(device)

    model = build_model(args.variant).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    best_va, best_state = -1, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), args.batch_size):
            bi = perm[s:s + args.batch_size]
            ep_ph = embed(model, args.variant, Xt[bi], Pt[bi])
            ep_host = embed(model, args.variant, Ht, HPt)
            loss = matrix_contrastive_loss(ep_ph, ep_host, yt[bi], 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_ph = embed(model, args.variant, Xt[va], Pt[va])
            va_host = embed(model, args.variant, Ht, HPt)
            va_top1, _, _, _ = rank_from_scores(torch.cdist(va_ph, va_host), yt[va], True)
        if va_top1 > best_va:
            best_va = va_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} va {va_top1:.4f} best {best_va:.4f}", flush=True)

    model.load_state_dict(best_state)
    model = recalibrate(model, args.variant, Xt, Pt, Ht, HPt, tr, args.batch_size)
    with torch.no_grad():
        te_ph = embed(model, args.variant, Xt[te], Pt[te])
        te_host = embed(model, args.variant, Ht, HPt)
        top1, top5, mrr, pred = rank_from_scores(torch.cdist(te_ph, te_host), yt[te], True)
    yte = y[te]
    result = {
        "dataset": "deephost", "seed": args.seed, "split_type": args.split,
        "variant": args.variant, "N": len(X), "hosts": len(shared),
        "split": [len(tr), len(va), len(te)],
        "test_top1": top1, "test_top5": top5, "test_mrr": mrr,
        "macro_precision": precision_score(yte, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(yte, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "seconds": round(time.time() - t0, 1),
    }
    tag = f"deephost_{args.split}_{args.variant}_seed{args.seed}"
    (RES / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(best_state, CKPT / f"{tag}.pt")
    print("RESULT", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
