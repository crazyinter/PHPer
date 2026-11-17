"""Train BMKMPHI variants on cached FCGR datasets.

Dimensions under test:
  --encoder        cnn | cnn_mamba
  --representation fcgr | random (k-mer frequency permutation)
  --loss           contrastive | ce (full-host softmax cross-entropy)
  --split          random | similarity
Logs top-1/top-5/MRR/macro-F1 and saves the best checkpoint.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from models_ext import build_variant, permute_representation
from splits import get_similarity_split

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
CKPT = ROOT / "models"
RES.mkdir(exist_ok=True)
CKPT.mkdir(exist_ok=True)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(name):
    fname = f"{name}_std_processed.npz" if name == "deephost" else f"{name}_processed.npz"
    ph = np.load(PROC / fname, allow_pickle=True)
    host_file = "cherry_host_fcgr.npz" if name == "cherry" else "deephost_host_fcgr.npz"
    hf = np.load(PROC / host_file, allow_pickle=True)
    phage_names = list(ph["host_names"])
    host_names = list(hf["host_names"])
    shared = sorted(set(phage_names) & set(host_names))
    h_idx = {h: i for i, h in enumerate(host_names)}
    new_label = {h: j for j, h in enumerate(shared)}
    keep, labels = [], []
    for i in range(len(ph["labels"])):
        orig = phage_names[int(ph["labels"][i])]
        if orig in new_label:
            keep.append(i)
            labels.append(new_label[orig])
    X = ph["matrices"][keep].astype(np.float32)
    H = np.stack([hf["matrices"][h_idx[h]] for h in shared]).astype(np.float32)
    return X, np.array(labels), H, shared


def random_split(n, seed, ratios=(0.8, 0.1, 0.1)):
    g = np.random.default_rng(seed)
    idx = g.permutation(n)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def matrix_contrastive_loss(emb_ph, emb_host, labels, margin):
    diff = emb_ph.unsqueeze(1) - emb_host.unsqueeze(0)
    dist_sq = (diff ** 2).sum(-1)
    dist = torch.sqrt(dist_sq.clamp_min(1e-12))
    hinge = torch.clamp(margin - dist, min=0.0) ** 2
    pos = torch.zeros_like(dist)
    pos[torch.arange(len(labels)), labels] = 1.0
    return (pos * dist_sq + (1 - pos) * hinge).mean() / 2.0


def rank_from_scores(scores, labels, smaller_is_better=True):
    """Return top1, top5, mrr, hard predictions from per-host scores."""
    order = torch.argsort(scores, dim=1, descending=not smaller_is_better)
    rank = (order == labels.unsqueeze(1)).nonzero()[:, 1]
    top1 = (rank == 0).float().mean().item()
    top5 = (rank < 5).float().mean().item()
    mrr = (1.0 / (rank + 1)).float().mean().item()
    return top1, top5, mrr, order[:, 0].cpu().numpy()


def recalibrate_bn(model, Xt, Ht, tr, batch_size, passes=3):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            order = np.random.permutation(tr)
            for s in range(0, len(order), batch_size):
                model(Xt[order[s:s + batch_size]])
            model(Ht)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cherry", choices=["cherry", "deephost"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--split", default="random", choices=["random", "similarity"])
    ap.add_argument("--mash_threshold", type=float, default=0.05)
    ap.add_argument("--encoder", default="cnn_mamba", choices=["cnn", "cnn_mamba"])
    ap.add_argument("--representation", default="fcgr", choices=["fcgr", "random"])
    ap.add_argument("--loss", default="contrastive", choices=["contrastive", "ce"])
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y, H, host_names = load_dataset(args.dataset)
    if args.representation == "random":
        X = permute_representation(X, args.seed)
        H = permute_representation(H, args.seed)
    if args.split == "similarity":
        tr, va, te = get_similarity_split(args.dataset, args.seed,
                                          threshold=args.mash_threshold)
    else:
        tr, va, te = random_split(len(X), args.seed)
    print(f"{args.dataset} N={len(X)} hosts={len(host_names)} split={len(tr)}/{len(va)}/{len(te)} "
          f"enc={args.encoder} rep={args.representation} loss={args.loss}")

    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    yt = torch.tensor(y).to(device)

    model = build_variant(args.encoder).to(device)
    params = list(model.parameters())
    cls_head = None
    if args.loss == "ce":
        cls_head = torch.nn.Linear(512, len(host_names)).to(device)
        params += list(cls_head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    best_va, best_state = -1.0, None
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        if cls_head is not None:
            cls_head.train()
        perm = np.random.permutation(tr)
        ep_loss = 0.0
        for s in range(0, len(perm), args.batch_size):
            bi = perm[s:s + args.batch_size]
            if args.loss == "contrastive":
                loss = matrix_contrastive_loss(model(Xt[bi]), model(Ht), yt[bi], args.margin)
            else:
                logits = cls_head(model(Xt[bi]))
                loss = torch.nn.functional.cross_entropy(logits, yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        model.eval()
        if cls_head is not None:
            cls_head.eval()
        with torch.no_grad():
            if args.loss == "contrastive":
                scores = torch.cdist(model(Xt[va]), model(Ht))
                va_top1, _, _, _ = rank_from_scores(scores, yt[va], True)
            else:
                va_top1, _, _, _ = rank_from_scores(cls_head(model(Xt[va])), yt[va], False)
        if va_top1 > best_va:
            best_va = va_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} loss {ep_loss:.4f} va {va_top1:.4f} best {best_va:.4f}", flush=True)

    model.load_state_dict(best_state)
    if args.loss == "contrastive":
        model = recalibrate_bn(model, Xt, Ht, tr, args.batch_size)
    model.eval()
    with torch.no_grad():
        if args.loss == "contrastive":
            scores = torch.cdist(model(Xt[te]), model(Ht))
            top1, top5, mrr, pred = rank_from_scores(scores, yt[te], True)
        else:
            cls_head.eval()
            top1, top5, mrr, pred = rank_from_scores(cls_head(model(Xt[te])), yt[te], False)
    yte = y[te]
    result = {
        "dataset": args.dataset, "seed": args.seed, "split_type": args.split,
        "encoder": args.encoder, "representation": args.representation, "loss": args.loss,
        "N": len(X), "hosts": len(host_names), "split": [len(tr), len(va), len(te)],
        "test_top1": top1, "test_top5": top5, "test_mrr": mrr,
        "macro_precision": precision_score(yte, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(yte, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "seconds": round(time.time() - t0, 1),
    }
    tag = (f"{args.dataset}_{args.split}_{args.encoder}_{args.representation}"
           f"_{args.loss}_seed{args.seed}")
    (RES / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(best_state, CKPT / f"{tag}.pt")
    print("RESULT", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
