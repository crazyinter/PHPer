"""Train PhageCGRNet (3-channel CGR + CNN) on identical splits."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from models_ext import PhageCGRCNN
from splits import get_similarity_split
from train import random_split, rank_from_scores

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
CKPT = ROOT / "models"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--split", default="random", choices=["random", "similarity"])
    args = ap.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(PROC / "cherry_cgr3d.npz", allow_pickle=True)
    X = d["matrices"].astype(np.float32)
    y = d["labels"].astype(np.int64)
    n_hosts = len(d["host_names"])
    if args.split == "similarity":
        tr, va, te = get_similarity_split("cherry", args.seed)
    else:
        tr, va, te = random_split(len(X), args.seed)

    Xt = torch.tensor(X).to(device)
    yt = torch.tensor(y).to(device)
    model = PhageCGRCNN(n_hosts).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    best_va, best_state = -1, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), args.batch_size):
            bi = perm[s:s + args.batch_size]
            loss = crit(model(Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_top1, _, _, _ = rank_from_scores(model(Xt[va]), yt[va], False)
        if va_top1 > best_va:
            best_va = va_top1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} va {va_top1:.4f} best {best_va:.4f}", flush=True)

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        top1, top5, mrr, pred = rank_from_scores(model(Xt[te]), yt[te], False)
    result = {
        "dataset": "cherry", "seed": args.seed, "split_type": args.split,
        "encoder": "phagecgrnet", "representation": "cgr3d", "loss": "ce",
        "N": len(X), "hosts": n_hosts, "split": [len(tr), len(va), len(te)],
        "test_top1": top1, "test_top5": top5, "test_mrr": mrr,
        "macro_precision": precision_score(y[te], pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y[te], pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y[te], pred, average="macro", zero_division=0),
        "seconds": round(time.time() - t0, 1),
    }
    tag = f"cherry_{args.split}_phagecgrnet_cgr3d_ce_seed{args.seed}"
    (RES / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(best_state, CKPT / f"{tag}.pt")
    print("RESULT", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
