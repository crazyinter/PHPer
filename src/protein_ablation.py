# -*- coding: utf-8 -*-
"""Ablation of protein-feature sub-blocks."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train import set_seed, random_split, matrix_contrastive_loss, rank_from_scores
from splits import get_similarity_split
from train_fusion import load_fusion_data

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
SEEDS = [123, 42, 2024]
BLOCKS = {
    "structure": (0, 8),
    "codon": (8, 72),
    "aa_phys": (72, 96),
    "codon_aa": (8, 96),
}


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 512))

    def forward(self, x):
        return self.net(x)


def main():
    X, y, H, P, HP, shared = load_fusion_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for split in ("random", "similarity"):
        out[split] = {}
        for name, (a, b) in BLOCKS.items():
            vals = []
            for seed in SEEDS:
                set_seed(seed)
                if split == "similarity":
                    tr, va, te = get_similarity_split("cherry", seed, 0.05)
                else:
                    tr, va, te = random_split(len(X), seed)
                Ps = P[:, a:b]; HPs = HP[:, a:b]
                mu, sd = Ps[tr].mean(0), Ps[tr].std(0) + 1e-6
                Ps = (Ps - mu) / sd; HPs = (HPs - HPs.mean(0)) / (HPs.std(0) + 1e-6)
                Pt = torch.tensor(Ps).to(device); HPt = torch.tensor(HPs).to(device)
                yt = torch.tensor(y).to(device)
                model = Net(b - a).to(device)
                opt = torch.optim.Adam(model.parameters(), 1e-3)
                best_va, best_state = -1, None
                for ep in range(100):
                    model.train()
                    perm = np.random.permutation(tr)
                    for s in range(0, len(perm), 32):
                        bi = perm[s:s + 32]
                        loss = matrix_contrastive_loss(model(Pt[bi]), model(HPt), yt[bi], 1.0)
                        opt.zero_grad(); loss.backward(); opt.step()
                    model.eval()
                    with torch.no_grad():
                        v, _, _, _ = rank_from_scores(torch.cdist(model(Pt[va]), model(HPt)), yt[va], True)
                    if v > best_va:
                        best_va = v; best_state = {k: q.detach().cpu().clone() for k, q in model.state_dict().items()}
                model.load_state_dict(best_state)
                with torch.no_grad():
                    top1, _, _, _ = rank_from_scores(torch.cdist(model(Pt[te]), model(HPt)), yt[te], True)
                vals.append(top1)
            out[split][name] = {"mean": float(np.mean(vals)), "values": vals}
            print(f"[{split:10s}] {name:9s} top1={np.mean(vals):.4f}", flush=True)
    (RES / "protein_ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
