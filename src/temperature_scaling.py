"""Temperature scaling for the distance-based confidence (Guo et al. 2017)."""
import json
from pathlib import Path

import numpy as np
import torch

from model_bmk import build_model
from train import load_dataset, random_split, recalibrate_bn

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
device = "cuda" if torch.cuda.is_available() else "cpu"


def ece_from_logits(logits, labels, n_bins=10):
    p = torch.softmax(logits, 1)
    conf, pred = p.max(1)
    acc = pred.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += m.sum().item() / len(labels) * abs(acc[m].float().mean() - conf[m].mean())
    return ece.item()


def main(seed=123):
    X, y, H, host_names = load_dataset("cherry")
    tr, va, te = random_split(len(X), seed)
    Xt = torch.tensor(X).unsqueeze(1).to(device)
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    yt = torch.tensor(y).to(device)

    model = build_model(6).to(device)
    model.load_state_dict(torch.load(ROOT / "models" /
                        f"cherry_random_cnn_mamba_fcgr_contrastive_seed{seed}.pt"))
    model = recalibrate_bn(model, Xt, Ht, tr, 32)
    with torch.no_grad():
        va_logits = -torch.cdist(model(Xt[va]), model(Ht))
        te_logits = -torch.cdist(model(Xt[te]), model(Ht))

    temp = torch.nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.LBFGS([temp], lr=0.01, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(va_logits / temp, yt[va])
        loss.backward()
        return loss
    opt.step(closure)

    ece_before = ece_from_logits(te_logits, yt[te])
    ece_after = ece_from_logits(te_logits / temp.detach(), yt[te])
    nll_before = torch.nn.functional.cross_entropy(te_logits, yt[te]).item()
    nll_after = torch.nn.functional.cross_entropy(te_logits / temp.detach(), yt[te]).item()
    out = {"seed": seed, "temperature": temp.item(),
           "ece_before": ece_before, "ece_after": ece_after,
           "nll_before": nll_before, "nll_after": nll_after}
    (RES / f"temperature_scaling_seed{seed}.json").write_text(json.dumps(out, indent=2),
                                                             encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
