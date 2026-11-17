# -*- coding: utf-8 -*-
"""Verify protein feature row alignment against load_dataset's keep order."""
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"

# reproduce load_dataset keep logic explicitly
ph = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
hf = np.load(PROC / "deephost_host_fcgr.npz", allow_pickle=True)
phage_names = list(ph["host_names"]); host_names = list(hf["host_names"])
shared = sorted(set(phage_names) & set(host_names))
new_label = {h: j for j, h in enumerate(shared)}
keep, labels = [], []
for i in range(len(ph["labels"])):
    orig = phage_names[int(ph["labels"][i])]
    if orig in new_label:
        keep.append(i); labels.append(new_label[orig])
print("total phages:", len(ph["labels"]), "kept:", len(keep))
print("keep is identity 0..N-1:", keep == list(range(len(ph["labels"]))))

pp = np.load(PROC / "deephost_protein_features.npz", allow_pickle=True)
print("protein accs == ph accs:", list(pp["accs"]) == list(ph["accs"]))
P = pp["features"]
print("protein feature rows:", P.shape[0], "kept rows:", len(keep))

# A simple MLP on the aligned (kept) protein data, labels from load_dataset
y = np.array(labels)
rng = np.random.default_rng(123)
idx = rng.permutation(len(y)); n_tr = int(0.8 * len(y))
tr, te = idx[:n_tr], idx[n_tr:]
Pk = P[keep]  # apply keep explicitly
mu, sd = Pk[tr].mean(0), Pk[tr].std(0) + 1e-6
Pk = (Pk - mu) / sd
device = "cuda" if torch.cuda.is_available() else "cpu"
Xt = torch.tensor(Pk).to(device); yt = torch.tensor(y).to(device)
torch.manual_seed(0)
mlp = torch.nn.Sequential(torch.nn.Linear(96, 256), torch.nn.ReLU(),
                          torch.nn.Linear(256, 256), torch.nn.ReLU(),
                          torch.nn.Linear(256, len(shared))).to(device)
opt = torch.optim.Adam(mlp.parameters(), 1e-3)
for _ in range(200):
    loss = torch.nn.functional.cross_entropy(mlp(Xt[tr]), yt[tr])
    opt.zero_grad(); loss.backward(); opt.step()
mlp.eval()
with torch.no_grad():
    acc = (mlp(Xt[te]).argmax(1) == yt[te]).float().mean().item()
print("aligned MLP acc:", round(acc, 4))
