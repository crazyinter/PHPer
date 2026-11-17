# -*- coding: utf-8 -*-
"""Build 96-D protein features for DeepHost phages and downloaded host genomes."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from protein_features import extract_features, _finder
from dataset import build_labeled, base_id

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
D = ROOT / "data" / "fasta" / "deephost_hosts"

std = {}
for line in (ROOT / "data/raw/DeepHost-master/data/species.txt").read_text().strip().splitlines():
    i, name = line.split("\t")
    std[int(i)] = name.strip()


def read_host_seq(i):
    return "".join(ln.strip() for ln in (D / f"{i:03d}.fasta").read_text(
        encoding="utf-8", errors="replace").splitlines() if not ln.startswith(">"))


def build_phage_features(finder):
    ph = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
    accs = [str(a) for a in ph["accs"]]
    labeled, _ = build_labeled()
    amap = {base_id(r["acc"]): r["seq"] for r in labeled}
    missing = [a for a in accs if a not in amap]
    if missing:
        print("WARNING missing sequences for", len(missing), "accs:", missing[:5])
    feats = np.stack([extract_features(amap[a], finder) for a in accs])
    np.savez_compressed(PROC / "deephost_protein_features.npz",
                        features=feats.astype(np.float32), accs=np.array(accs))
    print("phage protein features:", feats.shape)


def build_host_features(finder):
    present = sorted(set(np.load(PROC / "deephost_std_processed.npz",
                                allow_pickle=True)["labels"].tolist()))
    feats, names = [], []
    for i in present:
        feats.append(extract_features(read_host_seq(i), finder))
        names.append(std[i])
    hpf = np.stack(feats).astype(np.float32)
    np.savez_compressed(PROC / "deephost_host_protein_features.npz",
                        features=hpf, host_names=np.array(names))
    print("host protein features:", hpf.shape)


if __name__ == "__main__":
    finder = _finder()
    build_phage_features(finder)
    build_host_features(finder)
