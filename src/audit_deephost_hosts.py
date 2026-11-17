# -*- coding: utf-8 -*-
"""Audit actual sequence lengths of downloaded DeepHost host genomes."""
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
D = ROOT / "data" / "fasta" / "deephost_hosts"

std = {}
for line in (ROOT / "data/raw/DeepHost-master/data/species.txt").read_text().strip().splitlines():
    i, name = line.split("\t")
    std[int(i)] = name.strip()

def fasta_len(f):
    n = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.startswith(">"):
            n += len(line.strip())
    return n

files = sorted(D.glob("*.fasta"))
print("fasta files:", len(files))
short = []
for f in files:
    i = int(f.stem)
    ln = fasta_len(f)
    if ln < 200_000:
        short.append((i, std.get(i, "?"), ln))
print("\nshort/abnormal (<200 kb):")
for i, name, ln in short:
    print(f"  {i:3d} {name}: {ln}")
missing = [i for i in range(118) if not (D / f"{i:03d}.fasta").exists()]
print("\nmissing files:", [(i, std[i]) for i in missing])
