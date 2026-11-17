# -*- coding: utf-8 -*-
"""Analyze mapping of raw DeepHost species names to the 118 standard classes."""
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
# standard 118 classes
std = {}
for line in (ROOT / "data/raw/DeepHost-master/data/species.txt").read_text().strip().splitlines():
    i, name = line.split("\t")
    std[name.strip()] = int(i)

d = np.load(ROOT / "data/processed/deephost_processed.npz", allow_pickle=True)
names = [str(x) for x in d["host_names"]]
labels = d["labels"]

exact = [n for n in names if n in std]
notexact = [n for n in names if n not in std]
print("raw names:", len(names), "exact match to 118:", len(exact), "not exact:", len(notexact))
print("\nnot-exact names:")
for n in notexact:
    support = int((labels == names.index(n)).sum())
    print(f"  '{n}' (support {support})")

# phage counts
exact_idx = set(i for i, n in enumerate(names) if n in std)
n_exact_phage = int(sum((labels == i).sum() for i in exact_idx))
print("\nphages in exact-matched classes:", n_exact_phage, "/", len(labels))
print("standard classes covered:", len(exact))
