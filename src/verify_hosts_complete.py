# -*- coding: utf-8 -*-
"""Verify every class that contains phages has a valid >=200 kb host genome."""
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
D = ROOT / "data" / "fasta" / "deephost_hosts"
d = np.load(ROOT / "data/processed/deephost_std_processed.npz", allow_pickle=True)
present = sorted(set(d["labels"].tolist()))


def bp(i):
    f = D / f"{i:03d}.fasta"
    if not f.exists():
        return 0
    return sum(len(ln.strip()) for ln in f.read_text(encoding="utf-8", errors="replace").splitlines()
               if not ln.startswith(">"))


bad = [i for i in present if bp(i) < 200_000]
print("present classes:", len(present))
print("classes without a valid host genome:", bad)
print("RESULT:", "ALL OK" if not bad else "INCOMPLETE")
