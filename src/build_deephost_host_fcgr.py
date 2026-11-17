# -*- coding: utf-8 -*-
"""Build FCGR matrices for the downloaded DeepHost host genomes."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fcgr import encode_fcgr

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
D = ROOT / "data" / "fasta" / "deephost_hosts"
OUT = PROC / "deephost_host_fcgr.npz"

std = {}
for line in (ROOT / "data/raw/DeepHost-master/data/species.txt").read_text().strip().splitlines():
    i, name = line.split("\t")
    std[int(i)] = name.strip()


def read_genome(i):
    parts = []
    for line in (D / f"{i:03d}.fasta").read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(">"):
            parts.append(line.strip())
    return "".join(parts)


def main():
    present = set(np.load(PROC / "deephost_std_processed.npz",
                         allow_pickle=True)["labels"].tolist())
    ids = sorted(i for i in present if (D / f"{i:03d}.fasta").exists())
    matrices, names = [], []
    t_start = time.time()
    for k, i in enumerate(ids):
        seq = read_genome(i)
        t0 = time.time()
        m = encode_fcgr(seq)
        matrices.append(m)
        names.append(std[i])
        if k < 3:
            print(f"class {i} {std[i]}: {len(seq)} bp encoded in {time.time()-t0:.1f}s", flush=True)
    np.savez_compressed(OUT, matrices=np.stack(matrices).astype(np.float32),
                        host_names=np.array(names))
    print("saved", OUT, "hosts:", len(names), "total time:", round(time.time() - t_start, 1))


if __name__ == "__main__":
    main()
