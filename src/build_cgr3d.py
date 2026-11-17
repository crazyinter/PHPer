"""Build and cache 3-channel CGR (k=7) for CHERRY, aligned to cherry_processed order."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset import parse_fasta_file, base_id
from cgr3d import encode_cgr3d

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
C = ROOT / "data" / "raw" / "CHERRY-main"
PROC = ROOT / "data" / "processed"
K = 7


def main():
    out = PROC / "cherry_cgr3d.npz"
    if out.exists():
        print("already cached", out)
        return
    ph = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
    accs = list(ph["accs"])
    seq_map = {base_id(h): s for h, s in parse_fasta_file(C / "dataset" / "nucl.fasta")}
    n = 2 ** K
    raw = np.zeros((len(accs), 3, n, n), dtype=np.float32)
    t0 = time.time()
    for i, a in enumerate(accs):
        raw[i] = encode_cgr3d(seq_map[a], K)
        if (i + 1) % 200 == 0:
            print(f"encoded {i+1}/{len(accs)} {time.time()-t0:.0f}s", flush=True)
    # min-max normalization per channel across all samples (PhageCGRNet convention)
    norm = np.zeros_like(raw)
    for c in range(3):
        lo, hi = raw[:, c].min(), raw[:, c].max()
        norm[:, c] = (raw[:, c] - lo) / (hi - lo)
    np.savez_compressed(out, matrices=norm, labels=ph["labels"],
                        host_names=ph["host_names"], accs=ph["accs"])
    print("saved", out, norm.shape, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
