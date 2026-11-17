"""Verify efficient cgr3d against original PhageCGRNet functions."""
import sys
from functools import reduce
from pathlib import Path

import numpy as np

PHG = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\data\raw\PhageCGRNet-main")
sys.path.insert(0, str(PHG))
sys.path.insert(0, str(Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\src")))

import CGR_3D as orig
from cgr3d import encode_cgr3d

K = 7


def orig_raw(seq, k):
    ntypes = 4 ** k
    A = np.zeros((1, ntypes)); B = np.zeros((1, ntypes)); F = np.zeros((1, ntypes))
    cgr = orig.mk_cgr(seq)
    A, B = orig.cgr_position(seq, cgr, k, 0, A, B)
    F = orig.CGR_Frequency(seq, k, 0, F)
    n = 2 ** k
    return np.stack([A.reshape(n, n), B.reshape(n, n), F.reshape(n, n)])


def main():
    rng = np.random.default_rng(0)
    bases = list("ACGT")
    for trial, L in enumerate([50, 200, 1000, 3000]):
        seq = "".join(rng.choice(bases, L))
        o = orig_raw(seq, K)
        m = encode_cgr3d(seq, K)
        for c, nm in enumerate(["Ax", "By", "F"]):
            d = np.abs(o[c] - m[c]).max()
            print(f"L={L} {nm} maxdiff={d:.6e}")
    # real-like sequence from CHERRY
    from Bio import SeqIO
    f = r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\data\raw\CHERRY-main\dataset\nucl.fasta"
    rec = next(SeqIO.parse(f, "fasta"))
    seq = str(rec.seq).upper().replace("U", "T")
    o = orig_raw(seq, K); m = encode_cgr3d(seq, K)
    for c, nm in enumerate(["Ax", "By", "F"]):
        print(f"real len={len(seq)} {nm} maxdiff={np.abs(o[c]-m[c]).max():.6e}")


if __name__ == "__main__":
    main()
