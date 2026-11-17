"""FCGR (frequency chaos game representation), k=6.

Single-pass rolling-index implementation that reproduces the coordinate mapping
and normalization of the released BMKMPHI fasta2CGR.py, including the original
denominator (all k-mer positions, even those containing non-ACGT bases).
"""
import numpy as np

# Coordinate mapping consistent with fasta2CGR:
#   T adds x, C adds y, G adds x and y, A adds neither.
X_BIT = {"A": 0, "C": 0, "T": 1, "G": 1}
Y_BIT = {"A": 0, "C": 1, "T": 0, "G": 1}
_BASES = ("A", "C", "T", "G")


def _base_bits(seq):
    L = len(seq)
    xb = np.zeros(L, dtype=np.int64)
    yb = np.zeros(L, dtype=np.int64)
    valid = np.zeros(L, dtype=bool)
    arr = np.array(list(seq))
    for ch in _BASES:
        m = arr == ch
        xb[m] = X_BIT[ch]
        yb[m] = Y_BIT[ch]
        valid |= m
    return xb, yb, valid


def encode_fcgr(sequence, k=6):
    """Return (2**k, 2**k) FCGR matrix matching the released implementation."""
    n = 2 ** k
    seq = sequence.upper()
    L = len(seq)
    denom = L - k + 1
    matrix = np.zeros((n, n), dtype=np.float32)
    if denom <= 0:
        return matrix

    xb, yb, valid = _base_bits(seq)
    # prefix count of valid bases to test whether each k-mer window is all-ACGT
    cs = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    mask = n - 1

    x_idx = int(sum(int(xb[j]) << (k - 1 - j) for j in range(k)))
    y_idx = int(sum(int(yb[j]) << (k - 1 - j) for j in range(k)))

    for i in range(denom):
        if i > 0:
            x_idx = ((x_idx << 1) & mask) | int(xb[i + k - 1])
            y_idx = ((y_idx << 1) & mask) | int(yb[i + k - 1])
        if cs[i + k] - cs[i] == k:  # all k bases valid
            matrix[y_idx, x_idx] += 1.0

    matrix /= denom  # original normalization denominator
    return matrix.astype(np.float32)
