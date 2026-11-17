"""Efficient 3-channel CGR encoding (position x, position y, k-mer frequency).

Vectorized reimplementation of PhageCGRNet CGR_3D.cgr_3d for k=7.
CGR corner convention (PhageCGRNet):
  A=(0,0) G=(1,0) T=(1,1) C=(0,1), initial point (0.5, 0.5).
"""
import numpy as np
from scipy.signal import lfilter

_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
# corner coordinates
_CX = {"A": 0, "C": 0, "G": 1, "T": 1}
_CY = {"A": 0, "C": 1, "G": 0, "T": 1}


def cgr_coords(seq):
    """CGR point at every base via causal IIR p_t = 0.5 p_{t-1} + 0.5 corner_t."""
    L = len(seq)
    cx = np.array([_CX.get(ch, 0) for ch in seq], dtype=np.float64)
    cy = np.array([_CY.get(ch, 0) for ch in seq], dtype=np.float64)
    valid = np.array([ch in _CODE for ch in seq])
    # invalid bases do not move the point (corner 0 still halves it; to match
    # original behavior we keep the recurrence on every base)
    # prepend a virtual corner of 1 so the first filtered value equals 0.5
    # (the CGR center), then drop it; default zero initial state.
    px = lfilter([0.5], [1.0, -0.5], np.concatenate([[1.0], cx]))[1:]
    py = lfilter([0.5], [1.0, -0.5], np.concatenate([[1.0], cy]))[1:]
    return px, py, valid


def rolling_kmers(seq, k):
    """2-bit rolling k-mer codes in dictionary order (A=0,C=1,G=2,T=3); -1 invalid."""
    L = len(seq)
    code = np.full(L, -1, dtype=np.int64)
    for ch, c in _CODE.items():
        code[np.array([s == ch for s in seq])] = c
    mask = (1 << (2 * k)) - 1
    out = np.full(L - k + 1, -1, dtype=np.int64)
    cur, nvalid = 0, 0
    for t in range(L):
        if code[t] < 0:
            nvalid = 0
            cur = 0
            continue
        cur = ((cur << 2) | code[t]) & mask
        nvalid += 1
        if nvalid >= k:
            out[t - k + 1] = cur
    return out


def encode_cgr3d(seq, k=7):
    n = 2 ** k
    total_types = 4 ** k
    denom = len(seq) - k + 1
    A = np.zeros(total_types, dtype=np.float64)
    B = np.zeros(total_types, dtype=np.float64)
    cnt = np.zeros(total_types, dtype=np.float64)
    if denom <= 0:
        return np.stack([A.reshape(n, n)] * 3).astype(np.float32)
    km = rolling_kmers(seq, k)
    px, py, _ = cgr_coords(seq)
    t = np.arange(k - 1, len(seq))
    valid = km >= 0
    kv, tv = km[valid], t[valid]
    np.add.at(A, kv, px[tv])
    np.add.at(B, kv, py[tv])
    np.add.at(cnt, kv, 1.0)
    pos = cnt > 0
    A[pos] /= cnt[pos]
    B[pos] /= cnt[pos]
    F = cnt  # original CGR_Frequency stores raw counts; min-max normalization later
    return np.stack([A.reshape(n, n), B.reshape(n, n), F.reshape(n, n)]).astype(np.float32)
