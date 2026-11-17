"""Similarity-aware dataset splitting via MinHash / Mash distance.

Bottom-s MinHash sketches (k=21, s=1000, matching Mash defaults) are built for
each phage. Pairwise Jaccard is converted to Mash distance; phages closer than
a threshold are joined into connected components, and whole components are
assigned to train/val/test so no near-identical phage crosses the split.
"""
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
K = 21
SKETCH = 1000
MASK64 = (1 << 64) - 1

_BIT = {"A": 0, "C": 1, "T": 2, "G": 3}


def splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(MASK64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(MASK64)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(MASK64)
    return x ^ (x >> np.uint64(31))


def kmer_codes(seq, k=K):
    s = seq.upper()
    codes, valid_run = [], []
    cur = 0
    mask = (1 << (2 * k)) - 1
    n_valid = 0
    out = []
    for ch in s:
        if ch not in _BIT:
            n_valid = 0
            continue
        cur = ((cur << 2) | _BIT[ch]) & mask
        n_valid += 1
        if n_valid >= k:
            out.append(cur)
    return np.array(out, dtype=np.uint64)


def minhash_sketch(seq, k=K, s=SKETCH):
    codes = kmer_codes(seq, k)
    if len(codes) == 0:
        return np.empty(0, dtype=np.uint64)
    h = splitmix64(codes)
    if len(h) > s:
        h = np.partition(h, s - 1)[:s]
    return np.sort(h)


def build_sketches(name):
    cache = PROC / f"{name}_sketches.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return list(d["sketches"])
    ph = np.load(PROC / f"{name}_processed.npz", allow_pickle=True)
    # reconstruct sequences from accs is not cached; re-encode from raw is costly,
    # so we instead derive k-mers using the stored FCGR? No — reload raw sequences.
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import build_labeled, base_id
    if name == "cherry":
        from dataset import parse_fasta_file
        C = ROOT / "data" / "raw" / "CHERRY-main"
        seq_map = {base_id(h): s for h, s in parse_fasta_file(C / "dataset" / "nucl.fasta")}
        accs = list(ph["accs"])
        seqs = [seq_map[a] for a in accs]
    else:
        labeled, _ = build_labeled()
        amap = {base_id(r["acc"]): r["seq"] for r in labeled}
        seqs = [amap[a] for a in ph["accs"]]

    sketches = [minhash_sketch(s) for s in seqs]
    np.savez_compressed(cache, sketches=np.array(sketches, dtype=object))
    return sketches


def mash_distance_matrix(sketches, k=K):
    n = len(sketches)
    # global hash id dictionary
    hash_to_id = {}
    rows, cols = [], []
    sizes = np.array([len(sk) for sk in sketches])
    for i, sk in enumerate(sketches):
        for h in sk:
            hv = int(h)
            if hv not in hash_to_id:
                hash_to_id[hv] = len(hash_to_id)
            rows.append(i)
            cols.append(hash_to_id[hv])
    S = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(hash_to_id)))
    inter = (S @ S.T).toarray()
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    with np.errstate(divide="ignore"):
        arg = 2 * jaccard / np.maximum(1 + jaccard, 1e-12)
        d = np.where(jaccard > 0, -np.log(np.clip(arg, 1e-12, 1.0)) / k, 1.0)
    np.fill_diagonal(d, 0.0)
    return d


def connected_components(dist, threshold):
    n = dist.shape[0]
    adj = dist < threshold
    seen = np.zeros(n, dtype=bool)
    comp = -np.ones(n, dtype=np.int64)
    cid = 0
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        while stack:
            v = stack.pop()
            comp[v] = cid
            for w in np.where(adj[v] & ~seen)[0]:
                seen[w] = True
                stack.append(w)
        cid += 1
    return comp, cid


def group_split(comp, n, seed, ratios=(0.8, 0.1, 0.1)):
    """Assign whole connected components to train/val/test greedily by size."""
    rng = np.random.default_rng(seed)
    comp_ids = np.unique(comp)
    members = {c: np.where(comp == c)[0] for c in comp_ids}
    # largest components first for balanced allocation; randomize ties
    order = sorted(comp_ids, key=lambda c: (-len(members[c]), rng.random()))
    buckets = [[], [], []]
    counts = [0, 0, 0]
    targets = [ratios[0], ratios[1], ratios[2]]
    for c in order:
        # pick bucket whose current share is most below its target share
        deficits = [targets[b] - counts[b] / max(n, 1) for b in range(3)]
        b = int(np.argmax(deficits))
        buckets[b].extend(members[c])
        counts[b] += len(members[c])
    return (np.array(sorted(buckets[0])),
            np.array(sorted(buckets[1])),
            np.array(sorted(buckets[2])))


def get_similarity_split(name, seed, threshold=0.05, ratios=(0.8, 0.1, 0.1)):
    sketches = build_sketches(name)
    dist = mash_distance_matrix(sketches)
    comp, _ = connected_components(dist, threshold)
    return group_split(comp, len(sketches), seed, ratios)


if __name__ == "__main__":
    sketches = build_sketches("cherry")
    print("sketches built:", len(sketches), "median size", np.median([len(s) for s in sketches]))
    d = mash_distance_matrix(sketches)
    print("distance matrix:", d.shape, "median off-diag", np.median(d[np.triu_indices(len(d), 1)]))
    for thr in (0.05, 0.1):
        comp, nc = connected_components(d, thr)
        sizes = np.bincount(comp)
        print(f"threshold {thr}: {nc} components, largest {sizes.max()}, singletons {np.sum(sizes==1)}")
    tr, va, te = get_similarity_split("cherry", 123)
    print("group split sizes:", len(tr), len(va), len(te))
