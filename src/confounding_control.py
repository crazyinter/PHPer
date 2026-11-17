# -*- coding: utf-8 -*-
"""Confounding controls: GC/length/gene-count baseline and residualized codon signal."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from dataset import parse_fasta_file, base_id, build_labeled  # noqa
from train import random_split  # noqa

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
SEEDS = [123, 2024, 42, 7, 99]


def load(name):
    if name == "cherry":
        ph = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
        seq_map = {base_id(h): s for h, s in parse_fasta_file(
            ROOT / "data" / "raw" / "CHERRY-main" / "dataset" / "nucl.fasta")}
        feat = np.load(PROC / "cherry_protein_features.npz", allow_pickle=True)["features"]
    else:
        ph = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
        labeled, _ = build_labeled()
        seq_map = {base_id(r["acc"]): r["seq"] for r in labeled}
        feat = np.load(PROC / "deephost_protein_features.npz",
                       allow_pickle=True)["features"]
    accs = list(ph["accs"]); y = np.array(ph["labels"])
    seqs = [seq_map.get(a, "") for a in accs]
    gc = np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs])
    length = np.array([len(s) for s in seqs], dtype=float)
    gene = np.expm1(feat[:, 0])
    return y, feat, np.column_stack([gc, np.log1p(length), gene]), gc


def lr_top1(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1).fit(sc.transform(Xtr), ytr)
    return clf.score(sc.transform(Xte), yte)


def evaluate(name):
    y, feat, simple, gc = load(name)
    codon = feat[:, 8:72]
    n = len(y)
    corr_gc = np.abs([np.corrcoef(codon[:, j], gc)[0, 1] for j in range(64)])
    out = {"dataset": name,
           "codon_gc_corr_mean": float(np.nanmean(corr_gc)),
           "codon_gc_corr_max": float(np.nanmax(corr_gc)),
           "simple_correlations": {
               "gc_loglength": float(np.corrcoef(simple[:, 0], simple[:, 1])[0, 1]),
               "gc_genecount": float(np.corrcoef(simple[:, 0], simple[:, 2])[0, 1])},
           "random_split": {}, "similarity_seed123": {}}

    def run_split(idx_tr, idx_te, bucket):
        # residualize codon on GC using training-only regression
        lr = LinearRegression().fit(simple[idx_tr, 0:1], codon[idx_tr])
        codon_res = codon - lr.predict(simple[:, 0:1])
        sets = {"simple_GC_length_gene": simple,
                "codon_only": codon,
                "codon_residualized_on_GC": codon_res,
                "simple_plus_codon": np.column_stack([simple, codon]),
                "full_96": feat}
        for k, X in sets.items():
            out[bucket].setdefault(k, []).append(
                lr_top1(X[idx_tr], y[idx_tr], X[idx_te], y[idx_te]))

    for s in SEEDS:
        tr, va, te = random_split(n, s)
        run_split(tr, te, "random_split")
    # similarity seed123: reuse cached component split if available, else build
    sys.path.insert(0, str(Path(__file__).parent))
    from splits import build_sketches, mash_distance_matrix, connected_components, group_split
    if name == "cherry":
        sk = build_sketches("cherry")
    else:
        sk = list(np.load(PROC / "deephost_std_sketches.npz",
                          allow_pickle=True)["sketches"])
    comp, _ = connected_components(mash_distance_matrix(sk), 0.05)
    tr, va, te = group_split(comp, n, 123)
    run_split(tr, te, "similarity_seed123")

    for bucket in ("random_split", "similarity_seed123"):
        out[bucket] = {k: {"mean": float(np.mean(v)), "values": [round(x, 4) for x in v]}
                        for k, v in out[bucket].items()}
    return out


report = [evaluate("cherry"), evaluate("deephost")]
(RES / "confounding_controls.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
