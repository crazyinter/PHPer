# -*- coding: utf-8 -*-
"""Protein-level features derived from ORFs (pyrodigal), no external database.

Feature vector (96-D):
  8 genome/ORF structure stats
  64 codon relative frequencies
  20 amino-acid relative frequencies
  4 physicochemical summaries
"""
import gzip
from pathlib import Path

import numpy as np
import pyrodigal

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
BASES = "ACGT"
CODON_ORDER = [a + b + c for a in BASES for b in BASES for c in BASES]
CODON_IDX = {c: i for i, c in enumerate(CODON_ORDER)}
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}

# Kyte-Doolittle hydrophobicity (subset used here)
KD = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
      "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
      "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
      "W": -0.9, "Y": -1.3}


def _finder():
    return pyrodigal.GeneFinder(meta=True)


def _raw_stats(seq, finder):
    genes = finder.find_genes(seq.encode())
    n = len(seq)
    codon_counts = np.zeros(64)
    aa_counts = np.zeros(20)
    hydro, n_residue = 0.0, 0
    pos, neg = 0, 0
    complete = 0
    orf_lens, gaps, overlaps = [], [], 0
    prev_end = None
    basic = acidic = sulfur = 0

    for g in genes:
        if g.strand > 0:
            pos += 1
        else:
            neg += 1
        if not g.partial_begin and not g.partial_end:
            complete += 1
        orf_lens.append(g.end - g.begin)
        if prev_end is not None:
            d = g.begin - prev_end
            if d < 0:
                overlaps += 1
            else:
                gaps.append(d)
        prev_end = g.end

        gene_nuc = g.sequence().upper()
        for i in range(0, len(gene_nuc) - 2, 3):
            c = gene_nuc[i:i + 3]
            if c in CODON_IDX:
                codon_counts[CODON_IDX[c]] += 1
        prot = g.translate()
        for ch in prot:
            if ch in AA_IDX:
                aa_counts[AA_IDX[ch]] += 1
                hydro += KD[ch]; n_residue += 1
                if ch in ("R", "K", "H"):
                    basic += 1
                if ch in ("D", "E"):
                    acidic += 1
                if ch in ("C", "M"):
                    sulfur += 1
    return {"n": n, "ngenes": len(genes), "codon_counts": codon_counts,
            "aa_counts": aa_counts, "hydro": hydro, "n_residue": n_residue,
            "pos": pos, "neg": neg, "complete": complete, "orf_lens": orf_lens,
            "gaps": gaps, "overlaps": overlaps, "basic": basic,
            "acidic": acidic, "sulfur": sulfur}


def _finalize(raw):
    n = raw["n"]; ngenes = raw["ngenes"]
    orf_lens = raw["orf_lens"]; gaps = raw["gaps"]
    coding_density = float(sum(orf_lens)) / max(n, 1)
    struct = np.array([
        np.log1p(ngenes),
        coding_density,
        np.mean(orf_lens) / 1000 if orf_lens else 0,
        np.std(orf_lens) / 1000 if orf_lens else 0,
        raw["pos"] / max(ngenes, 1),
        raw["complete"] / max(ngenes, 1),
        raw["overlaps"] / max(ngenes, 1),
        np.mean(gaps) / 1000 if gaps else 0,
    ], dtype=np.float32)
    codon_freq = raw["codon_counts"] / max(raw["codon_counts"].sum(), 1)
    aa_freq = raw["aa_counts"] / max(raw["aa_counts"].sum(), 1)
    phys = np.array([
        raw["hydro"] / max(raw["n_residue"], 1),
        raw["basic"] / max(raw["n_residue"], 1),
        raw["acidic"] / max(raw["n_residue"], 1),
        raw["sulfur"] / max(raw["n_residue"], 1),
    ], dtype=np.float32)
    return np.concatenate([struct, codon_freq.astype(np.float32),
                           aa_freq.astype(np.float32), phys]).astype(np.float32)


def extract_features(seq, finder=None):
    if finder is None:
        finder = _finder()
    return _finalize(_raw_stats(seq, finder))


def extract_features_contigs(contigs, finder=None):
    """Run gene calling independently on every contig, then pool raw counts so
    no ORF can span a contig boundary."""
    if finder is None:
        finder = _finder()
    raws = [_raw_stats(s, finder) for s in contigs if s]
    if not raws:
        return _finalize(_raw_stats("", finder))
    pooled = raws[0].copy()
    for r in raws[1:]:
        for key in ("codon_counts", "aa_counts"):
            pooled[key] = pooled[key] + r[key]
        for key in ("n", "ngenes", "hydro", "n_residue", "pos", "neg",
                    "complete", "overlaps", "basic", "acidic", "sulfur"):
            pooled[key] += r[key]
        pooled["orf_lens"] = pooled["orf_lens"] + r["orf_lens"]
        pooled["gaps"] = pooled["gaps"] + r["gaps"]
    return _finalize(pooled)


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return "".join(line.strip() for line in fh if not line.startswith(">"))


def build_all():
    from dataset import parse_fasta_file, base_id
    finder = _finder()
    # phages
    ph = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
    C = ROOT / "data" / "raw" / "CHERRY-main"
    seq_map = {base_id(h): s for h, s in parse_fasta_file(C / "dataset" / "nucl.fasta")}
    accs = list(ph["accs"])
    pf = np.stack([extract_features(seq_map[a], finder) for a in accs])
    np.savez_compressed(PROC / "cherry_protein_features.npz",
                        features=pf, accs=np.array(accs))
    print("phage protein features:", pf.shape)

    # hosts: iterate over the gz files that actually exist
    import csv
    hf = np.load(PROC / "cherry_host_fcgr.npz", allow_pickle=True)
    host_names = list(hf["host_names"])
    acc2sp = {}
    with open(C / "dataset" / "prokaryote.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            acc2sp[r["Accession"]] = r["Species"]
    gz_dir = C / "prokaryote"
    sp2feat = {}
    for p in sorted(gz_dir.glob("*.fa.gz")):
        sp = acc2sp[p.name.replace(".fa.gz", "")]
        if sp not in sp2feat:  # first available genome wins for a species
            sp2feat[sp] = extract_features(read_gz(p), finder)
    hpf = np.stack([sp2feat[sp] for sp in host_names])
    np.savez_compressed(PROC / "cherry_host_protein_features.npz",
                        features=hpf, host_names=np.array(host_names))
    print("host protein features:", hpf.shape)


if __name__ == "__main__":
    build_all()
