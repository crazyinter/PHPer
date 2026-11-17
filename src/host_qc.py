# -*- coding: utf-8 -*-
"""Sequence-level host QC: ambiguous fraction, non-ACGTN symbols, ORF density (recovered from
already-built protein features), outlier flags. RefSeq records are curated; no CheckM run."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
FDIR = ROOT / "data" / "fasta" / "deephost_hosts"
manifest = json.loads((FDIR / "manifest.json").read_text(encoding="utf-8"))

# recover gene counts from already-built host protein features (first component = log1p(ngenes))
hpf = np.load(PROC / "deephost_host_protein_features.npz", allow_pickle=True)
ngenes_map = {n: int(round(float(np.expm1(hpf["features"][i][0]))))
              for i, n in enumerate(hpf["host_names"])}


def read_contigs(path):
    seqs, cur = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(">"):
            if cur:
                s = "".join(cur)
                if s:
                    seqs.append(s)
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        s = "".join(cur)
        if s:
            seqs.append(s)
    return seqs


rows = []
for cid in sorted(manifest, key=int):
    path = FDIR / f"{int(cid):03d}.fasta"
    if not path.exists():
        continue
    contigs = read_contigs(path)
    if not contigs:
        continue
    species = manifest[cid]["species"]
    total = sum(len(s) for s in contigs)
    upper = [s.upper() for s in contigs]
    n_n = sum(s.count("N") for s in upper)
    non_acgtn = sum(1 for s in upper for ch in s if ch not in "ACGTN")
    n_genes = ngenes_map.get(species, 0)
    rows.append({
        "class_id": cid, "species": species, "total_length": total,
        "n_contig": len(contigs), "ambiguous_fraction": round(n_n / total, 6),
        "non_acgtn_bases": non_acgtn, "n_genes": n_genes,
        "genes_per_mb": round(n_genes / (total / 1e6), 2),
    })

dens = np.array([r["genes_per_mb"] for r in rows])
med = float(np.median(dens)); mad = float(np.median(np.abs(dens - med))) or 1.0
flags = []
for r in rows:
    rz = 0.6745 * (r["genes_per_mb"] - med) / mad
    problems = []
    if r["ambiguous_fraction"] > 0.01:
        problems.append("ambiguous>1%")
    if r["non_acgtn_bases"] > 0:
        problems.append("non-ACGTN symbols")
    if abs(rz) > 3.5:
        problems.append("ORF-density outlier")
    r["robust_z_orf_density"] = round(rz, 2)
    r["qc_flag"] = "; ".join(problems) if problems else "pass"
    if problems:
        flags.append((r["class_id"], r["species"], r["qc_flag"]))

with open(RES / "host_qc.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

summary = {
    "method": "RefSeq curated records; screen for ambiguous fraction, non-ACGTN symbols and robust ORF-density outliers; no CheckM run",
    "n_genomes_screened": len(rows),
    "median_genes_per_mb": round(med, 2),
    "max_ambiguous_fraction": max(r["ambiguous_fraction"] for r in rows),
    "genomes_with_non_acgtn": sum(r["non_acgtn_bases"] > 0 for r in rows),
    "n_flagged": len(flags), "flagged": flags, "records": rows,
}
(RES / "host_qc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("screened", len(rows), "median genes/Mb", round(med, 2), "flagged", len(flags))
for x in flags:
    print(x)
