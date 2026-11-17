# -*- coding: utf-8 -*-
"""Host assembly provenance (reviewer #10).

Computes per-host assembly statistics directly from the pinned FASTA files
(contig count, N50, ambiguous bases, longest contig) and extracts strain and
assembly level from the record title. Records the no-cross-contig-ORF rule.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
FDIR = ROOT / "data" / "fasta" / "deephost_hosts"
manifest = json.loads((FDIR / "manifest.json").read_text(encoding="utf-8"))


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


def n50(lengths):
    total = sum(lengths); run = 0
    for L in sorted(lengths, reverse=True):
        run += L
        if run >= total / 2:
            return L
    return 0


def strain_from_title(title):
    m = re.search(r"strain\s+([^,]+?)(?:\s+chromosome|,|$)", title)
    return m.group(1).strip() if m else ""


def level_from_title(title, n_contig):
    t = title.lower()
    if "complete genome" in t:
        return "Complete genome"
    if "scaffold" in t:
        return "Scaffold"
    if "contig" in t:
        return "Contig"
    return "Complete genome" if n_contig == 1 else "Contig"


rows = []
for cid in sorted(manifest, key=int):
    meta = manifest[cid]
    path = FDIR / f"{int(cid):03d}.fasta"
    contigs = read_contigs(path)
    entry = {"class_id": cid, "species": meta["species"], "accession": meta["id"],
             "title": meta["title"], "strain": strain_from_title(meta["title"])}
    if not contigs:
        entry.update({"assembly_level": "No sequence available", "n_contig": 0,
                      "total_length": 0, "ambiguous_bases": 0, "n50": 0,
                      "longest_contig": 0})
    else:
        lengths = [len(s) for s in contigs]
        entry.update({"assembly_level": level_from_title(meta["title"], len(contigs)),
                      "n_contig": len(contigs), "total_length": sum(lengths),
                      "ambiguous_bases": sum(s.upper().count("N") for s in contigs),
                      "n50": n50(lengths), "longest_contig": max(lengths)})
    rows.append(entry)

protocol = {
    "orf_rule": "pyrodigal is invoked per contig; genes never span contig boundaries, "
                "and ambiguous N bases are not translated into codon counts.",
    "n_hosts": len(rows),
    "with_sequence": sum(r["n_contig"] > 0 for r in rows),
    "multi_contig": sum(r["n_contig"] > 1 for r in rows),
    "complete_genomes": sum(r["assembly_level"] == "Complete genome" for r in rows),
}
(RES / "host_assembly_table.json").write_text(
    json.dumps({"protocol": protocol, "records": rows}, indent=2), encoding="utf-8")
with open(RES / "host_assembly_table.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(json.dumps(protocol, indent=2))
print("multi-contig records:",
      [(r["class_id"], r["species"], r["n_contig"]) for r in rows if r["n_contig"] > 1])
