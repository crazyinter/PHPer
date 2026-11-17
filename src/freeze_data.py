# -*- coding: utf-8 -*-
"""Freeze data: SHA-256 manifest, DeepHost empty classes, host accession table."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# 1. SHA-256 manifest of processed files (exclude the large cgr3d, documented separately)
files_manifest = []
for f in sorted(PROC.glob("*")):
    if f.is_file():
        files_manifest.append({"file": str(f.relative_to(ROOT)), "bytes": f.stat().st_size,
                               "sha256": sha256(f)})
(RES / "data_sha256_manifest.json").write_text(
    json.dumps(files_manifest, indent=2), encoding="utf-8")
print("processed files hashed:", len(files_manifest))

# 2. DeepHost empty classes (118 standard classes with zero phages)
dh = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
host_names = list(dh["host_names"])
present = set(int(x) for x in dh["labels"])
empty = [i for i in range(118) if i not in present]
(RES / "deephost_empty_classes.json").write_text(
    json.dumps({"standard_classes": 118, "occupied": len(present),
                "empty_class_ids": empty, "candidate_set_at_eval": len(present)}, indent=2),
    encoding="utf-8")
print("DeepHost occupied classes:", len(present), "empty:", empty)

# 3. Host accession table from the fasta manifest
fasta_manifest = json.loads((ROOT / "data" / "fasta" / "deephost_hosts" / "manifest.json")
                            .read_text(encoding="utf-8"))
accession_rows = [{"class_id": k, "species": v["species"], "accession": v["id"],
                   "length": v["length"], "title": v["title"]}
                  for k, v in sorted(fasta_manifest.items(), key=lambda kv: int(kv[0]))]
(RES / "host_accession_table.json").write_text(
    json.dumps(accession_rows, indent=2, ensure_ascii=False), encoding="utf-8")
print("host accession rows:", len(accession_rows))
