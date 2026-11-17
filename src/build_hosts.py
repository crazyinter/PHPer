"""Build host (bacterial genome) FCGR matrices from CHERRY prokaryote/*.fa.gz."""
import csv
import gzip
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fcgr import encode_fcgr  # noqa: E402

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
C = ROOT / "data" / "raw" / "CHERRY-main"
PROC = ROOT / "data" / "processed"
OUT = PROC / "cherry_host_fcgr.npz"


def read_gz_fasta(path):
    parts = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip())
    return "".join(parts)


def main():
    rows = {}
    with open(C / "dataset" / "prokaryote.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["Accession"]] = r["Species"]
    print("prokaryote.csv rows:", len(rows), "unique species:", len(set(rows.values())))

    gz_files = sorted((C / "prokaryote").glob("*.fa.gz"))
    print("gz genomes available:", len(gz_files))
    missing_gz = [g for g in rows if not (C / "prokaryote" / f"{g}.fa.gz").exists()]
    print("csv entries without gz:", missing_gz)

    host_names, matrices = [], []
    for i, gz in enumerate(gz_files):
        gcf = gz.name.replace(".fa.gz", "")
        species = rows.get(gcf)
        if species is None:
            continue
        seq = read_gz_fasta(gz)
        matrices.append(encode_fcgr(seq))
        host_names.append(species)
        if (i + 1) % 25 == 0:
            print(f"  host FCGR {i+1}/{len(gz_files)}", flush=True)

    np.savez_compressed(OUT, matrices=np.stack(matrices).astype(np.float32),
                        host_names=np.array(host_names))
    print("saved", OUT, "hosts:", len(host_names))

    # coverage against the phage host labels used in the pairs
    ph = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
    pair_hosts = set(ph["host_names"])
    host_set = set(host_names)
    print("pair host species:", len(pair_hosts), "covered by host FCGR:", len(pair_hosts & host_set))
    print("uncovered:", sorted(pair_hosts - host_set)[:10])


if __name__ == "__main__":
    main()
