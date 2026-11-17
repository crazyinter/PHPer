"""Preprocess datasets once: encode FCGR and cache to processed/*.npz."""
from pathlib import Path
import sys

import numpy as np
import xlrd

sys.path.insert(0, str(Path(__file__).parent))
from dataset import parse_fasta_file, base_id, build_labeled  # noqa: E402
from fcgr import encode_fcgr  # noqa: E402

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
C = ROOT / "data" / "raw" / "CHERRY-main"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def cherry_pairs(xls_name, sheet_name):
    book = xlrd.open_workbook(str(C / "Interactiondata" / xls_name))
    sh = book.sheet_by_name(sheet_name)
    rows = []
    for r in range(2, sh.nrows):
        acc = str(sh.cell_value(r, 0)).strip()
        hsp = str(sh.cell_value(r, 15)).strip()
        if acc and hsp:
            rows.append((acc, hsp))
    return rows


def preprocess_cherry():
    out = PROC / "cherry_processed.npz"
    if out.exists():
        print("cherry already cached:", out)
        return
    pairs = cherry_pairs("VHM_PAIR_TAX.xls", "PAIR1426") + cherry_pairs("TEST_PAIR_TAX.xls", "PAIR671")
    seq_map = {base_id(h): s for h, s in parse_fasta_file(C / "dataset" / "nucl.fasta")}

    # dedup by base accession, keep first
    seen, recs = set(), []
    for acc, hsp in pairs:
        b = base_id(acc)
        if b in seen or b not in seq_map:
            continue
        seen.add(b)
        recs.append((b, hsp, seq_map[b]))
    print("cherry records to encode:", len(recs))

    host_names = sorted(set(h for _, h, _ in recs))
    label_map = {h: i for i, h in enumerate(host_names)}
    matrices = np.zeros((len(recs), 64, 64), dtype=np.float32)
    labels = np.zeros(len(recs), dtype=np.int64)
    accs = np.empty(len(recs), dtype=object)
    for i, (b, h, s) in enumerate(recs):
        matrices[i] = encode_fcgr(s)
        labels[i] = label_map[h]
        accs[i] = b
        if (i + 1) % 200 == 0:
            print(f"  encoded {i+1}/{len(recs)}", flush=True)

    np.savez_compressed(out, matrices=matrices, labels=labels,
                        host_names=np.array(host_names), accs=accs)
    print("saved", out, "N=", len(recs), "hosts=", len(host_names))


def preprocess_deephost():
    out = PROC / "deephost_processed.npz"
    if out.exists():
        print("deephost already cached:", out)
        return
    matched, _ = build_labeled()
    # exact dedup, keep species labels; restrict to non-empty species
    seen, recs = set(), []
    for row in matched:
        if row["seq"] in seen or not row["species"]:
            continue
        seen.add(row["seq"])
        recs.append((base_id(row["acc"]), row["species"], row["seq"]))
    print("deephost records to encode:", len(recs))

    host_names = sorted(set(h for _, h, _ in recs))
    label_map = {h: i for i, h in enumerate(host_names)}
    matrices = np.zeros((len(recs), 64, 64), dtype=np.float32)
    labels = np.zeros(len(recs), dtype=np.int64)
    accs = np.empty(len(recs), dtype=object)
    for i, (b, h, s) in enumerate(recs):
        matrices[i] = encode_fcgr(s)
        labels[i] = label_map[h]
        accs[i] = b
        if (i + 1) % 400 == 0:
            print(f"  encoded {i+1}/{len(recs)}", flush=True)

    np.savez_compressed(out, matrices=matrices, labels=labels,
                        host_names=np.array(host_names), accs=accs)
    print("saved", out, "N=", len(recs), "hosts=", len(host_names))


if __name__ == "__main__":
    preprocess_cherry()
    preprocess_deephost()
