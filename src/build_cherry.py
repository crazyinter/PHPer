from pathlib import Path
import sys
from collections import Counter

import xlrd

sys.path.insert(0, str(Path(__file__).parent))
from dataset import parse_fasta_file, base_id  # noqa: E402

C = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\data\raw\CHERRY-main")
PROC = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\data\processed")


def load_pairs(xls_name, sheet_name):
    book = xlrd.open_workbook(str(C / "Interactiondata" / xls_name))
    sh = book.sheet_by_name(sheet_name)
    pairs = []
    for r in range(2, sh.nrows):  # two header rows
        acc = str(sh.cell_value(r, 0)).strip()
        host_species = str(sh.cell_value(r, 15)).strip()
        host_genus = str(sh.cell_value(r, 14)).strip()
        if acc:
            pairs.append({"acc": acc, "host_species": host_species, "host_genus": host_genus})
    return pairs


def main():
    train = load_pairs("VHM_PAIR_TAX.xls", "PAIR1426")
    test = load_pairs("TEST_PAIR_TAX.xls", "PAIR671")
    print("raw pairs: train", len(train), "test", len(test), "total", len(train) + len(test))

    seq_map = {base_id(h): s for h, s in parse_fasta_file(C / "dataset" / "nucl.fasta")}
    print("nucl sequence map:", len(seq_map))

    all_pairs = train + test
    uniq_acc = set(base_id(p["acc"]) for p in all_pairs)
    print("unique phage accessions in pairs:", len(uniq_acc))
    covered = [a for a in uniq_acc if a in seq_map]
    print("unique phage with sequences:", len(covered))

    hosts = Counter(p["host_species"] for p in all_pairs)
    print("unique host species (all pairs):", len(hosts))
    hg = Counter(p["host_genus"] for p in all_pairs)
    print("unique host genera:", len(hg))

    # pair-level coverage
    miss = [p["acc"] for p in all_pairs if base_id(p["acc"]) not in seq_map]
    print("pairs missing sequence:", len(miss), miss[:10])

    # empty host species check
    empty = [p["acc"] for p in all_pairs if not p["host_species"]]
    print("pairs with empty host species:", len(empty))


if __name__ == "__main__":
    main()
