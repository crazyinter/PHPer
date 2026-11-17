"""Data loading: parse downloaded FASTA batches and align with host_info labels."""
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RAW = ROOT / "data" / "raw"


def base_id(acc):
    return acc.split(".")[0]


def parse_fasta_file(path):
    """Yield (header_id, sequence) from a fasta file."""
    hid, parts = None, []
    with open(path, encoding="utf-8", errors="strict") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if hid is not None:
                    yield hid, "".join(parts)
                hid = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
    if hid is not None:
        yield hid, "".join(parts)


def load_host_info():
    rows = []
    with open(RAW / "DeepHost-master" / "data" / "host_info.txt", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append({"acc": p[0], "species": p[1], "genus": p[2]})
    return rows


def load_downloaded():
    """Map base accession -> sequence from all available batch files."""
    out = {}
    dirs = [RAW / "genome_batches", RAW / "slow_batches"]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.fasta")))
    for f in files:
        for hid, seq in parse_fasta_file(f):
            out[base_id(hid)] = seq
    return out


def build_labeled():
    """Align host_info rows with downloaded sequences.

    Returns (matched list, missing accessions).
    Each matched item: dict(acc, species, genus, seq).
    """
    info = load_host_info()
    downloaded = load_downloaded()
    matched, missing = [], []
    for row in info:
        b = base_id(row["acc"])
        if b in downloaded:
            matched.append({**row, "seq": downloaded[b]})
        else:
            missing.append(row["acc"])
    return matched, missing


if __name__ == "__main__":
    matched, missing = build_labeled()
    print("host_info total:", len(matched) + len(missing))
    print("matched with sequences:", len(matched))
    print("missing (not downloaded / obsolete):", len(missing))
    print("missing sample:", missing[:15])
    lens = [len(m["seq"]) for m in matched]
    print("seq length min/median/max:", min(lens), sorted(lens)[len(lens) // 2], max(lens))
