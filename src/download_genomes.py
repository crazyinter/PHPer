import json
import time
from pathlib import Path

from Bio import Entrez

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RAW = ROOT / "data" / "raw"
BATCH_DIR = RAW / "genome_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = BATCH_DIR / "progress.json"

Entrez.email = "research@nefu.edu.cn"
BATCH_SIZE = 200
MAX_RETRY = 6


def load_accessions():
    accs = []
    with open(RAW / "DeepHost-master" / "data" / "host_info.txt", encoding="utf-8") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                accs.append(p[0])
    return accs


def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"done": [], "failed_batches": []}


def save_progress(progress):
    PROGRESS.write_text(json.dumps(progress, indent=2), encoding="utf-8")


class PartialBatch(Exception):
    def __init__(self, returned, expected):
        self.returned = returned
        self.expected = expected
        super().__init__(f"partial batch {returned}/{expected}")


def fetch_batch(ids):
    last = None
    for attempt in range(MAX_RETRY):
        try:
            handle = Entrez.efetch(db="nuccore", id=",".join(ids),
                                   rettype="fasta", retmode="text")
            text = handle.read()
            handle.close()
            n = sum(1 for line in text.splitlines() if line.startswith(">"))
            if n == len(ids):
                return text
            if n == 0:
                # rate limit / transient service issue: long backoff
                print(f"    0 records (rate limited), waiting {20 + attempt*10}s", flush=True)
                time.sleep(20 + attempt * 10)
                continue
            # 0 < n < expected: some accessions are obsolete
            raise PartialBatch(n, len(ids))
        except PartialBatch:
            raise
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"batch failed after retries: {last}")


def main():
    accs = load_accessions()
    progress = load_progress()
    done = set(progress["done"])
    batches = [accs[i:i + BATCH_SIZE] for i in range(0, len(accs), BATCH_SIZE)]
    print(f"total {len(accs)} accessions in {len(batches)} batches; already done {len(done)}")

    for bi, ids in enumerate(batches):
        bfile = BATCH_DIR / f"batch_{bi:03d}.fasta"
        if bi in done and bfile.exists():
            continue
        try:
            text = fetch_batch(ids)
            bfile.write_text(text, encoding="utf-8")
            done.add(bi)
            progress["done"] = sorted(done)
            save_progress(progress)
            print(f"batch {bi:03d} ok ({len(ids)} seqs), total done {len(done)}/{len(batches)}", flush=True)
        except PartialBatch as exc:
            print(f"batch {bi:03d} PARTIAL ({exc.returned} returned), will resolve individually", flush=True)
            progress.setdefault("partial", []).append(bi)
            save_progress(progress)
        except Exception as exc:
            print(f"batch {bi:03d} FAILED: {exc}", flush=True)
            progress["failed_batches"].append(bi)
            save_progress(progress)
        time.sleep(1.5)

    print("download loop finished. done batches:", len(done), "/", len(batches))
    print("partial:", progress.get("partial", []), "failed:", progress.get("failed_batches", []))


if __name__ == "__main__":
    main()
