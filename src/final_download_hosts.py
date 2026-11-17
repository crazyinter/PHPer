# -*- coding: utf-8 -*-
"""Final robust downloader for DeepHost host genomes (aliases + tiered queries + pinned)."""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
OUTDIR = ROOT / "data" / "fasta" / "deephost_hosts"
MANIFEST = OUTDIR / "manifest.json"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MINLEN = 200_000

ALIASES = {
    "Mycobacterium smegmatis": ["Mycolicibacterium smegmatis"],
    "Mycobacteruim smegmatis": ["Mycobacterium smegmatis", "Mycolicibacterium smegmatis"],
    "Mycobacterium phlei": ["Mycolicibacterium phlei"],
    "Propionibacterium acnes": ["Cutibacterium acnes"],
    "Propionibacterium freudenreichii": ["Cutibacterium freudenreichii"],
    "Clostridium difficile": ["Clostridioides difficile"],
}
PINNED = {0: "NC_008596", 100: "NC_008596"}


def get(url, tries=4, timeout=180):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2 ** k + 2)
    raise last


def fasta_bp(text):
    return sum(len(ln.strip()) for ln in text.splitlines() if not ln.startswith(">"))


def search_ids(term, retmax=30):
    q = urllib.parse.quote(term)
    x = json.loads(get(f"{BASE}/esearch.fcgi?db=nuccore&term={q}&retmax={retmax}&retmode=json"))
    return x["esearchresult"]["idlist"]


def summarize(ids):
    if not ids:
        return []
    s = json.loads(get(f"{BASE}/esummary.fcgi?db=nuccore&id={','.join(ids)}&retmode=json"))
    return [(int(s["result"][i].get("slen", 0)), i, s["result"][i].get("title", "")) for i in ids]


def gather_candidates(i, species):
    names = [species] + ALIASES.get(species, [])
    found = {}
    for name in names:
        terms = [
            f'"{name}"[Organism] AND refseq[filter] AND "complete genome"[Title]',
            f'"{name}"[Organism] AND refseq[filter] AND "chromosome"[Title]',
            f'"{name}"[Organism] AND refseq[filter]',
        ]
        for term in terms:
            try:
                for slen, gid, title in summarize(search_ids(term)):
                    found[gid] = (slen, gid, title)
            except Exception:
                pass
    if i in PINNED:
        # efetch accepts the accession directly; skip esummary key mismatch
        acc = PINNED[i]
        found[acc] = (9_999_999, acc, "pinned reference genome")
    cands = sorted(found.values(), key=lambda z: z[0], reverse=True)
    return [c for c in cands if c[0] >= MINLEN] or cands


def load_std():
    out = []
    for line in (ROOT / "data/raw/DeepHost-master/data/species.txt").read_text().strip().splitlines():
        i, name = line.split("\t")
        out.append((int(i), name.strip()))
    return out


def current_bp(i):
    f = OUTDIR / f"{i:03d}.fasta"
    return fasta_bp(f.read_text(encoding="utf-8", errors="replace")) if f.exists() else 0


def main():
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    present = set()
    d = np.load(ROOT / "data/processed/deephost_std_processed.npz", allow_pickle=True)
    present = set(d["labels"].tolist())
    for i, species in load_std():
        if i not in present:
            continue  # no phages -> host genome not needed
        if current_bp(i) >= MINLEN:
            continue
        try:
            cands = gather_candidates(i, species)
            done = False
            for slen, gid, title in cands:
                fa = get(f"{BASE}/efetch.fcgi?db=nuccore&id={gid}&rettype=fasta&retmode=text",
                         timeout=180).decode()
                bp = fasta_bp(fa)
                if bp >= MINLEN:
                    (OUTDIR / f"{i:03d}.fasta").write_text(fa, encoding="utf-8")
                    manifest[str(i)] = {"species": species, "id": gid, "length": bp, "title": title}
                    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    print(f"[ {i:3d}] {species}: {gid} bp={bp}", flush=True)
                    done = True
                    break
            if not done:
                print(f"[ {i:3d}] {species}: no valid candidate", flush=True)
        except Exception as e:
            print(f"[ {i:3d}] {species}: ERROR {e}", flush=True)
        time.sleep(1.2)
    good = sum(1 for i in present if current_bp(i) >= MINLEN)
    print("complete; valid genomes for present classes:", good, "/", len(present))


if __name__ == "__main__":
    import numpy as np  # noqa: E402
    main()
