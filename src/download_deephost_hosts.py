# -*- coding: utf-8 -*-
"""Download one RefSeq representative genome for each of the 118 DeepHost host classes."""
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
OUTDIR = ROOT / "data" / "fasta" / "deephost_hosts"
OUTDIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUTDIR / "manifest.json"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 ** k + 1)


def species_list():
    out = []
    for line in (ROOT / "data/raw/DeepHost-master/data/species.txt").read_text().strip().splitlines():
        i, name = line.split("\t")
        out.append((int(i), name.strip()))
    return out


def pick_assembly(species):
    q = urllib.parse.quote(f'"{species}"[Organism] AND refseq[filter]')
    x = json.loads(get(f"{BASE}/esearch.fcgi?db=nuccore&term={q}&retmax=10&retmode=json"))
    ids = x["esearchresult"]["idlist"]
    if not ids:
        return None
    s = json.loads(get(f"{BASE}/esummary.fcgi?db=nuccore&id={','.join(ids)}&retmode=json"))
    cands = []
    for i in ids:
        it = s["result"][i]
        ln = int(it.get("slen", 0))
        cands.append((ln, i, it.get("title", "")))
    # prefer a chromosome-sized sequence; fall back to the largest available
    cands.sort(key=lambda z: (z[0] >= 200_000, z[0]), reverse=True)
    return cands[0] if cands else None


def main():
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for i, species in species_list():
        f = OUTDIR / f"{i:03d}.fasta"
        if f.exists() and str(i) in manifest:
            continue
        try:
            pick = pick_assembly(species)
            if not pick:
                print(f"[ {i:3d}] {species}: no assembly", flush=True)
                continue
            ln, gid, title = pick
            fa = get(f"{BASE}/efetch.fcgi?db=nuccore&id={gid}&rettype=fasta&retmode=text").decode()
            f.write_text(fa, encoding="utf-8")
            manifest[str(i)] = {"species": species, "id": gid, "length": ln, "title": title}
            MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[ {i:3d}] {species}: {gid} len={ln}", flush=True)
        except Exception as e:
            print(f"[ {i:3d}] {species}: ERROR {e}", flush=True)
        time.sleep(0.5)
    print("download pass complete; classes:", len(manifest))


if __name__ == "__main__":
    main()
