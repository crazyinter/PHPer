# -*- coding: utf-8 -*-
"""Fetch every contig of a WGS project and merge into one FASTA file."""
import json
import time
import sys
import urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from redownload_deephost_hosts import get, fasta_bp, BASE

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
OUTDIR = ROOT / "data" / "fasta" / "deephost_hosts"
MANIFEST = OUTDIR / "manifest.json"

JOBS = {
    42: ("NZ_MUML00000000", "Streptomyces viridochromogenes"),
    105: ("NZ_JBMVCF000000000", "Streptomyces azureus"),
}


def esearch_all(master):
    q = urllib.parse.quote(f"{master}[All Fields]")
    x = json.loads(get(f"{BASE}/esearch.fcgi?db=nuccore&term={q}&retmax=2000&retmode=json"))
    return x["esearchresult"]["idlist"]


def esummary_map(ids):
    out = {}
    for b in range(0, len(ids), 100):
        chunk = ids[b:b + 100]
        s = json.loads(get(f"{BASE}/esummary.fcgi?db=nuccore&id={','.join(chunk)}&retmode=json"))
        for i in chunk:
            r = s["result"][i]
            out[i] = (int(r.get("slen", 0)), r.get("caption", i))
        time.sleep(0.4)
    return out


def efetch_batch(ids):
    fa = get(f"{BASE}/efetch.fcgi?db=nuccore&id={','.join(ids)}&rettype=fasta&retmode=text",
             timeout=300).decode()
    return fa


for cls, (master, species) in JOBS.items():
    print("=" * 70)
    all_ids = esearch_all(master)
    info = esummary_map(all_ids)
    master_gid = next((g for g in all_ids if info[g][1] == master.replace("NZ_", "")), all_ids[0])
    contigs = [g for g in all_ids if g != master_gid and info[g][0] > 0]
    print(f"class {cls}: {len(contigs)} contigs, expected total "
          f"{sum(info[g][0] for g in contigs)} bp")
    parts = []
    for b in range(0, len(contigs), 40):
        parts.append(efetch_batch(contigs[b:b + 40]))
        time.sleep(1.0)
    merged = "".join(parts)
    bp = fasta_bp(merged)
    (OUTDIR / f"{cls:03d}.fasta").write_text(merged, encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text())
    manifest[str(cls)] = {"species": species, "id": master, "length": bp,
                          "title": "WGS contigs merged"}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  saved {cls:03d}.fasta with {bp} bp, {merged.count('>')} contigs")
