# -*- coding: utf-8 -*-
"""Alternative representative-genome sensitivity (real downloads + recomputed features + prediction flip)."""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from fcgr import encode_fcgr
from protein_features import extract_features_contigs, _finder
from model_bmk import build_model
from train import load_dataset, random_split
from splits import get_similarity_split

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
ALTDIR = ROOT / "data" / "fasta" / "alt_representatives"
ALTDIR.mkdir(parents=True, exist_ok=True)
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TARGETS = [
    "Escherichia coli", "Pseudomonas aeruginosa", "Salmonella enterica",
    "Bacillus cereus", "Klebsiella pneumoniae", "Staphylococcus aureus",
    "Streptomyces viridochromogenes", "Gordonia terrae",
]


def get(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k + 1)


def parse_fasta(text):
    contigs, cur = [], []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                contigs.append("".join(cur))
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        contigs.append("".join(cur))
    return [c for c in contigs if c]


def cosine(a, b):
    a = a.ravel(); b = b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pearson(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


manifest = json.loads((ROOT / "data/fasta/deephost_hosts/manifest.json").read_text(encoding="utf-8"))
name_to_class = {v["species"]: (int(k), v["id"]) for k, v in manifest.items()}

# primary host features
hfc = np.load(PROC / "deephost_host_fcgr.npz", allow_pickle=True)
hname = list(hfc["host_names"]); hmat = hfc["matrices"]
hpf = np.load(PROC / "deephost_host_protein_features.npz", allow_pickle=True)
hpf_names = list(hpf["host_names"]); hpf_feat = hpf["features"]
finder = _finder()

records = []
for sp in TARGETS:
    if sp not in name_to_class:
        print("skip (not in manifest):", sp); continue
    cls, primary_id = name_to_class[sp]
    alt_path = ALTDIR / f"{cls:03d}_alt.fasta"
    if not alt_path.exists():
        q = urllib.parse.quote(f'"{sp}"[Organism] AND refseq[filter]')
        x = json.loads(get(f"{BASE}/esearch.fcgi?db=nuccore&term={q}&retmax=200&retmode=json"))
        ids = x["esearchresult"]["idlist"]
        # esummary in chunks of 100
        summ = {}
        for b in range(0, len(ids), 100):
            chunk = ids[b:b + 100]
            ss = json.loads(get(f"{BASE}/esummary.fcgi?db=nuccore&id={','.join(chunk)}&retmode=json"))
            summ.update(ss["result"])
        cands = sorted(((int(summ[i].get("slen", 0)), i) for i in ids), reverse=True)
        chosen = None
        # prefer a full chromosome/WGS master record of another strain, then fall back to >=200 kb
        for minlen in (1_000_000, 200_000):
            for ln, uid in cands:
                if ln < minlen or str(uid) == str(primary_id):
                    continue
                fa = get(f"{BASE}/efetch.fcgi?db=nuccore&id={uid}&rettype=fasta&retmode=text").decode()
                contigs = parse_fasta(fa)
                if sum(len(c) for c in contigs) >= 200_000:
                    alt_path.write_text(fa, encoding="utf-8")
                    chosen = (uid, contigs); break
                time.sleep(0.4)
            if chosen is not None:
                break
        if chosen is None:
            print("no usable alt for", sp); continue
        alt_uid, contigs = chosen
        time.sleep(0.5)
    else:
        contigs = parse_fasta(alt_path.read_text(encoding="utf-8"))
        alt_uid = alt_path.stem

    # primary
    pi = hname.index(sp)
    primary_fcgr = hmat[pi]
    primary_prot = hpf_feat[hpf_names.index(sp)]
    # alt features
    alt_concat = "".join(contigs)
    alt_fcgr = encode_fcgr(alt_concat)
    alt_prot = extract_features_contigs(contigs, finder)

    rec = {
        "species": sp, "class": cls, "alt_id": alt_uid,
        "primary_len": int(sum(len(x) for x in parse_fasta(
            (ROOT / f'data/fasta/deephost_hosts/{cls:03d}.fasta').read_text(encoding='utf-8')))),
        "alt_len": len(alt_concat), "alt_n_contig": len(contigs),
        "fcgr_cosine": round(cosine(primary_fcgr, alt_fcgr), 6),
        "protein_pearson": round(pearson(primary_prot, alt_prot), 6),
        "alt_fcgr": alt_fcgr,
    }
    records.append(rec)
    print(sp, "fcgr cos", rec["fcgr_cosine"], "prot r", rec["protein_pearson"])

# ---- end-to-end prediction flip ----
device = "cuda" if torch.cuda.is_available() else "cpu"
X, y, H, shared = load_dataset("deephost")
sp_to_hostidx = {n: i for i, n in enumerate(shared)}


def eval_split(split_type):
    if split_type == "random":
        tr, va, te = random_split(len(X), 123)
        ckpt = "deephost_random_fcgr_seed123.pt"
    else:
        tr, va, te = get_similarity_split("deephost_std", 123, threshold=0.05)
        ckpt = "deephost_similarity_fcgr_seed123.pt"
    model = build_model(6).to(device)
    model.load_state_dict(torch.load(ROOT / "models" / ckpt, map_location=device))
    model.eval()
    Ht = torch.tensor(H).unsqueeze(1).to(device)
    Xte = torch.tensor(X[te]).unsqueeze(1).to(device)
    out = {}
    with torch.no_grad():
        base_emb = model(Ht)
        base_dist = torch.cdist(model(Xte), base_emb)
        base_pred = base_dist.argmin(dim=1).cpu().numpy()
        base_top1 = float((base_pred == y[te]).mean())
        per_species = {}
        for rec in records:
            if rec["species"] not in sp_to_hostidx:
                continue
            hi = sp_to_hostidx[rec["species"]]
            Hswap = Ht.clone()
            Hswap[hi] = torch.tensor(rec["alt_fcgr"]).unsqueeze(0).to(device)
            with torch.no_grad():
                swap_emb = model(Hswap)
                dist = torch.cdist(model(Xte), swap_emb)
                pred = dist.argmin(dim=1).cpu().numpy()
            flips = int((pred != base_pred).sum())
            top1 = float((pred == y[te]).mean())
            per_species[rec["species"]] = {
                "prediction_flips": flips, "n_test": len(te),
                "flip_rate": round(flips / len(te), 6),
                "top1_after": round(top1, 6), "delta_top1": round(top1 - base_top1, 6),
            }
        out["baseline_top1"] = round(base_top1, 6)
        out["per_species"] = per_species
    return out


results = {
    "selection_protocol": "one RefSeq representative per species; chromosome-sized sequences preferred (>=200 kb); alternative is a different strain's RefSeq assembly",
    "feature_level": [
        {"species": r["species"], "class": r["class"], "alt_id": r["alt_id"],
         "primary_len": r["primary_len"], "alt_len": r["alt_len"],
         "alt_n_contig": r["alt_n_contig"], "fcgr_cosine": r["fcgr_cosine"],
         "protein_pearson": r["protein_pearson"]} for r in records],
    "random_split": eval_split("random"),
    "similarity_split": eval_split("similarity"),
}
(RES / "representative_sensitivity.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nsaved representative_sensitivity.json")
print("random:", json.dumps(results["random_split"], indent=1))
