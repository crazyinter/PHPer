# -*- coding: utf-8 -*-
"""Host-phylogeny controls (reviewer comment #4).

Builds a data-driven host hierarchy from host FCGR and uses the explicit genus
rank, then asks whether the codon signal survives after shared ancestry is
controlled: (1) genus-level prediction, (2) within-genus species discrimination,
(3) genus-mean residualization, (4) host-tree-PC residualization.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train import random_split  # noqa
from splits import (build_sketches, mash_distance_matrix,  # noqa
                    connected_components, group_split)

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
SEEDS = [123, 2024, 42, 7, 99]
NPC = 10


def load(name):
    if name == "cherry":
        ph = np.load(PROC / "cherry_processed.npz", allow_pickle=True)
        feat = np.load(PROC / "cherry_protein_features.npz",
                       allow_pickle=True)["features"]
        hf = np.load(PROC / "cherry_host_fcgr.npz", allow_pickle=True)
    else:
        ph = np.load(PROC / "deephost_std_processed.npz", allow_pickle=True)
        feat = np.load(PROC / "deephost_protein_features.npz",
                       allow_pickle=True)["features"]
        hf = np.load(PROC / "deephost_host_fcgr.npz", allow_pickle=True)
    y = np.array(ph["labels"])
    host_names = [str(x) for x in ph["host_names"]]
    # align host FCGR to the processed host-name order; empty classes -> NaN
    hn = [str(x) for x in hf["host_names"]]
    pos = {n: i for i, n in enumerate(hn)}
    grid = hf["matrices"].shape[1:]
    hfc = np.full((len(host_names),) + grid, np.nan, dtype=np.float32)
    for i, n in enumerate(host_names):
        if n in pos:
            hfc[i] = hf["matrices"][pos[n]]
    genus = [n.split()[0] for n in host_names]
    return y, feat[:, 8:72], host_names, genus, hfc


def lr_fit_score(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1).fit(sc.transform(Xtr), ytr)
    return clf.score(sc.transform(Xte), yte)


def evaluate(name):
    y, codon, host_names, genus, hfc = load(name)
    n = len(y)
    genera = sorted(set(genus))
    gid = {g: i for i, g in enumerate(genera)}
    y_genus = np.array([gid[genus[t]] for t in y])
    multi_genera = [g for g in genera if sum(np.array(genus) == g) >= 2]
    # data-driven host hierarchy and its principal axes (valid hosts only)
    flat_all = hfc.reshape(len(host_names), -1)
    valid = ~np.isnan(flat_all).any(1)
    flat = flat_all[valid]
    Z = linkage(flat, method="average", metric="euclidean")
    pca = PCA(n_components=NPC).fit(flat)
    host_pc = np.zeros((len(host_names), NPC))
    host_pc[valid] = pca.transform(flat)
    phage_pc = host_pc[y]

    out = {"dataset": name, "n_species": len(host_names), "n_genera": len(genera),
           "multi_species_genera": len(multi_genera),
           "host_tree_pc_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
           "tree_linkage": "average linkage on host FCGR Euclidean distances",
           "random_split": {}, "similarity_seed123": {}}

    def within_genus(tr, te):
        correct = total = 0
        per = []
        for g in multi_genera:
            spec = np.where(np.array(genus) == g)[0]
            local = {s: k for k, s in enumerate(spec)}
            trg = [i for i in tr if genus[y[i]] == g]
            teg = [i for i in te if genus[y[i]] == g]
            if not trg or not teg:
                continue
            yltr = np.array([local[y[i]] for i in trg])
            ylte = np.array([local[y[i]] for i in teg])
            if len(np.unique(yltr)) < 2:
                c = int((ylte == yltr[0]).sum())  # only one class learnable
            else:
                sc = StandardScaler().fit(codon[trg])
                clf = LogisticRegression(max_iter=3000, C=1).fit(
                    sc.transform(codon[trg]), yltr)
                c = int((clf.predict(sc.transform(codon[teg])) == ylte).sum())
            correct += c; total += len(teg)
            per.append({"genus": g, "n_test": len(teg), "correct": c})
        return correct / max(total, 1), per

    def genus_residualized(tr):
        means = {}
        for g in genera:
            m = [i for i in tr if genus[y[i]] == g]
            if m:
                means[g] = codon[m].mean(0)
        glob = codon[tr].mean(0)
        return codon - np.array([means.get(genus[y[i]], glob) for i in range(n)])

    def tree_residualized(tr):
        reg = LinearRegression().fit(phage_pc[tr], codon[tr])
        return codon - reg.predict(phage_pc)

    def run_split(tr, te, bucket):
        codon_res_g = genus_residualized(tr)
        codon_res_t = tree_residualized(tr)
        vals = {
            "species_codon_baseline": lr_fit_score(codon[tr], y[tr], codon[te], y[te]),
            "genus_level_accuracy": lr_fit_score(codon[tr], y_genus[tr],
                                                 codon[te], y_genus[te]),
            "genus_residualized_species": lr_fit_score(codon_res_g[tr], y[tr],
                                                       codon_res_g[te], y[te]),
            "treePC_residualized_species": lr_fit_score(codon_res_t[tr], y[tr],
                                                        codon_res_t[te], y[te]),
        }
        wg, per = within_genus(tr, te)
        vals["within_genus_species_accuracy"] = wg
        for k, v in vals.items():
            out[bucket].setdefault(k, []).append(v)
        out[bucket].setdefault("within_genus_detail", []).append(per)

    for s in SEEDS:
        tr, va, te = random_split(n, s)
        run_split(tr, te, "random_split")
    if name == "cherry":
        sk = build_sketches("cherry")
    else:
        sk = list(np.load(PROC / "deephost_std_sketches.npz",
                          allow_pickle=True)["sketches"])
    comp, _ = connected_components(mash_distance_matrix(sk), 0.05)
    tr, va, te = group_split(comp, n, 123)
    run_split(tr, te, "similarity_seed123")

    for bucket in ("random_split", "similarity_seed123"):
        detail = out[bucket].pop("within_genus_detail")
        summary = {}
        for k, v in out[bucket].items():
            summary[k] = {"mean": float(np.mean(v)), "values": [round(x, 4) for x in v]}
        summary["within_genus_detail_seed123"] = detail[-1]
        out[bucket] = summary
    return out


report = [evaluate("cherry"), evaluate("deephost")]
(RES / "phylogeny_controls.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
for r in report:
    print(r["dataset"], "| random:",
          {k: round(v["mean"], 4) for k, v in r["random_split"].items()
           if k != "within_genus_detail_seed123"})
    print(r["dataset"], "| similarity:",
          {k: round(v["mean"], 4) for k, v in r["similarity_seed123"].items()
           if k != "within_genus_detail_seed123"})
print("saved phylogeny_controls.json")
