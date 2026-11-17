# -*- coding: utf-8 -*-
"""Classical ML baselines under the identical data, splits, and evaluation protocol."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from train import load_dataset, random_split
from splits import get_similarity_split

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"; RES = ROOT / "results"
SEEDS = [123, 2024, 42, 7, 99]
CLASSIFIERS = {
    "Logistic regression": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    "Random forest": lambda: make_pipeline(
        StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
    "K-nearest neighbors": lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=10, n_jobs=-1)),
    "RBF support vector machine": lambda: make_pipeline(
        StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True)),
}


def load(dataset):
    X, y, H, shared = load_dataset(dataset)
    fcgr = X.reshape(len(X), -1).astype(np.float32)
    pf = "cherry_protein_features.npz" if dataset == "cherry" else "deephost_protein_features.npz"
    prot = np.load(PROC / pf, allow_pickle=True)["features"].astype(np.float32)
    return {"FCGR": fcgr, "Protein": prot}, y, shared


def metrics_from_proba(proba, classes_, labels):
    # map classifier-local column indices to global class ids (some classes may be absent in train)
    order = classes_[np.argsort(-proba, axis=1)]
    rank = np.where(order == labels[:, None])[1]
    top1 = float((rank == 0).mean())
    top5 = float((rank < 5).mean())
    mrr = float((1.0 / (rank + 1)).mean())
    pred = order[:, 0]
    f1 = float(f1_score(labels, pred, average="macro", zero_division=0))
    return top1, top5, mrr, f1


def split_for(dataset, split, seed, n):
    if split == "random":
        return random_split(n, seed)
    name = "cherry" if dataset == "cherry" else "deephost_std"
    return get_similarity_split(name, seed, 0.05)


records = []
for dataset in ("cherry", "deephost"):
    feats, y, shared = load(dataset)
    for split in ("random", "similarity"):
        for feature_name, feature in feats.items():
            for cname, cfn in CLASSIFIERS.items():
                runs = []
                for seed in SEEDS:
                    tr, va, te = split_for(dataset, split, seed, len(y))
                    clf = cfn()
                    clf.fit(feature[tr], y[tr])
                    proba = clf.predict_proba(feature[te])
                    runs.append(metrics_from_proba(proba, clf.classes_, y[te]))
                rec = {
                    "dataset": dataset, "split": split, "feature": feature_name,
                    "classifier": cname, "n_seeds": len(SEEDS),
                    "top1_mean": float(np.mean([r[0] for r in runs])),
                    "top1_std": float(np.std([r[0] for r in runs], ddof=1)),
                    "top5_mean": float(np.mean([r[1] for r in runs])),
                    "mrr_mean": float(np.mean([r[2] for r in runs])),
                    "macro_f1_mean": float(np.mean([r[3] for r in runs])),
                }
                records.append(rec)
                print(f"{dataset:9s} {split:10s} {feature_name:7s} {cname:28s} "
                      f"top1 {rec['top1_mean']:.4f} +/- {rec['top1_std']:.4f}", flush=True)

(RES / "ml_baselines.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
import pandas as pd
pd.DataFrame(records).to_csv(RES / "ml_baselines_summary.csv", index=False)
print("saved ml_baselines.json and summary")
