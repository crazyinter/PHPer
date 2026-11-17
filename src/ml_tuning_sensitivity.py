# -*- coding: utf-8 -*-
"""Uniform validation-set hyperparameter tuning for classical ML; default vs tuned comparison."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from train import load_dataset, random_split
from splits import get_similarity_split

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PROC = ROOT / "data" / "processed"; RES = ROOT / "results"
SEED = 123

GRIDS = {
    "Logistic regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "Random forest": [{"n_estimators": n, "max_depth": d}
                      for n in (100, 300) for d in (None, 20)],
    "K-nearest neighbors": [{"n_neighbors": k} for k in (3, 10, 30)],
    "RBF support vector machine": [{"C": c, "gamma": g}
                                   for c in (0.1, 1.0, 10.0) for g in ("scale", 0.01)],
}
DEFAULT = {
    "Logistic regression": {"C": 1.0},
    "Random forest": {"n_estimators": 300, "max_depth": None},
    "K-nearest neighbors": {"n_neighbors": 10},
    "RBF support vector machine": {"C": 1.0, "gamma": "scale"},
}


def make_clf(name, params, probability):
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **params))
    if name == "Random forest":
        return make_pipeline(StandardScaler(),
                             RandomForestClassifier(random_state=0, n_jobs=-1, **params))
    if name == "K-nearest neighbors":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_jobs=-1, **params))
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", probability=probability, **params))


def score_proba(clf, X, classes_, labels):
    proba = clf.predict_proba(X)
    order = classes_[np.argsort(-proba, axis=1)]
    rank = np.where(order == labels[:, None])[1]
    return {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
            "mrr": float((1.0 / (rank + 1)).mean()),
            "macro_f1": float(f1_score(labels, order[:, 0], average="macro", zero_division=0))}


def load(dataset):
    X, y, H, shared = load_dataset(dataset)
    fcgr = X.reshape(len(X), -1).astype(np.float32)
    pf = "cherry_protein_features.npz" if dataset == "cherry" else "deephost_protein_features.npz"
    prot = np.load(PROC / pf, allow_pickle=True)["features"].astype(np.float32)
    return {"FCGR": fcgr, "Protein": prot}, y


out = []
for dataset in ("cherry", "deephost"):
    feats, y = load(dataset)
    for split in ("random", "similarity"):
        if split == "random":
            tr, va, te = random_split(len(y), SEED)
        else:
            sname = "cherry" if dataset == "cherry" else "deephost_std"
            tr, va, te = get_similarity_split(sname, SEED, 0.05)
        for fname, F in feats.items():
            for cname in GRIDS:
                # grid selection on validation (predict-based, no probability)
                best, best_va = None, -1.0
                for params in GRIDS[cname]:
                    c = make_clf(cname, params, False)
                    c.fit(F[tr], y[tr])
                    va_top1 = float((c.predict(F[va]) == y[va]).mean())
                    if va_top1 > best_va:
                        best_va, best = va_top1, params
                # final default vs tuned on test, with probability-based ranking
                c_default = make_clf(cname, DEFAULT[cname], True)
                c_default.fit(F[tr], y[tr])
                m_default = score_proba(c_default, F[te], c_default.classes_, y[te])
                c_tuned = make_clf(cname, best, True)
                c_tuned.fit(F[tr], y[tr])
                m_tuned = score_proba(c_tuned, F[te], c_tuned.classes_, y[te])
                rec = {"dataset": dataset, "split": split, "feature": fname,
                       "classifier": cname, "selected_params": best,
                       "default_top1": round(m_default["top1"], 4),
                       "tuned_top1": round(m_tuned["top1"], 4),
                       "delta_top1": round(m_tuned["top1"] - m_default["top1"], 4),
                       "tuned_top5": round(m_tuned["top5"], 4),
                       "tuned_mrr": round(m_tuned["mrr"], 4),
                       "tuned_macro_f1": round(m_tuned["macro_f1"], 4)}
                out.append(rec)
                print(f"{dataset:8s} {split:10s} {fname:7s} {cname:28s} "
                      f"default {rec['default_top1']} tuned {rec['tuned_top1']} "
                      f"d {rec['delta_top1']}", flush=True)

(RES / "ml_tuning_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
deltas = np.array([r["delta_top1"] for r in out])
print("\nsaved. mean |delta top1| =", round(float(np.abs(deltas).mean()), 4),
      "max =", round(float(np.abs(deltas).max()), 4),
      "n improved =", int((deltas > 0).sum()), "n worsened =", int((deltas < 0).sum()))
