# -*- coding: utf-8 -*-
"""Build the complete provenance: run id -> training script, config, split source, output JSON, checkpoint."""
import csv
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
CKPT = ROOT / "models"

rows = []
for f in sorted(RES.glob("*.json")):
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if "test_top1" not in d:
        continue
    run = f.stem
    loss = d.get("loss", "")
    variant = d.get("variant", "")
    split_type = d.get("split_type", "")
    seed = d.get("seed", "")

    # training script
    if "phagecgrnet" in run:
        script = "train_phagecgrnet.py"
    elif variant:
        script = "train_fusion.py" if d.get("dataset") == "cherry" else "train_deephost_fusion.py"
    else:
        script = "train.py"

    # config file
    if loss == "ce" or run.endswith("_ce") or "_ce_" in run:
        config = "pipeline_ce.json"
    else:
        config = "pipeline_contrastive.json"

    # split source
    if split_type == "similarity":
        split_source = f"splits.py get_similarity_split(seed={seed}, threshold=0.05)"
    else:
        split_source = f"train.py random_split(seed={seed}, 8:1:1)"

    ckpt_path = CKPT / f"{run}.pt"
    rows.append({
        "run_id": run,
        "training_script": script,
        "config_file": config,
        "split_source": split_source,
        "output_json": f"results/{run}.json",
        "checkpoint": f"models/{run}.pt" if ckpt_path.exists() else "not retained",
        "dataset": d.get("dataset", ""),
        "split_type": split_type,
        "seed": seed,
        "train_val_test": "/".join(str(x) for x in d.get("split", [])),
        "test_top1": round(d["test_top1"], 4),
    })

cols = ["run_id", "training_script", "config_file", "split_source", "output_json",
        "checkpoint", "dataset", "split_type", "seed", "train_val_test", "test_top1"]
with open(RES / "full_provenance.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print("full provenance rows:", len(rows))
missing_ckpt = [r["run_id"] for r in rows if r["checkpoint"] == "not retained"]
print("runs without retained checkpoint:", len(missing_ckpt))
for r in rows[:6]:
    print(r["run_id"], "|", r["config_file"], "|", r["split_source"], "|", r["checkpoint"])
