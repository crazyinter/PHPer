# -*- coding: utf-8 -*-
"""Run all DeepHost FCGR/protein/fusion x random/similarity x 5 seeds serially."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
PY = r"C:\Users\Admin\.conda\envs\torch\python.exe"
RES = ROOT / "results"
SEEDS = [123, 2024, 42, 7, 99]
VARIANTS = ["fcgr", "protein", "fusion"]
SPLITS = ["random", "similarity"]


def main():
    jobs = [(v, s, seed) for s in SPLITS for v in VARIANTS for seed in SEEDS]
    total = len(jobs)
    for k, (v, split, seed) in enumerate(jobs):
        tag = f"deephost_{split}_{v}_seed{seed}"
        out = RES / f"{tag}.json"
        if out.exists():
            print(f"[{k+1}/{total}] skip {tag} (exists)", flush=True)
            continue
        t0 = time.time()
        p = subprocess.run([PY, "train_deephost_fusion.py", "--variant", v,
                            "--split", split, "--seed", str(seed), "--epochs", "100"],
                           cwd=ROOT / "src", capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode != 0:
            print(f"[{k+1}/{total}] FAIL {tag}\n{p.stdout[-1500:]}\n{p.stderr[-2000:]}",
                  flush=True)
            sys.exit(1)
        print(f"[{k+1}/{total}] done {tag} in {time.time()-t0:.0f}s", flush=True)
    print("ALL DEEPHOST RUNS COMPLETE")


if __name__ == "__main__":
    main()
