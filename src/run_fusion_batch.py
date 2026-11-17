import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
for variant in ("fcgr", "protein", "fusion"):
    for split in ("random", "similarity"):
        for seed in (123, 42, 2024, 7, 99):
            tag = f"cherry_{split}_{variant}_fusion_seed{seed}"
            if (RES / f"{tag}.json").exists():
                print("skip", tag); continue
            cmd = [sys.executable, str(ROOT / "src" / "train_fusion.py"),
                   "--variant", variant, "--split", split, "--seed", str(seed)]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if r.returncode == 0:
                print("done", tag, flush=True)
            else:
                print("FAIL", tag, r.stderr[-800:], flush=True)
