import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
for split in ("random", "similarity"):
    for seed in (123, 42, 2024):
        tag = f"cherry_{split}_phagecgrnet_cgr3d_ce_seed{seed}"
        if (RES / f"{tag}.json").exists():
            print("skip", tag); continue
        cmd = [sys.executable, str(ROOT / "src" / "train_phagecgrnet.py"),
               "--seed", str(seed), "--split", split]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        print(r.stdout[-300:] if r.returncode == 0 else r.stderr[-1000:], flush=True)
