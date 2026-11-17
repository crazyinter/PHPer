"""Configuration-driven batch runner with resume support."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
CORE_SEEDS = [123, 42, 2024, 7, 99]
ABL_SEEDS = [123, 42, 2024]
EPOCHS = 120


def build_configs():
    configs = []
    # core: full model, random vs similarity split, 5 seeds
    for split in ("random", "similarity"):
        configs.append((split, "cnn_mamba", "fcgr", "contrastive", CORE_SEEDS))
    # ablations, 3 seeds
    for split in ("random", "similarity"):
        configs.append((split, "cnn", "fcgr", "contrastive", ABL_SEEDS))        # CNN only
        configs.append((split, "cnn_mamba", "random", "contrastive", ABL_SEEDS))  # random rep
        configs.append((split, "cnn_mamba", "fcgr", "ce", ABL_SEEDS))           # cross-entropy
    return configs


def tag(dataset, split, encoder, rep, loss, seed):
    return f"{dataset}_{split}_{encoder}_{rep}_{loss}_seed{seed}"


def main(dataset="cherry"):
    total = 0
    for split, encoder, rep, loss, seeds in build_configs():
        for seed in seeds:
            total += 1
            t = tag(dataset, split, encoder, rep, loss, seed)
            if (RES / f"{t}.json").exists():
                print(f"skip {t}", flush=True)
                continue
            cmd = [sys.executable, str(ROOT / "src" / "train.py"),
                   "--dataset", dataset, "--seed", str(seed), "--epochs", str(EPOCHS),
                   "--split", split, "--encoder", encoder,
                   "--representation", rep, "--loss", loss]
            print("RUN", t, flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                print("FAILED", t, r.stderr[-1500:], flush=True)
            else:
                print(r.stdout[-400:], flush=True)
    print("all", total, "slots processed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cherry")
