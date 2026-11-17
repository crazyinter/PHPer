"""Parameter count and inference latency for each architecture."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from model_bmk import build_model
from models_ext import CNNOnly, PhageCGRCNN

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
device = "cuda" if torch.cuda.is_available() else "cpu"


def measure(name, model, sample_shape, n_warm=10, n_iter=100):
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(sample_shape, device=device)
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_iter):
                model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        lat = (time.time() - t0) / n_iter * 1000
    return {"model": name, "parameters": n_params, "latency_ms_per_batch": round(lat, 3)}


def main():
    out = [
        measure("BMKMPHI (CNN+BiMamba2)", build_model(6), (32, 1, 64, 64)),
        measure("CNN-only", CNNOnly(7), (32, 1, 64, 64)),
        measure("PhageCGRNet CNN", PhageCGRCNN(206), (32, 3, 128, 128)),
    ]
    for r in out:
        print(r)
    (RES / "model_efficiency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
