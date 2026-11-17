# -*- coding: utf-8 -*-
"""Efficiency protocol with hardware, repeated measurements (mean +/- SD), and scope."""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from model_bmk import build_model
from models_ext import CNNOnly, PhageCGRCNN

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
RES = ROOT / "results"
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH, N_WARM, N_ITER, N_REPEAT = 32, 10, 100, 5


def measure(name, model, sample_shape):
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(sample_shape, device=device)
    with torch.no_grad():
        for _ in range(N_WARM):
            model(x)
        times = []
        for _ in range(N_REPEAT):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(N_ITER):
                model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0) / N_ITER * 1000)
    return {"model": name, "parameters": n_params,
            "latency_ms_mean": round(float(np.mean(times)), 3),
            "latency_ms_sd": round(float(np.std(times)), 3)}


hw = {"device": device,
      "gpu": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
      "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
      "precision": "FP32 (no AMP)", "batch_size": BATCH, "warmup_iters": N_WARM,
      "timed_iters": N_ITER, "independent_repeats": N_REPEAT,
      "scope": "Forward pass only; excludes sequence loading, ORF prediction, FCGR construction "
               "and host-database I/O. GPU synchronized around each timing window."}
out = {"protocol": hw,
       "models": [
           measure("BMKMPHI (CNN+BiMamba2)", build_model(6), (32, 1, 64, 64)),
           measure("CNN-only", CNNOnly(7), (32, 1, 64, 64)),
           measure("PhageCGRNet CNN", PhageCGRCNN(206), (32, 3, 128, 128))]}
(RES / "efficiency_protocol.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
