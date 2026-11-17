# -*- coding: utf-8 -*-
"""Build MinHash sketches for the DeepHost standard phage set."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from splits import build_sketches

t0 = time.time()
sketches = build_sketches("deephost_std")
print("sketches:", len(sketches), "median size",
      __import__("numpy").median([len(s) for s in sketches]),
      "seconds:", round(time.time() - t0, 1))
