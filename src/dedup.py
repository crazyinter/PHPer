from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_labeled  # noqa: E402

matched, missing = build_labeled()
print("matched before dedup:", len(matched))

# exact-string dedup (paper: removing identical sequences)
seen = {}
for row in matched:
    key = row["seq"]
    if key not in seen:
        seen[key] = row
dedup = list(seen.values())
print("after exact dedup:", len(dedup))

sp = Counter(r["species"] for r in dedup)
ge = Counter(r["genus"] for r in dedup)
print("unique host species:", len(sp))
print("unique host genera:", len(ge))

# restrict to the 118 DeepHost species classes and recount
RAW = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\data\raw")
classes = set()
with open(RAW / "DeepHost-master" / "data" / "species.txt", encoding="utf-8") as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) == 2:
            classes.add(p[1])
in_cls = [r for r in dedup if r["species"] in classes]
print("\ndedup rows in DeepHost species classes:", len(in_cls))
print("unique species among them:", len(set(r["species"] for r in in_cls)))

# per-class size distribution (helps judge trainability)
sizes = Counter(r["species"] for r in in_cls)
small = sorted(sizes.items(), key=lambda x: x[1])[:10]
print("smallest classes:", small)
print("largest classes:", sorted(sizes.items(), key=lambda x: -x[1])[:10])
