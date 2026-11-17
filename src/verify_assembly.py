# -*- coding: utf-8 -*-
import json
from collections import Counter
from pathlib import Path

p = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\results\host_assembly_table.json")
d = json.loads(p.read_text(encoding="utf-8"))
recs = d["records"]
print("records listed:", len(recs))
levels = Counter(r.get("assembly_level", "(missing)") for r in recs)
print("assembly_level counts:", dict(levels))
multi = [r for r in recs if r.get("n_contig", 1) > 1]
print("n_contig>1:", len(multi))
for r in multi:
    print("  MULTI:", r["class_id"], r["species"], r["assembly_level"], r["n_contig"])
# non-complete records
noncomp = [r for r in recs if r.get("assembly_level") != "Complete genome"]
print("non-complete records:", len(noncomp))
for r in noncomp:
    print("  NONCOMP:", r["class_id"], r["species"], "| level:", r.get("assembly_level"),
          "| contig:", r.get("n_contig"), "| len:", r.get("total_length"))
