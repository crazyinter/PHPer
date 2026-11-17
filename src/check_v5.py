# -*- coding: utf-8 -*-
import re
from pathlib import Path
from docx import Document

p = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\manuscript\Leakage_and_protein_signals_phage_host_manuscript.docx")
doc = Document(p)
texts = [para.text for para in doc.paragraphs]
full = "\n".join(texts)

# 1. leftover markers / placeholders
bad = [t for t in texts if "CONTINUE" in t or ">>> " in t]
print("leftover markers:", bad)

# 2. defined captions
table_nums = sorted({int(m) for t in texts for m in re.findall(r"Table\s+(\d+)\.", t)})
fig_nums = sorted({int(m) for t in texts for m in re.findall(r"Figure\s+(\d+)\.", t)})
supp_fig = sorted({m for t in texts for m in re.findall(r"Supplementary Figure\s+(S\d+)\.", t)})
print("table captions:", table_nums)
print("figure captions:", fig_nums)
print("supp figures:", supp_fig)

# 3. all in-text references (Table N / Figure N), including non-caption mentions
ref_tables = {int(m) for t in texts for m in re.findall(r"Table\s+(\d+)", t)}
ref_figs = {int(m) for t in texts for m in re.findall(r"Figure\s+(\d+)", t)}
print("table refs missing caption:", sorted(ref_tables - set(table_nums)))
print("figure refs missing caption:", sorted(ref_figs - set(fig_nums)))

# 4. continuity
def contiguous(nums):
    return nums == list(range(1, len(nums) + 1))
print("tables contiguous:", contiguous(table_nums), "| figures contiguous:", contiguous(fig_nums))

# 5. section headings 1..5
heads = [t for t in texts if re.match(r"^[1-5]\. ", t)]
print("numbered H1:", heads)

# 6. table count matches
print("physical tables:", len(doc.tables), "caption tables:", len(table_nums))

# 7. terminology consistency: forbidden prose names
for term in ["Nd-Mamba2", "BiMamba2", "Nd Mamba"]:
    hits = [t for t in texts if term in t]
    print(f"forbidden term {term!r}:", len(hits))
print("BMKMPHI mentions:", len([t for t in texts if "BMKMPHI" in t]))
print("bidirectional Mamba2 mentions:", len([t for t in texts if "bidirectional Mamba2" in t]))
