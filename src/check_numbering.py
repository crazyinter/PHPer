# -*- coding: utf-8 -*-
"""Audit figure/table/section numbering and cross-references in the built docx."""
import re
from pathlib import Path

from docx import Document

PATH = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\manuscript"
            r"\Leakage_and_protein_signals_phage_host_manuscript.docx")
doc = Document(PATH)
texts = [p.text for p in doc.paragraphs]

fig_def = sorted(int(m) for t in texts for m in re.findall(r"Figure (\d+)\.", t))
tab_def = sorted(int(m) for t in texts for m in re.findall(r"Table (\d+)\.", t))
fig_ref = sorted(set(int(m) for t in texts for m in re.findall(r"Figure (\d+)(?!\.)", t)))
tab_ref = sorted(set(int(m) for t in texts for m in re.findall(r"Table (\d+)(?!\.)", t)))
sec = sorted(set(re.findall(r"\b[23]\.(\d)\b", " ".join(texts))))
h2_def = sorted(m for t in texts for m in re.findall(r"^([23]\.\d) ", t))


def report(name, defined, expected):
    exp = list(range(1, expected + 1))
    missing = [x for x in exp if x not in defined]
    extra = [x for x in defined if x not in exp]
    print(f"{name}: defined={defined}")
    print(f"  -> {'OK' if not missing and not extra else 'PROBLEM'} missing={missing} extra={extra}")


report("Figure definitions", fig_def, 8)
report("Table definitions", tab_def, 12)
print("Figure references in text:", fig_ref, "| all defined:", all(r in fig_def for r in fig_ref))
print("Table references in text:", tab_ref, "| all defined:", all(r in tab_def for r in tab_ref))
print("Heading-2 sections defined:", h2_def)
print("Section cross-references:", sorted(set(f"2.{x}" for x in sec) | set(f"3.{x}" for x in sec)))
# sanity: no orphan references to old figure file names
old = [t for t in texts if "split_comparison" in t or "fusion_comparison" in t or "deephost_comparison" in t]
print("Stale old-figure references:", old if old else "none")
