# -*- coding: utf-8 -*-
"""Auditable CHERRY curation flow: step-by-step counts, empty/multi-label handling."""
import json
import sys
from collections import Counter
from pathlib import Path

import xlrd

sys.path.insert(0, str(Path(__file__).parent))
from dataset import parse_fasta_file, base_id  # noqa: E402

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
C = ROOT / "data" / "raw" / "CHERRY-main"
OUT = ROOT / "results" / "cherry_curation_audit.json"


def load_pairs(xls_name, sheet_name):
    book = xlrd.open_workbook(str(C / "Interactiondata" / xls_name))
    sh = book.sheet_by_name(sheet_name)
    rows = []
    for r in range(2, sh.nrows):
        rows.append({"acc": str(sh.cell_value(r, 0)).strip(),
                     "species": str(sh.cell_value(r, 15)).strip(),
                     "genus": str(sh.cell_value(r, 14)).strip()})
    return rows


pairs = load_pairs("VHM_PAIR_TAX.xls", "PAIR1426") + load_pairs("TEST_PAIR_TAX.xls", "PAIR671")
seq_map = {base_id(h): s for h, s in parse_fasta_file(C / "dataset" / "nucl.fasta")}

empty_species = [p for p in pairs if not p["species"]]
empty_with_genus = [p for p in empty_species if p["genus"]]
missing_seq = [p for p in pairs if base_id(p["acc"]) not in seq_map]
retained = [p for p in pairs if base_id(p["acc"]) in seq_map]

# multi-label: one phage accession with more than one recorded host species
by_acc = {}
for p in retained:
    by_acc.setdefault(base_id(p["acc"]), set()).add(p["species"])
multi = {a: sp for a, sp in by_acc.items() if len(sp) > 1}

final_species = Counter(p["species"] for p in retained if p["species"])
per_class = sorted(final_species.items(), key=lambda kv: -kv[1])

audit = {
    "raw_pairs": len(pairs),
    "pairs_missing_sequence": len(missing_seq),
    "pairs_with_empty_species": len(empty_species),
    "empty_species_with_genus_label": len(empty_with_genus),
    "retained_pairs_with_sequence": len(retained),
    "unique_retained_phages": len(by_acc),
    "multilabel_phages": len(multi),
    "final_occupied_host_species": len(final_species),
    "host_classes_with_ge5": sum(1 for _, n in per_class if n >= 5),
    "singleton_host_classes": sum(1 for _, n in per_class if n == 1),
    "rule": "Only species-level labels are used; empty-species and genus-only records never enter "
            "training as positives. No genus fallback is applied.",
    "largest_classes": per_class[:8],
}
OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(audit, indent=2, ensure_ascii=False))
