# -*- coding: utf-8 -*-
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

p = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro\manuscript\Leakage_and_protein_signals_phage_host_manuscript.docx")
doc = Document(p)

# 1. real Heading styles
h1 = [par.text for par in doc.paragraphs if par.style.name == "Heading 1"]
h2 = [par.text for par in doc.paragraphs if par.style.name == "Heading 2"]
print("Heading 1 count:", len(h1))
for t in h1:
    print("  H1:", t)
print("Heading 2 count:", len(h2))

# 2. image alt text (docPr title/descr)
body = doc.element.body
docPrs = body.findall(".//" + qn("wp:docPr"))
missing_alt = [d.get("name") for d in docPrs if not (d.get("title") and d.get("descr"))]
print("embedded images:", len(docPrs), "| missing alt:", missing_alt)

# 3. table header repeat and cantSplit
no_header = []
split_rows = 0
total_rows = 0
for ti, tbl in enumerate(doc.tables, start=1):
    first = tbl.rows[0]._tr
    trPr = first.find(qn("w:trPr"))
    has_header = trPr is not None and trPr.find(qn("w:tblHeader")) is not None
    if not has_header:
        no_header.append(ti)
    for row in tbl.rows:
        total_rows += 1
        trPr = row._tr.find(qn("w:trPr"))
        if trPr is None or trPr.find(qn("w:cantSplit")) is None:
            split_rows += 1
print("tables:", len(doc.tables), "| first row missing tblHeader:", no_header)
print("rows:", total_rows, "| rows missing cantSplit:", split_rows)

# 4. stale wording that should be gone
joined = "\n".join(par.text for par in doc.paragraphs)
for stale in ["to be inserted at final typesetting", "full expansion to be provided",
              "will be deposited in a public archive", "significantly better than a permuted"]:
    print(f"stale phrase {stale!r}:", joined.count(stale))
