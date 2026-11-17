# -*- coding: utf-8 -*-
"""Build the third-round reviewer-revised manuscript (v5) as a formatted, editable Word file."""
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
FIG = ROOT / "figures"
OUT = ROOT / "manuscript" / "Leakage_and_protein_signals_phage_host_manuscript.docx"
OUT.parent.mkdir(exist_ok=True)
BLACK = RGBColor(0, 0, 0)


def set_font(run, size=11, bold=False, italic=False,
             cn="Times New Roman", en="Times New Roman"):
    run.font.name = en; run.font.size = Pt(size)
    run.font.bold = bold; run.font.italic = italic
    run.font.color.rgb = BLACK
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cn)


def emit_runs(p, text, size, bold=False):
    markers = {"{i}": ("i", True), "{/i}": ("i", False),
               "{sub}": ("sub", True), "{/sub}": ("sub", False)}
    pos = 0; italic = False; sub = False; buf = ""

    def flush():
        nonlocal buf
        if buf:
            r = p.add_run(buf); set_font(r, size=size, bold=bold, italic=italic)
            r.font.subscript = sub; buf = ""

    while pos < len(text):
        hit = None
        for mk, (typ, val) in markers.items():
            if text.startswith(mk, pos):
                hit = (mk, typ, val); break
        if hit:
            flush(); mk, typ, val = hit
            if typ == "i":
                italic = val
            else:
                sub = val
            pos += len(mk)
        else:
            buf += text[pos]; pos += 1
    flush()


def body(doc, text, size=11, justify=True):
    p = doc.add_paragraph(); pf = p.paragraph_format
    pf.line_spacing = 1.5; pf.space_before = Pt(2); pf.space_after = Pt(2)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    emit_runs(p, text, size)
    return p


def formula(doc, text, num=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format; pf.line_spacing = 1.3; pf.space_before = Pt(4); pf.space_after = Pt(4)
    emit_runs(p, text, 11)
    if num:
        set_font(p.add_run(f"     ({num})"), size=11)
    return p


def h1(doc, num, text):
    p = doc.add_paragraph(style="Heading 1"); pf = p.paragraph_format
    pf.space_before = Pt(12); pf.space_after = Pt(4); pf.keep_with_next = True
    set_font(p.add_run(f"{num}. {text}" if num else text), size=13, bold=True)


def h2(doc, num, text):
    p = doc.add_paragraph(style="Heading 2"); pf = p.paragraph_format
    pf.space_before = Pt(8); pf.space_after = Pt(3); pf.keep_with_next = True
    set_font(p.add_run(f"{num} {text}"), size=11.5, bold=True)


def caption(doc, text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format; pf.space_before = Pt(3); pf.space_after = Pt(8)
    emit_runs(p, text, 10, bold=True)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), color); tcPr.append(sh)


def fill(cell, text, header=False, align="center"):
    cell.vertical_alignment = 1
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if align == "center" else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(1)
    set_font(p.add_run(text), size=9.5, bold=header)
    if header:
        shade(cell, "D9D9D9")


def mark_row(row, header=False):
    trPr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit"); cant.set(qn("w:val"), "true"); trPr.append(cant)
    if header:
        th = OxmlElement("w:tblHeader"); th.set(qn("w:val"), "true"); trPr.append(th)


def make_table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    mark_row(t.rows[0], header=True)
    for j, hh in enumerate(headers):
        fill(t.rows[0].cells[j], hh, True)
    for r in rows:
        row = t.add_row(); mark_row(row)
        cells = row.cells
        for j, v in enumerate(r):
            fill(cells[j], v, align=("left" if j == 0 and len(v) > 14 else "center"))
    return t


def add_figure(doc, path, width_cm=15.5, alt=""):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(2)
    run = p.add_run(); run.add_picture(str(path), width=Cm(width_cm))
    if alt:
        docPr = run._element.find(qn("w:drawing")).find(".//" + qn("wp:docPr"))
        docPr.set("title", alt); docPr.set("descr", alt)


def pc(mean, sd):
    return f"{mean*100:.2f} ± {sd*100:.2f}"


doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.2)
sec.left_margin = sec.right_margin = Cm(2.2)

# ---- Title ----
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.LEFT
p.paragraph_format.space_after = Pt(8)
set_font(p.add_run("Neighbor leakage and ORF-derived coding signals in alignment-free "
                   "phage-host prediction: a controlled benchmark"), size=16, bold=True)

# ---- Authors / affiliations ----
p = doc.add_paragraph(); set_font(p.add_run(
    "Yuan Hou(1), Haoran Gu(3), Xingshun Song(2,*), Yan Miao(3,*)"), size=11)
for aff in [
    "(1) Aulin College, Northeast Forestry University, Harbin, Heilongjiang, China",
    "(2) College of Life Science, Northeast Forestry University, Harbin, Heilongjiang, China",
    "(3) School of Computer Science and Artificial Intelligence, Northeast Forestry University, Harbin, Heilongjiang, China",
]:
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(0)
    set_font(p.add_run(aff), size=9.5)
p = doc.add_paragraph(); set_font(p.add_run(
    "* Correspondence: Yan Miao and Xingshun Song. [Corresponding institutional email addresses to "
    "be supplied by the authors before submission.]"), size=9.5, italic=True)

# ---- Abstract ----
h1(doc, "", "Abstract")
abstract = ("Motivation: Alignment-free models rank bacterial hosts from phage nucleotide sequences, but their "
"evidence is weakened by sequence-level splits that place near-identical phages in both training and test sets and by "
"single-run comparisons. Results: We curate two benchmarks from public resources, CHERRY (1,875 phages, 206 hosts) "
"and DeepHost (2,845 phages, 108 hosts), and compare deep sequence models, four classical classifiers, and an "
"ORF-derived coding-feature encoder under a random and a similarity-aware split, over five seeds. The data partition "
"is the largest effect: the full model falls from 74.68% to 44.92% top-1 accuracy (29.76 points; Holm-adjusted p = "
"0.0033). A logistic regression on the same FCGR features reaches 82.3% and 83.6%, matching or exceeding every deep "
"model. A phage-level paired bootstrap and sign-flip permutation show that the architectural components do not give a "
"stable gain: the bidirectional Mamba2 block does not improve ranking on a fixed split, while in an exploratory "
"fixed-split analysis the genuine FCGR signal is higher than a permuted representation under the similarity split "
"(unadjusted permutation p = 0.043). A uniform "
"validation-set grid search confirms that the published classical-model defaults are already adequate for logistic "
"regression, random forest, and the support vector machine (mean absolute change 2.29 points), with only k-nearest "
"neighbors sensitive to its neighbor count. ORF-derived coding features outperform FCGR; the strongest contrasts "
"survive family-wise correction across a single, explicitly defined family of 13 tests (Holm p as low as 0.0082), "
"whereas borderline contrasts hold under false-discovery-rate control but not under the stricter Holm procedure. This "
"signal is not explained by GC content, sequence length, or gene count: codon frequencies retain their accuracy after "
"residualization against GC, while a GC/length/gene-count baseline reaches only 39%. Host-phylogeny controls that use "
"the real genus label are kept strictly post-hoc and never enter the deployed predictor, which uses only "
"sequence-derived features. Because the two resources overlap substantially (767 shared phage accessions; 85.5% of "
"CHERRY phages have a cosine >=0.999 DeepHost neighbor) and cross-dataset accuracy collapses after sequence "
"de-duplication (91.1% to 19.2% in the strongest direction), we do not treat the second dataset as independent "
"external validation. Temperature scaling lowers mean expected calibration error from 0.729 to 0.067, and a calibrated "
"confidence detects withheld hosts (AUROC 0.91-0.94). We describe codon usage as predictive and consistent with, but "
"not proof of, translational adaptation. Availability: the code, processed data, and per-run results are provided in "
"the reproducibility package and in the public repository https://github.com/QWERasd12345-ASD/BMKMPHI.")
body(doc, abstract)
p = doc.add_paragraph(); set_font(p.add_run(
    "Keywords: phage-host prediction; data leakage; similarity-aware split; classical machine-learning "
    "baselines; ORF-derived coding features; codon usage; confidence calibration"), size=10, bold=True)

# ---- 1 Introduction ----
h1(doc, "1", "Introduction")
body(doc, "Bacteriophages are viruses that infect bacteria and are among the most abundant biological entities in "
"microbial communities. Their host ranges are typically narrow, and infection can reshape bacterial population "
"structure, horizontal gene transfer, and processes linked to disease. Assigning a bacterial host to a phage sequence "
"recovered from a metagenomic sample is therefore a central step in viral ecology and in phage-based applications.")
body(doc, "Experimental host assignment through culturing is low-throughput and expensive, so computational methods "
"dominate at scale. Alignment-based tools exploit homology, and CRISPR-spacer or receptor-binding protein approaches "
"use specific molecular evidence, but these depend on reference completeness or on a recognizable immune record. "
"Alignment-free methods instead encode sequence composition-k-mer frequencies, oligonucleotide usage, or frequency "
"chaos game representation (FCGR) [7]-and learn a mapping from phage to host, often with a contrastive or "
"classification objective [8].")
body(doc, "Two methodological problems weaken the evidence for many alignment-free models. First, benchmarks commonly "
"split individual sequences randomly after removing only exact duplicates. Closely related phages, and fragments "
"derived from the same genome, can then occur in both training and test sets, and a model can succeed by retrieving a "
"near neighbor rather than by learning a generalizable host signal. Second, architectural contributions are often "
"demonstrated with single runs and point estimates, without repeated seeds, confidence intervals, paired significance "
"tests, or multiple-comparison correction, and with baselines reimplemented under unequal settings. It is then "
"difficult to distinguish genuine advances from variation driven by initialization, negative sampling, or the "
"partition itself.")
body(doc, "These concerns are not hypothetical. In the FCGR pipeline that motivates this work, whole-genome gains over "
"the closest competing method are small, and the difference between FCGR and a randomly permuted k-mer layout is "
"typically below one percentage point, suggesting that k-mer composition rather than two-dimensional topology carries "
"the discriminative signal. To our knowledge, no controlled study has simultaneously quantified neighbor leakage, "
"benchmarked simple classifiers against deep architectures, tested whether a biologically different representation "
"can improve host ranking under a leakage-controlled split, and checked whether that signal is confounded by simple "
"sequence composition.")
body(doc, "We make three contributions. (1) We build two versioned, checksum-verified benchmarks and quantify neighbor "
"leakage with a similarity-aware split and a multi-seed threshold sweep, paired across models, with Holm and "
"Benjamini-Hochberg correction over a documented family. (2) We derive coding features from predicted open reading "
"frames, test whether they outperform FCGR under leakage control, and explicitly evaluate GC, length, and gene-count "
"confounding. (3) We provide multi-seed calibration, open-set rejection, per-class and imbalance analysis, and an "
"efficiency protocol, delineating both the validity and the limits of each representation.")

# ---- 2 Materials and Methods ----
h1(doc, "2", "Materials and Methods")

h2(doc, "2.1", "Benchmark datasets and curation audit")
body(doc, "We use two public resources so that conclusions can be examined across more than one data source; Table 1 "
"summarizes their statistics. The CHERRY release [5] provides phage nucleotide sequences and matched candidate "
"bacterial genomes. We re-ran the original curation table rather than trusting downstream counts. Of 2,097 raw "
"phage-host pairs, 222 lack a usable sequence and 157 carry an empty host-species field; 142 of the empty-species "
"records still contain a genus label. We apply one explicit rule: only species-level labels are used, and "
"empty-species or genus-only records never enter training as positives, so no genus fallback is applied. This retains "
"1,875 phages mapped to 206 host species across 135 genera, with zero multi-label phages, 88 singleton classes, and "
"59 classes containing at least five phages. The largest class is {i}Mycolicibacterium smegmatis{/i} (312 phages), "
"followed by {i}Escherichia coli{/i} (226) and {i}Pseudomonas aeruginosa{/i} (117). Taxonomy is updated to current "
"names, including the reclassification of {i}Mycobacterium smegmatis{/i} to {i}Mycolicibacterium smegmatis{/i}.")
caption(doc, "Table 1. Summary of the two benchmark datasets.")
make_table(doc, ["Property", "CHERRY", "DeepHost"], [
    ["Raw phage-host pairs", "2,097", "-"],
    ["Pairs missing sequence", "222", "-"],
    ["Empty-species entries", "157", "-"],
    ["Phage sequences retained", "1,875", "2,845"],
    ["Candidate host classes", "206", "108"],
    ["Reference taxonomy classes", "206", "118 (10 empty)"],
    ["Singleton classes", "88", "-"],
    ["Host classes with >=5 phages", "59", "-"],
    ["Largest host class (phages)", "M. smegmatis (312)", "-"],
])
body(doc, "The DeepHost reference taxonomy [6] defines 118 standardized host species. Its processed release contains "
"3,567 phage sequences annotated against 366 raw host-name strings. We map each raw name to the standardized taxonomy "
"and retain only phages whose host matches a standardized class exactly, yielding 2,845 phages over 108 occupied "
"classes. Ten classes are empty and are excluded from the candidate set; we report them rather than silently dropping "
"them. Outdated names are updated to current taxonomy, including {i}Cutibacterium acnes{/i} and {i}Clostridioides "
"difficile{/i}.")
body(doc, "To make the data version auditable, every processed file is recorded with its byte size and SHA-256 digest "
"in a machine-readable manifest (13 files), and the host reference genomes are listed with class identifier, "
"scientific name, accession, and sequence length. The identical encoding, features, splitting, and objective are "
"applied to both datasets, so the benchmarks differ only in the underlying data.")

h2(doc, "2.2", "Sequence and ORF-derived coding representations")
body(doc, "Each nucleotide sequence is encoded with FCGR at {i}k = 6. The chaos-game walk maps the four bases to the "
"corners of a unit square and, at each position {i}t, moves halfway from the current point to the corner for the base "
"{i}s{/i}{sub}{i}t{/i}{/sub}. Coordinates are binned into a 64 x 64 grid whose 4,096 cells correspond one-to-one to "
"the 4,096 possible 6-mers. For a sequence of length {i}L, let {i}c{/i}{sub}m{/sub} be the number of length-{i}k{/i} "
"windows whose 6-mer is {i}m{/i}; windows containing an ambiguous base {i}N{/i} are not counted, while the denominator "
"remains {i}L - k + 1. The cell value is then")
formula(doc, "{i}f{/i}{sub}m{/sub} = {i}c{/i}{sub}m{/sub} / ( {i}L - k + 1 )", 1)
body(doc, "No reverse-complement merging is applied. To separate k-mer composition from the two-dimensional layout, a "
"permuted control keeps the same 4,096 frequencies but rearranges them with a fixed, reproducible permutation before "
"reshaping, so the frequency content is identical while the FCGR topology is removed.")
body(doc, "Open reading frames are predicted with pyrodigal [4], a Cython binding and interface to the Prodigal "
"gene-finder [3] that implements the Prodigal dynamic-programming algorithm rather than a pure-Python "
"reimplementation. We construct the finder with pyrodigal.GeneFinder(meta=True), running it in metagenomic mode with "
"the default bacterial translation table 11, and pin the library version in the requirements file. The finder is "
"called separately on every contig, so gene calls cannot span a contig boundary. Partial genes are retained and "
"counted; overlapping ORFs are counted independently without de-duplication. The three in-frame stop codons TAA, "
"TAG, and TGA are part of the 64 ACGT codon bins and are counted, whereas any codon containing an ambiguous base N "
"has no index in the 64-bin table and is skipped. Predicted ORFs are translated to proteins, from which we construct "
"a 96-dimensional vector in four blocks. The structural block (indices 0-7) contains log gene count, coding density, "
"mean and SD of ORF length, fraction of forward-strand ORFs, fraction of complete genes, overlap fraction, and mean "
"intergenic length. The codon block (indices 8-71) contains 64 relative codon frequencies. The amino-acid block "
"(indices 72-91) contains 20 relative amino-acid frequencies. The physicochemical block (indices 92-95) contains mean "
"Kyte-Doolittle hydrophobicity and the fractions of basic, acidic, and sulfur-containing residues. Each codon "
"frequency is the count of that codon divided by the total number of codons across all predicted ORFs in the genome, "
"so the 64 values sum to one. Coding features are standardized using the training-set mean and SD.")

h2(doc, "2.3", "Host reference genomes: selection, assembly level, and quality control")
body(doc, "DeepHost does not ship host genomes, so we obtained one pinned reference assembly per taxonomy class from "
"NCBI under an explicit, documented rule. For each species we select a single RefSeq representative, preferring "
"chromosome-sized sequences of at least 200 kb and, where several qualify, the longest such record. Whole-genome "
"shotgun entries are represented by multiple contigs; for gene calling we process each contig independently rather "
"than joining them, so no predicted ORF can cross a contig boundary. Of the 118 taxonomy classes, 108 carry a "
"sequence; 104 are complete genomes and four are contig-level assemblies, giving 104 + 4 = 108. Two of those four "
"are multi-contig, the most fragmented being {i}Streptomyces viridochromogenes{/i} (757 contigs, N50 18,965 bp) "
"and {i}Streptomyces azureus{/i} (86 contigs, N50 228,249 bp); the other two contig-level records (class 92 and "
"class 97) each consist of a single contig annotated at the contig level rather than as a complete genome.")
body(doc, "Because no contamination-model tool such as CheckM was run, we apply a transparent sequence-level screen "
"instead and report its limits. Every assembly is checked for the ambiguous-base fraction, the presence of symbols "
"outside the ACGT alphabet, and a robust z-score of ORF density. Across the 108 genomes the median density is 938.86 "
"genes per Mb. Three records are flagged rather than silently removed: {i}Vibrio parahaemolyticus{/i} (class 26; "
"13.33% ambiguous bases, 1,203 non-ACGTN symbols, 787.9 genes/Mb), {i}Lactobacillus gasseri{/i} (class 111; 7,842 "
"non-ACGTN symbols), and {i}Burkholderia thailandensis{/i} (class 115; 5.20% ambiguous bases). We retain these "
"classes with an explicit flag so that any downstream effect can be traced.")

h2(doc, "2.4", "Model architecture and the two independent training pipelines")
body(doc, "We name the proposed model BMKMPHI [full expansion to be supplied by the authors before submission]. Its "
"sequence encoder "
"applies two convolutional layers (kernel size 7, stride 2; batch normalization and ReLU; max pooling after the "
"second layer; channels 64 then 128) to the 64 x 64 matrix, followed by a bidirectional Mamba2 block over the "
"flattened spatial tokens and a fully connected head producing a 512-dimensional embedding. The bidirectional "
"Mamba2 block pads the token grid to a multiple of eight, runs one forward and one reverse Mamba2 layer, and sums "
"their two outputs before cropping back to the original grid. A CNN-only variant removes this block. The coding "
"encoder is a multilayer perceptron (96 to 256 to 256 to 512). The fusion model encodes both views, projects the "
"coding branch to 128 dimensions, concatenates it with the 512-dimensional sequence embedding, and maps the "
"640-dimensional vector back to 512. Phages and candidate hosts share the same encoder in every variant. We also "
"include an equivalent reimplementation of the published PhageCGRNet three-channel architecture [7].")
body(doc, "A single configuration file cannot represent two different training objectives, so the architecture and the "
"training target are deliberately separated. One shared architecture specification (method_config.json) fixes the "
"layers and feature blocks, and two independent pipeline files specify how the network is trained. The default "
"contrastive pipeline (pipeline_contrastive.json) attaches no classification head, trains the 512-dimensional "
"embedding with the matrix contrastive loss, ranks hosts by Euclidean distance (smaller is better), and applies "
"three BatchNorm recalibration passes before evaluation. The separate cross-entropy pipeline (pipeline_ce.json) "
"adds a Linear(512, n_hosts) head, trains with a full-host softmax cross-entropy, ranks by logits (larger is better), "
"and does not perform BatchNorm recalibration. Every run record names the pipeline file it used, so the two objectives "
"are never conflated.")
caption(doc, "Table 2. Fixed architecture (shared method_config.json); the output head depends on the pipeline.")
make_table(doc, ["Component", "Configuration"], [
    ["Input", "1 x 64 x 64 FCGR (k=6, 4,096 cells); no reverse-complement merge, no pseudocount"],
    ["Convolution 1", "Conv2d(1, 64, k=7, s=2) -> 64 x 29; BatchNorm; ReLU"],
    ["Convolution 2", "Conv2d(64, 128, k=7, s=2) -> 128 x 12; BatchNorm; ReLU"],
    ["Pooling", "MaxPool2d(2) -> 128 x 6"],
    ["Mamba2 block", "single bidirectional block; d_state=128, d_conv=4, d_inner=256, 4 heads, chunk=64"],
    ["Sequence head", "flatten 4,608 -> Linear(512); dropout 0"],
    ["Coding MLP", "96 -> 256 -> 256 -> 512"],
    ["Fusion", "(512 + 128) = 640 -> Linear(512) -> ReLU"],
    ["Embedding", "512-d shared output; no head in the contrastive pipeline"],
    ["CE head (CE pipeline only)", "Linear(512, n_candidate_hosts); full-host softmax"],
])
caption(doc, "Table 3. The two independent pipelines and their incompatible settings.")
make_table(doc, ["Setting", "Contrastive pipeline (default)", "Cross-entropy pipeline"], [
    ["Config file", "pipeline_contrastive.json", "pipeline_ce.json"],
    ["Classification head", "none", "Linear(512, n_hosts)"],
    ["Training objective", "matrix contrastive loss", "softmax cross-entropy"],
    ["Ranking quantity", "Euclidean distance (smaller better)", "logits (larger better)"],
    ["BN recalibration", "3 passes, momentum=None", "not applied"],
    ["Epochs", "120 (100 fusion)", "120"],
    ["Optimizer", "Adam lr=1e-3, no weight decay", "Adam lr=1e-3, no weight decay"],
])

h2(doc, "2.5", "Splitting protocols and component allocation")
body(doc, "The random split assigns individual sequences to training, validation, and test sets at an 8:1:1 ratio. For "
"leakage control we additionally build a similarity-aware split. MinHash signatures [1] (k = 21, sketch size 1,000, "
"following Mash defaults [2]) are converted from pairwise Jaccard similarity to a Mash distance. Phages connected at a "
"Mash distance no greater than 0.05, approximately 95% average nucleotide identity, are joined into connected "
"components, and each component is assigned as a whole to one split with a deterministic, seed-controlled allocation "
"that preserves the 8:1:1 size ratio. At 0.05 this produces 1,049 components for CHERRY (largest 38; 793 singletons) "
"and 1,343 for DeepHost (largest 52; 938 singletons). Because no two phages above the identity threshold cross "
"splits, the test set cannot be solved by retrieving a near-identical training neighbor.")
body(doc, "Component allocation has a cost that we report explicitly. Because a component cannot be divided, the classes "
"are not evenly covered: under the CHERRY seed-123 split, 21 of the 80 test classes are absent from the training set "
"entirely, and two of the 79 DeepHost test classes are absent. A model cannot learn these classes from training data, "
"so the similarity-split decline reflects both the removal of neighbor leakage and uneven class coverage. We therefore "
"avoid attributing the entire gap to leakage.")

h2(doc, "2.6", "Training objectives")
body(doc, "Each contrastive batch contains B phages and re-encodes the full set of J candidate hosts, then forms the "
"B x J matrix of pairwise Euclidean distances. Recorded pairs are positives and label-mismatched pairs are proxy "
"negatives, denoting the absence of a recorded interaction rather than confirmed non-infection. Let {i}z{/i}{sub}i{/sub} "
"be phage {i}i{/i}'s embedding (i = 1..B), {i}h{/i}{sub}j{/sub} host {i}j{/i}'s embedding (j = 1..J), "
"{i}d{/i}{sub}ij{/sub} = || {i}z{/i}{sub}i{/sub} - {i}h{/i}{sub}j{/sub} ||, let {i}y{/i}{sub}i{/sub} be phage "
"{i}i{/i}'s recorded host, and let the margin {i}m = 1. The matrix contrastive loss is")
formula(doc, "{i}L = (1/(2BJ)){/i} Σ{sub}i=1..B{/sub} Σ{sub}j=1..J{/sub} {i}[ 1[j = y{/i}{sub}i{/sub}{i}] d{/i}"
"{sub}ij{/sub}{i}^2 + 1[j ≠ y{/i}{sub}i{/sub}{i}] (max(0, m - d{/i}{sub}ij{/sub}{i}))^2 ]", 2)
body(doc, "The indexing and normalization make the loss fully specified. The double sum runs over every one of the "
"B x J phage-host pairs, so all label-mismatched pairs enter the loss, not only pairs within the mini-batch. Each "
"phage has exactly one positive, its recorded host {i}y{/i}{sub}i{/sub}, and J - 1 proxy negatives; because the "
"candidate set is fixed, every phage has the same number of negatives. There is no separate positive or negative "
"weight: the single factor 1/(2BJ) averages over all entries, matching the implementation, which takes the mean over "
"the B x J matrix and then divides by two. The positive-negative imbalance (one versus J - 1) is therefore not "
"explicitly rebalanced, and the hinge terms dominate numerically.")
body(doc, "The cross-entropy pipeline instead applies a fair full-host softmax over all candidate hosts, rather than the "
"defective single-negative binary objective in the original study. Both pipelines are optimized with Adam at a "
"learning rate of 1e-3 with no weight decay; the sequence benchmark runs 120 epochs and the representation comparison "
"100 epochs. The checkpoint with the highest validation accuracy is retained. In the contrastive pipeline, "
"batch-normalization statistics are re-estimated on the training set before final evaluation without changing the "
"weights.")

h2(doc, "2.7", "Metrics, correction family, and resampling")
body(doc, "We report top-1 and top-5 accuracy, mean reciprocal rank, macro-averaged F1, and balanced accuracy, the last "
"being the mean per-class recall that gives equal weight to rare classes. Every primary comparison is repeated over "
"the fixed seeds 123, 2024, 42, 7, and 99; architecture ablations use three seeds. Seeds quantify training stability "
"under different initializations and data orders; they are not independent biological replicates. We therefore label "
"the seed-level paired t-tests and their standardized effect size Cohen's d [15] as stability descriptions rather "
"than population-level claims. Because the smallest exact two-sided Wilcoxon p with five seeds is 0.0625, we use the "
"sign of all five paired differences only as a robustness check.")
body(doc, "All hypothesis tests are assigned to one explicitly defined correction family so that the adjusted p values "
"can be reproduced from the table. The study uses a single global family of m = 13 tests: twelve paired "
"representation contrasts (FCGR, protein, and fusion across two datasets and two splits) plus the full-model "
"random-versus-similarity leakage contrast. Within this family the Holm step-down procedure [12] controls the "
"family-wise error rate and the Benjamini-Hochberg (BH) step-up procedure [13] controls the false discovery rate. "
"The complete list, with the rank, the unrounded raw p to six decimals, and both adjusted values, is released as "
"correction_full_list.csv and summarized in the results; recomputing Holm and BH from these raw values reproduces "
"every reported number.")
body(doc, "To obtain inference at the phage level rather than across the handful of seeds, we add a paired bootstrap and "
"a sign-flip permutation on the same test phages. For a fixed split we record whether each test phage is correct "
"under two models, resample the paired indicators 2,000 times with replacement, and report the percentile 95% "
"interval for the mean difference. The sign-flip permutation randomly reverses the within-phage model labels 2,000 "
"times to build a null distribution for the observed difference. This pairs the two predictions on exactly the same "
"phages, so the comparison is not confounded by a changing test set. We distinguish a non-significant result, which "
"means the present data do not provide evidence for an effect, from evidence that no effect exists; with few seeds "
"these are not equivalent.")

h2(doc, "2.8", "Classical baselines and a uniform tuning sensitivity")
body(doc, "We add four scikit-learn [16] classifiers on the flattened FCGR (4,096 dimensions) and coding (96 "
"dimensions) features: logistic regression (L2, C = 1, up to 3,000 iterations), random forest (300 trees), "
"k-nearest neighbors (k = 10), and a radial-basis support vector machine (C = 1). All features use a "
"StandardScaler fitted on the training partition, and probability columns are mapped back to global class "
"identifiers so rare-class handling cannot misalign the ranking. To test whether the published defaults are "
"adequate, we run one uniform validation-set grid: for each model family every candidate is trained on the training "
"partition and chosen by validation top-1, after which the default and the selected model are scored on the same "
"test set. The grids cover the regularization strength for logistic regression, tree count and depth for the forest, "
"the neighbor count for k-nearest neighbors, and the cost and kernel width for the support vector machine.")

h2(doc, "2.9", "Calibration and open-set rejection")
body(doc, "A single temperature parameter [9] is fitted on the validation set by minimizing the negative log-likelihood "
"with L-BFGS, which rescales the softmax over the candidate-host axis without changing the ranking. Expected "
"calibration error uses ten equal-width confidence bins, is sample-weighted, and compares the accuracy and mean "
"confidence in each bin [10]; the softmax is taken over the candidate hosts and label indices are verified against "
"the host table. The open-set construction is defined precisely. On the seed-123 random split, each of the 188 test "
"phages contributes two scored samples: one in-distribution sample over the full candidate set and one unknown sample "
"from which exactly one candidate, that phage's true host, is removed before the softmax is renormalized, giving 376 "
"paired samples. The removed host is not a genuinely unseen species: it is present in the training data and is only "
"withheld at inference, so this is a candidate-removal experiment that tests whether the model abstains when the "
"correct host is unavailable, not extrapolation to a novel organism. Each sample is scored by the calibrated maximum "
"softmax probability and the minimum distance. The temperature (0.1459) is the multi-seed validation-fit mean and is "
"fixed before the test set is touched, never re-estimated on it. The AUROC is computed per phage with equal weight "
"for the paired ID and unknown samples, and the false-positive rate at 95% true-positive rate is reported with a "
"2,000-resample paired phage bootstrap interval, together with coverage-accuracy at several confidence thresholds.")

h2(doc, "2.10", "Host-phylogeny controls and the label-use boundary")
body(doc, "Because related hosts share composition by descent, we build a data-driven host hierarchy directly from the "
"candidate host FCGRs using average-linkage agglomerative clustering on Euclidean distances, and also use the "
"explicit genus rank extracted from each scientific name. We then apply four controls with the same logistic "
"regression protocol: (1) genus-level prediction; (2) within-genus species discrimination, which conditions on "
"shared higher ancestry; (3) genus-mean residualization, which subtracts the training-set genus mean before "
"predicting species; and (4) host-tree residualization on the first ten principal components of the host FCGR.")
body(doc, "The label boundary is stated explicitly. The genus-level, genus-mean, and within-genus analyses use the real "
"genus label and are therefore post-hoc diagnostics: a genus label is not available when a new phage is processed, "
"so these quantities cannot be prediction inputs and must not be read as deployable accuracy. The host-tree "
"principal components are learned without labels, although the label is used only to index a phage's host. The "
"deployed BMKMPHI pipeline consumes only sequence-derived FCGR and pyrodigal features; genus labels, genus means, "
"and within-genus partitions never enter its training inputs, so there is no label leakage in the deployed "
"predictor.")

h2(doc, "2.11", "Cross-dataset de-duplicated transfer")
body(doc, "To test whether cross-benchmark results rely on near-identical sequences, we restrict evaluation to the host "
"species represented on both resources with host FCGRs available (87 species), train the full model on one resource, "
"and evaluate it on the other. For each target phage we compute its maximum FCGR cosine similarity to any source "
"phage, then report top-1 accuracy on the full target set and on the subsets whose nearest source similarity is "
"below 0.999, 0.99, and 0.95. A large drop after de-duplication indicates that the apparent transfer is driven by "
"sequence neighbors rather than independent generalization. This is a zero-shot transfer across separately pinned "
"reference genomes, so its absolute accuracy is not directly comparable to within-dataset evaluation.")

# ---- 3 Results ----
h1(doc, "3", "Results")
body(doc, "Figure 1 summarizes the end-to-end pipeline linking curation, representations, encoders, objectives, and "
"evaluation.")
add_figure(doc, FIG / "fig_workflow.png", width_cm=16,
           alt="End-to-end workflow from curation and representations to encoders, objectives and evaluation")
caption(doc, "Figure 1. End-to-end workflow. Phage and host genomes are encoded through FCGR and ORF-derived coding "
"features, processed by shared sequence and coding encoders, optionally fused, and used to rank candidate hosts; the "
"bottom band summarizes the two splitting protocols.")

h2(doc, "3.1", "Complete run provenance")
body(doc, "Because the manuscript draws on several training scripts and two pipelines, we make the origin of every "
"number explicit rather than relying on an abridged table. The full provenance file (full_provenance.csv) contains "
"one row for each of the 94 runs, recording the run identifier, the training script, the pipeline config file, the "
"split source and its seed, the output JSON, the saved checkpoint, the dataset, the seed, the train/validation/test "
"sizes, and the test top-1. All 94 checkpoints are retained and every output JSON is present, so each reported "
"number can be traced to one exact run, configuration, and split. Apparent discrepancies across tables are not "
"numerical errors: they come from distinct, intentionally different runs. The architecture benchmark uses the "
"120-epoch train.py pipeline, while the representation comparison uses the separate 100-epoch fusion pipeline, and "
"each row names whether it used the contrastive or the cross-entropy config.")
caption(doc, "Table 4. Provenance structure (the full 94-row file is released; representative fields shown).")
make_table(doc, ["Run identifier", "Script", "Config", "Split source", "Output / checkpoint"], [
    ["cherry_random_cnn_fcgr_contrastive_seed123", "train.py", "pipeline_contrastive.json",
     "random_split(seed=123)", "JSON + .pt both present"],
    ["cherry_random_cnn_mamba_fcgr_ce_seed123", "train.py", "pipeline_ce.json",
     "random_split(seed=123)", "JSON + .pt both present"],
    ["... 92 further runs", "train.py / fusion scripts", "named per row",
     "random or similarity split", "all retained"],
])

h2(doc, "3.2", "Benchmark and comparison with existing methods")
body(doc, "Table 5 places every model family on the same FCGR features and splits. A logistic regression is the strongest "
"single model on both datasets, reaching 82.32% (CHERRY) and 83.64% (DeepHost) under the random split and 59.50% and "
"60.42% under the similarity split, matching or exceeding every deep network including the 67-million-parameter "
"PhageCGRNet. Random forest and the RBF support vector machine are competitive, while k-nearest neighbors is weaker. "
"Figure 2 shows the same comparison with seed variability. The FCGR signal is therefore largely linearly separable, "
"and deep architectures add little measurable capacity on this task.")
caption(doc, "Table 5. Test top-1 accuracy on FCGR across families and splits (five-seed mean).")
make_table(doc, ["Model (type)", "CHERRY random", "CHERRY sim.", "DeepHost random", "DeepHost sim."], [
    ["Logistic regression (classical)", "82.32", "59.50", "83.64", "60.42"],
    ["Random forest (classical)", "76.41", "47.01", "77.86", "48.49"],
    ["RBF SVM (classical)", "67.36", "45.47", "76.10", "52.43"],
    ["K-nearest neighbors (classical)", "66.82", "42.85", "67.07", "44.64"],
    ["CNN-only variant (deep)", "77.30", "50.27", "-", "-"],
    ["BMKMPHI full (deep)", "74.68", "44.92", "80.91", "54.30"],
    ["PhageCGRNet (deep)", "74.47", "40.29", "-", "-"],
])
add_figure(doc, FIG / "fig_ml.png", width_cm=16,
           alt="Classical classifiers and deep reference lines on FCGR under random and similarity splits")
caption(doc, "Figure 2. Classical classifiers on FCGR under random and similarity splits, with deep-model reference "
"lines, on CHERRY and DeepHost (mean +/- SD).")

h2(doc, "3.3", "Neighbor leakage: magnitude and partition caveats")
body(doc, "Table 6 reports the deep sequence models under both protocols with their variability. Under the random split "
"all models cluster between 71.8% and 77.3%, and the lightweight CNN-only model is highest. Under the similarity "
"split every model degrades sharply; the full contrastive model falls from 74.68% to 44.92% and the CNN-only model "
"from 77.30% to 50.27%. Across five seeds the full-model random-versus-similarity gap is 29.76 points, the largest "
"effect in the study, and it remains significant after family-wise correction (Holm p = 0.0033; bootstrap 95% "
"interval 27.07 to 32.87; Cohen's d = 5.49). Figure 3 shows the pattern with seed variability.")
caption(doc, "Table 6. Test top-1 accuracy (mean +/- SD) of the deep sequence models under two splits.")
make_table(doc, ["Model", "Random split", "Similarity split", "Seeds"], [
    ["CNN-only variant", "77.30 ± 4.33", "50.27 ± 4.04", "3"],
    ["BMKMPHI full (contrastive)", "74.68 ± 2.05", "44.92 ± 3.72", "5"],
    ["Full + permuted representation", "74.11 ± 2.93", "44.74 ± 2.69", "3"],
    ["Full + cross-entropy", "71.81 ± 4.73", "41.18 ± 1.07", "3"],
    ["PhageCGRNet", "74.47 ± 1.84", "40.29 ± 3.79", "3"],
])
add_figure(doc, FIG / "fig_leakage.png", width_cm=15.5,
           alt="Top-1 accuracy of deep sequence models under random and similarity splits with error bars")
caption(doc, "Figure 3. Top-1 accuracy of the deep sequence models under random and similarity splits (mean +/- SD), "
"CHERRY.")
body(doc, "The threshold sweep is now repeated over three seeds rather than reported from one seed-123 run (Table 7). A "
"smaller Mash threshold is a stricter identity criterion and splits components more aggressively, and it also changes "
"which samples and classes land in the test set rather than varying one factor in isolation. Performance is highest "
"at the strictest 0.05 setting and declines as the threshold relaxes, but because the test composition itself "
"changes, this is a descriptive association across partitions, not a controlled dose-response.")
caption(doc, "Table 7. Multi-seed Mash-threshold sensitivity (three seeds, mean +/- SD).")
make_table(doc, ["Mash threshold", "Components", "Top-1", "Top-5", "MRR"], [
    ["0.05", "1,049", "46.35 ± 1.11", "68.09 ± 2.53", "56.53 ± 0.44"],
    ["0.10", "807", "29.59 ± 2.23", "53.65 ± 3.13", "41.23 ± 0.54"],
    ["0.20", "622", "26.92 ± 3.27", "54.37 ± 2.95", "39.79 ± 2.68"],
])
add_figure(doc, FIG / "fig_threshold.png", width_cm=14.5,
           alt="Top-1, top-5 and MRR across Mash distance thresholds")
caption(doc, "Figure 4. Top-1, top-5, and MRR across Mash thresholds; each threshold also changes the test composition.")

h2(doc, "3.4", "Paired component analysis on a fixed split")
body(doc, "The seed-level t-tests describe stability, so we add phage-level paired inference on the same test phages. "
"We classify this as an exploratory fixed-split analysis: it uses one seed-123 split, and its four permutation "
"contrasts are deliberately kept separate from the pre-specified 13-test family, so they are not Holm-corrected and "
"we do not call them family-wise significant. Table 8 reports the mean paired difference with a 2,000-resample "
"bootstrap interval and a sign-flip permutation p as directional evidence. Two findings follow. First, on the "
"seed-123 split the CNN-only point estimate is slightly above the full model, with a negative full-minus-CNN "
"difference of -5.85 points on the random split and -5.35 on the similarity split; the random-split interval "
"excludes zero. The incremental value of the bidirectional Mamba2 block is not stable in direction on a fixed "
"split. Second, the genuine FCGR is higher than a permuted representation under the similarity split (+6.42 points, "
"interval 0.011 to 0.123, unadjusted permutation p = 0.043), whereas under the random split there is no difference "
"(p = 1.0). Read as exploratory, this suggests that once near-neighbor retrieval is removed, real sequence "
"composition carries generalizable information, while the two-dimensional layout and the Mamba2 block do not by "
"themselves add a stable gain; the contrast should be confirmed on further splits before a strong claim.")
caption(doc, "Table 8. Exploratory fixed-split paired bootstrap and sign-flip permutation (seed 123, 2,000 resamples; "
"not part of the 13-test family).")
make_table(doc, ["Paired comparison (A vs B)", "Split", "Mean diff (pp)", "Bootstrap 95% CI", "Permutation p"], [
    ["Full vs CNN-only", "random", "-5.85", "[-10.11, -2.13]", "0.011"],
    ["Full vs CNN-only", "similarity", "-5.35", "[-10.70, -0.52]", "0.066"],
    ["FCGR vs permuted", "random", "-0.53", "[-4.26, 3.19]", "1.000"],
    ["FCGR vs permuted", "similarity", "+6.42", "[1.07, 12.30]", "0.043"],
])

h2(doc, "3.5", "ORF-derived coding features and the full correction list")
body(doc, "Table 9 compares the three representation variants over five seeds, with uncertainty for every metric. The "
"coding encoder reaches 78.62% under the random split and 53.80% under the similarity split, exceeding the FCGR "
"encoder in both, and the advantage is larger in the harder, de-leaked setting. The fusion model does not improve "
"over the coding encoder by a meaningful margin; essentially all of its gain over FCGR is inherited from the coding "
"branch.")
caption(doc, "Table 9. Three representation variants over five seeds (mean +/- SD for every metric).")
make_table(doc, ["Variant / split", "Top-1", "Top-5", "MRR", "Macro-F1"], [
    ["FCGR / random", "74.79 ± 1.90", "85.32 ± 1.79", "79.75 ± 1.93", "42.20 ± 4.47"],
    ["Coding / random", "78.62 ± 1.66", "88.83 ± 3.70", "83.40 ± 2.30", "46.84 ± 3.91"],
    ["Fusion / random", "78.94 ± 1.90", "89.15 ± 3.09", "83.51 ± 1.89", "45.98 ± 4.87"],
    ["FCGR / similarity", "43.64 ± 3.90", "67.49 ± 3.96", "54.74 ± 2.52", "22.18 ± 2.58"],
    ["Coding / similarity", "53.80 ± 3.15", "71.34 ± 3.56", "62.09 ± 2.80", "28.13 ± 2.88"],
    ["Fusion / similarity", "54.12 ± 3.24", "76.36 ± 5.39", "63.57 ± 3.52", "28.98 ± 3.68"],
])
add_figure(doc, FIG / "fig_protein.png", width_cm=16,
           alt="Sequence, coding and fusion variants under random and similarity splits")
caption(doc, "Figure 5. Sequence-only, coding-only, and fusion variants under the two splits on CHERRY and DeepHost "
"(mean +/- SD, n=5).")
body(doc, "The complete correction list in Table 10 makes every adjusted p reproducible. The single global family has "
"m = 13 tests; the table gives the rank, the context and contrast, the unrounded raw p to six decimals, and both the "
"Holm and the BH values. The strongest contrasts are significant under family-wise correction (Holm p as low as "
"0.0033), while the fusion-versus-coding contrasts at the bottom are non-significant under either procedure. "
"Recomputing the Holm step-down and BH step-up adjustments from the raw column reproduces the table exactly.")
caption(doc, "Table 10. Full correction list for the global family of 13 tests (raw p to six decimals).")
make_table(doc, ["Rank", "Context / contrast", "Raw p", "Holm p", "BH p"], [
    ["1", "CHERRY full leakage: random vs similarity", "0.000250", "0.003250", "0.002990"],
    ["2", "DeepHost sim: fusion vs FCGR", "0.000680", "0.008160", "0.002990"],
    ["3", "CHERRY random: protein vs FCGR", "0.000840", "0.009240", "0.002990"],
    ["4", "DeepHost sim: protein vs FCGR", "0.000920", "0.009240", "0.002990"],
    ["5", "DeepHost random: fusion vs FCGR", "0.008550", "0.076950", "0.019977"],
    ["6", "CHERRY random: fusion vs FCGR", "0.009220", "0.076950", "0.019977"],
    ["7", "CHERRY sim: protein vs FCGR", "0.014650", "0.102550", "0.027207"],
    ["8", "DeepHost random: protein vs FCGR", "0.020420", "0.122520", "0.033183"],
    ["9", "CHERRY sim: fusion vs FCGR", "0.024640", "0.123200", "0.035591"],
    ["10", "DeepHost random: fusion vs protein", "0.177810", "0.711240", "0.231153"],
    ["11", "DeepHost sim: fusion vs protein", "0.397670", "1.000000", "0.469974"],
    ["12", "CHERRY random: fusion vs protein", "0.710200", "1.000000", "0.752690"],
    ["13", "CHERRY sim: fusion vs protein", "0.752690", "1.000000", "0.752690"],
])

h2(doc, "3.6", "Classical-model tuning sensitivity")
body(doc, "The uniform validation grid covers 32 settings (two datasets, two splits, two feature sets, four "
"classifiers). Across all settings the mean absolute change in test top-1 is 2.29 points and the maximum is 9.93 "
"points; 20 settings improve, 4 worsen, and 8 are unchanged. The improvements are concentrated in k-nearest "
"neighbors, where selecting a smaller neighbor count raises top-1 by 5.6 to 9.9 points; logistic regression, random "
"forest, and the support vector machine change by less than two points in most settings. The published defaults are "
"therefore already adequate for these three families, and the only materially sensitive choice is the neighbor count, "
"whose tuned value still leaves k-nearest neighbors below the best classical and deep models. Table 11 shows "
"representative default-versus-tuned values.")
body(doc, "The reporting scope must be stated to avoid a seeming conflict with Table 5. Table 11 is a single-seed, "
"single-split result at seed 123, not a five-seed mean: the random entries use random_split(seed=123) and the "
"similarity entries use get_similarity_split(seed=123, threshold=0.05). The default and tuned models are trained on "
"the same training partition, the grid is chosen on the same validation partition, and both are scored on the same "
"held-out test partition, so the only thing that changes is the hyperparameter. Table 5 instead averages five seeds, "
"which is why its logistic-regression value (82.32) differs from the seed-123 default in Table 11 (85.14); the two "
"tables answer different questions and are not contradictory.")
caption(doc, "Table 11. Single-seed (seed 123) default versus tuned test top-1 on one fixed split; default and tuned "
"use identical train/validation/test partitions (full 32-setting file released).")
make_table(doc, ["Dataset / split / feature", "Classifier", "Default", "Tuned", "Change (pp)"], [
    ["CHERRY random FCGR", "Logistic regression", "85.14", "85.71", "+0.57"],
    ["CHERRY random FCGR", "Random forest", "78.86", "77.14", "-1.71"],
    ["CHERRY random FCGR", "K-nearest neighbors", "66.86", "76.00", "+9.14"],
    ["CHERRY random FCGR", "RBF SVM", "68.57", "66.86", "-1.71"],
    ["DeepHost random FCGR", "Logistic regression", "79.58", "79.93", "+0.35"],
    ["DeepHost similarity FCGR", "K-nearest neighbors", "43.62", "53.55", "+9.93"],
])

h2(doc, "3.7", "Host representative sensitivity")
body(doc, "To test whether the conclusions depend on the chosen reference strain, we replaced one species' representative "
"at a time with a different strain's RefSeq assembly and recomputed both the features and the end-to-end predictions. "
"The seven species were not drawn at random: they are the best-sampled hosts spanning different bacterial phyla for "
"which a second, qualifying RefSeq assembly of another strain exists. An eighth target, {i}S. viridochromogenes{/i}, "
"had no qualifying second chromosome-sized record within the 200 RefSeq candidates and was therefore excluded rather "
"than forced. All end-to-end runs use seed 123, with random_split(123) for the random protocol and "
"get_similarity_split(123, 0.05) for the similarity protocol, and only the one host under test is swapped.")
body(doc, "At the feature level the alternatives are close: FCGR cosine ranges from 0.969 to 0.9999 and protein-feature "
"Pearson correlation from 0.993 to 0.99999. Under the random split the flip rate is at most 4.56% and top-1 changes "
"by -1.40 to +0.35 points. Under the similarity split the flip rate is at most 10.21% and is concentrated in "
"{i}E. coli{/i}, where top-1 falls by 4.93 points, a change we do not describe as negligible; four other species "
"show no flips. We therefore state the conclusion narrowly: results from seven representative species suggest "
"limited sensitivity to reference-strain choice, but broader strain-level validation is still needed.")
caption(doc, "Table 12. Alternative-strain feature agreement and end-to-end change (seven selected species, seed 123).")
make_table(doc, ["Species", "FCGR cosine", "Protein r", "Random flip / delta", "Sim. flip / delta"], [
    ["Escherichia coli", "0.9991", "0.99998", "4.56% / -1.05", "10.21% / -4.93"],
    ["Pseudomonas aeruginosa", "0.9904", "0.99666", "0.00% / 0.00", "2.46% / 0.00"],
    ["Salmonella enterica", "0.9999", "0.99999", "0.70% / +0.35", "0.00% / 0.00"],
    ["Bacillus cereus", "0.9695", "0.99767", "1.75% / -1.05", "0.70% / -0.70"],
    ["Klebsiella pneumoniae", "0.9998", "0.99999", "0.00% / 0.00", "0.00% / 0.00"],
    ["Staphylococcus aureus", "0.9969", "0.99888", "0.70% / -0.70", "0.00% / 0.00"],
    ["Gordonia terrae", "0.9947", "0.99310", "2.11% / -1.40", "1.41% / -1.06"],
])

h2(doc, "3.8", "Confounding controls: not explained by GC or length")
body(doc, "A direct concern is that codon frequencies may simply restate GC content. The numerical correlation is real: "
"across the 64 codon features, the mean absolute correlation with GC is 0.73-0.74 and reaches 0.94-0.95 for the most "
"redundant codons. However, three results argue against GC being the source of the prediction. First, a baseline using "
"only GC, log length, and gene count reaches just 39.15% on CHERRY and 37.05% on DeepHost under the random split, far "
"below the codon-only result of 77.98% and 82.88%. Second, when every codon feature is residualized against GC using "
"a training-only regression, accuracy is essentially unchanged, at 78.62% and 83.65%. Third, adding the simple "
"features to codon gives no extra gain. The coding signal therefore carries information beyond GC, length, and gene "
"count. We do not control host phylogeny in this analysis, so a contribution from shared ancestry remains possible "
"and is reflected in the wording.")
caption(doc, "Table 13. Confounding controls on the random split (top-1, five-seed mean).")
make_table(doc, ["Feature set", "CHERRY", "DeepHost"], [
    ["GC + length + gene count", "39.15", "37.05"],
    ["Codon only", "77.98", "82.88"],
    ["Codon residualized on GC", "78.62", "83.65"],
    ["Simple + codon", "78.40", "83.30"],
    ["Full 96 features", "79.26", "82.32"],
])

h2(doc, "3.9", "Host-phylogeny controls: a mixed signal")
body(doc, "Table 14 separates the part of the codon signal that operates at the genus level from the part that survives "
"control for shared ancestry. Predicting the coarser genus is easier than predicting species, and subtracting each "
"genus's training-set mean codon vector causes a large decline, showing that a substantial component of the signal is "
"shared at the genus rank. Two results qualify this. First, within-genus discrimination among the species of each "
"multi-species genus remains high, so the codon features also carry fine-grained information that distinguishes "
"related species after conditioning on higher ancestry. Second, residualizing on the first ten host-tree principal "
"components changes accuracy only slightly, indicating the signal is not driven by a few global composition gradients. "
"These numbers are post-hoc and use the real genus label as documented in Section 2.10; they diagnose the signal rather "
"than report deployable performance.")
caption(doc, "Table 14. Host-phylogeny controls (top-1; random five-seed mean, similarity seed 123).")
make_table(doc, ["Control", "CHERRY random", "CHERRY sim.", "DeepHost random", "DeepHost sim."], [
    ["Species codon baseline", "77.98", "54.01", "82.88", "63.38"],
    ["Genus-level prediction", "85.43", "70.05", "88.98", "71.48"],
    ["Genus-mean residualized", "30.85", "15.51", "28.42", "14.79"],
    ["Host-tree PC residualized", "74.36", "52.41", "82.18", "56.69"],
    ["Within-genus discrimination", "85.75", "68.07", "88.94", "79.17"],
])

h2(doc, "3.10", "Cross-benchmark replication does not survive de-duplication")
body(doc, "The same within-dataset patterns recur on DeepHost, where the coding encoder again exceeds FCGR with a larger "
"margin after leakage control (Table 15). The two resources are not independent: they share 767 phage accessions and "
"89 host species, the cross-dataset FCGR nearest-neighbor cosine has a median near 1.0, and 1,604 CHERRY phages "
"(85.5%) have a DeepHost neighbor with cosine at least 0.999.")
caption(doc, "Table 15. DeepHost within-dataset replication over five seeds (mean +/- SD for every metric).")
make_table(doc, ["Variant / split", "Top-1", "Top-5", "MRR", "Macro-F1"], [
    ["FCGR / random", "80.91 ± 2.17", "90.53 ± 1.24", "85.36 ± 1.80", "55.06 ± 3.29"],
    ["Coding / random", "83.44 ± 2.80", "92.84 ± 1.44", "87.72 ± 1.89", "62.11 ± 3.57"],
    ["Fusion / random", "84.00 ± 2.19", "93.54 ± 1.83", "88.17 ± 1.50", "61.53 ± 3.76"],
    ["FCGR / similarity", "54.30 ± 1.04", "73.38 ± 2.15", "63.45 ± 1.09", "31.89 ± 3.11"],
    ["Coding / similarity", "63.52 ± 1.55", "82.25 ± 2.51", "71.89 ± 1.06", "40.21 ± 1.38"],
    ["Fusion / similarity", "62.96 ± 1.52", "80.49 ± 2.75", "70.96 ± 1.44", "39.16 ± 2.90"],
])
body(doc, "The de-duplicated transfer test in Table 16 shows that the overlap is doing the work. Training on DeepHost "
"and transferring to CHERRY gives 91.06% on the full target set, but restricting the target to phages without a "
"near-identical source neighbor drops accuracy to 62.22% below cosine 0.999 and to 19.23% below 0.95. The reverse "
"direction is weak even before de-duplication and falls to 7.59-11.35% on the de-duplicated subsets. Because "
"performance collapses once sequence neighbors are removed, the cross-benchmark result is within-neighbor replication "
"rather than independent evidence. A prospectively collected, non-overlapping validation set remains required.")
caption(doc, "Table 16. Zero-shot cross-dataset transfer after FCGR cosine de-duplication (seed 123, 100 epochs).")
make_table(doc, ["Direction", "Full target", "< 0.999", "< 0.99", "< 0.95"], [
    ["DeepHost -> CHERRY (n=1297)", "91.06", "62.22 (n=90)", "58.33 (n=72)", "19.23 (n=26)"],
    ["CHERRY -> DeepHost (n=2190)", "15.48", "11.35 (n=960)", "9.84 (n=742)", "7.59 (n=303)"],
])

h2(doc, "3.11", "Coding sub-blocks, calibration, imbalance, and efficiency")
body(doc, "The sub-block ablation in Table 17 identifies codon frequencies as the strongest single block, reaching 50.98% "
"under the similarity split on its own, close to the full encoder; the amino-acid block is next and the structural "
"block weakest, and combining codon with amino-acid features gives the highest result. Figure 6 presents the same "
"comparison with seed variability. We describe this carefully: codon usage is the dominant predictive component and "
"is consistent with translational adaptation to the host [11], but this observational association does not establish "
"that mechanism, because it does not distinguish adaptation from correlated GC, composition, or phylogeny.")
caption(doc, "Table 17. Coding sub-block ablation (top-1, three-seed mean).")
make_table(doc, ["Feature block", "Random", "Similarity"], [
    ["Structure only (8)", "58.51", "27.63"],
    ["Codon only (64)", "76.77", "50.98"],
    ["Amino-acid + physicochemical (24)", "74.65", "44.39"],
    ["Codon + amino-acid (88)", "79.08", "53.12"],
])
add_figure(doc, FIG / "fig_ablation.png", width_cm=15.5,
           alt="Coding feature group ablation under two splits with error bars")
caption(doc, "Figure 6. Coding feature-group ablation under the two splits (mean +/- SD, n=3), CHERRY.")
body(doc, "The uncalibrated models are over-confident, with mean ECE of 0.729 on the random split and 0.436 on the "
"similarity split. Fitted temperatures are stable and reduce ECE to 0.067 and 0.088 and NLL to 1.21 and 2.57 without "
"changing the ranking (Table 18, Figure 7). When the true host is withheld, the calibrated confidence and the distance "
"score distinguish unknown from known cases with AUROC 0.911 (95% CI 0.881-0.937) and 0.935 (95% CI "
"0.909-0.957), and the false-positive rate at 95% true-positive rate is 0 (bootstrap CI 0-0); at a 0.9 "
"confidence threshold the model accepts 28.2% of predictions with 98.1% accuracy.")
caption(doc, "Table 18. Multi-seed calibration summary (five seeds per split, mean +/- SD).")
make_table(doc, ["Split", "Temperature", "ECE before", "ECE after", "NLL after"], [
    ["Random", "0.146 ± 0.009", "0.729 ± 0.018", "0.067 ± 0.008", "1.214 ± 0.133"],
    ["Similarity", "0.143 ± 0.036", "0.436 ± 0.031", "0.088 ± 0.017", "2.568 ± 0.160"],
])
add_figure(doc, FIG / "fig_calibration.png", width_cm=13,
           alt="Expected calibration error and negative log-likelihood before and after temperature scaling")
caption(doc, "Figure 7. Expected calibration error and negative log-likelihood before and after temperature scaling.")
body(doc, "The imbalance analysis quantifies the rare-class limit. On the seed-123 random split, balanced accuracy is only "
"47.29% against a top-1 of 73.4%, while a uniform choice gives 0.49% and a majority-class choice 17.55%. Singleton "
"classes average F1 0.318 with 29 zero-F1 classes, classes with at least five samples average 0.890, and 32 classes "
"have zero F1 overall. Within the 15 best-supported classes the confusion matrix is largely diagonal, and the residual "
"errors concentrate among related hosts.")
add_figure(doc, FIG / "fig_confusion.png", width_cm=14,
           alt="Confusion matrix for the 15 best-supported host classes")
caption(doc, "Figure 8. Confusion matrix for the 15 best-supported host classes (seed 123); the diagonal-dominant "
"statement is restricted to these classes.")
body(doc, "Phage-level uncertainty is reported separately from seed variation: resampling the test phages 2,000 times "
"gives percentile 95% intervals, for example CHERRY random top-1 0.734 (0.670-0.798) and similarity top-1 0.487 "
"(0.417-0.562), and DeepHost top-1 0.628 (0.572-0.681) and 0.482 (0.423-0.542). The full per-class recall "
"distribution is shown in Supplementary Figures S1 and S2: on CHERRY 32 of 65 evaluated classes have zero recall with "
"mean recall 0.473, and on DeepHost 19 of 65 have zero recall with mean 0.636. The similarity-split result is also "
"stable across five component allocations (CHERRY top-1 0.479 ± 0.024; DeepHost 0.543 ± 0.010). Additional "
"diagnostics show the multiclass Brier score improving from 0.974 to 0.356 (CHERRY) and 0.949 to 0.372 (DeepHost), "
"split conformal coverage of 92.0% and 88.8% with mean set sizes 5.66 and 2.75, and energy-score AUROC of 0.936 and "
"0.899, while the frequency-only baselines remain weak (uniform 0.49%/0.93%; majority 17.55%/16.14%).")
add_figure(doc, FIG / "fig_full_class_recall_cherry.png", width_cm=15.5,
           alt="Per-class recall across all evaluated CHERRY host classes")
caption(doc, "Supplementary Figure S1. Per-class recall across all evaluated CHERRY host classes (seed 123).")
add_figure(doc, FIG / "fig_full_class_recall_deephost.png", width_cm=15.5,
           alt="Per-class recall across all evaluated DeepHost host classes")
caption(doc, "Supplementary Figure S2. Per-class recall across all evaluated DeepHost host classes (seed 123).")
body(doc, "Under the documented efficiency protocol (NVIDIA RTX 4090, CUDA 12.4, FP32, batch 32, 10 warm-up and 100 "
"timed iterations over five repeats, GPU-synchronized, forward only), the full model uses 3.07 million parameters and "
"2.52 ± 0.02 ms per batch, the CNN-only variant 2.76 million and 0.24 ms, and PhageCGRNet 67.4 million and 2.68 ms. "
"The coding encoder is a small MLP and adds negligible cost, and the classical classifiers need no accelerator.")
caption(doc, "Table 19. Model size and forward-pass efficiency (mean +/- SD).")
make_table(doc, ["Model", "Parameters", "Latency (ms/batch)"], [
    ["BMKMPHI full", "3,066,520", "2.522 ± 0.017"],
    ["CNN-only variant", "2,764,928", "0.241 ± 0.005"],
    ["PhageCGRNet", "67,424,846", "2.679 ± 0.005"],
])

# ---- 4 Discussion ----
h1(doc, "4", "Discussion")
body(doc, "The most consistent finding is that the data partition, not the model family, controls the apparent accuracy. "
"Every deep model loses roughly 30 points when near-identical phages are kept together, and the leakage contrast is "
"the largest and most stable effect in the study. Because the similarity split also changes class coverage and leaves "
"some test classes absent from training, we do not attribute the full gap to leakage; the honest statement is that "
"random-split numbers mix genuine generalization with neighbor retrieval, and the similarity split gives a lower, "
"more conservative estimate. The phage-level paired analysis adds a second, exploratory line of evidence: in the "
"fixed-split analysis the real FCGR signal is higher than a permuted one under the similarity split (unadjusted "
"p = 0.043), consistent with sequence composition carrying generalizable information after retrieval is removed, "
"although this should be confirmed on further splits before a strong claim.")
body(doc, "A second finding is that the architectural complexity is not matched by a measurable gain. A logistic "
"regression on the same features is the strongest single model, and the bidirectional Mamba2 block does not improve "
"ranking on a fixed split; its seed-level comparison is non-significant and its paired bootstrap difference is "
"negative. The permuted representation is indistinguishable from FCGR under the random split, which indicates that "
"the two-dimensional layout contributes little and that k-mer composition is the operative signal. The uniform "
"tuning grid further shows that the classical defaults are already adequate for logistic regression, random forest, "
"and the support vector machine, and that only the neighbor count matters materially, without changing the ranking.")
body(doc, "The ORF-derived coding features are the one representation that improves host ranking under leakage control, "
"with a larger margin in the harder split. The strongest contrasts survive family-wise correction, whereas borderline "
"ones hold only under the less stringent false-discovery-rate procedure, and the fusion model adds little over the "
"coding branch. The effect is not explained by GC, length, or gene count, because residualizing codon frequencies on "
"GC leaves accuracy unchanged and the simple baseline is far weaker. We retain a cautious biological interpretation: "
"codon usage is predictive and is consistent with translational adaptation, but this observational result does not "
"prove it, and a component shared at the genus rank remains. We therefore avoid claiming adaptation as a mechanism.")
body(doc, "Several limitations bound the conclusions. The similarity split reduces coverage and cannot be treated as a "
"controlled single-factor experiment. The phylogeny controls use the real genus label and are post-hoc, so they "
"diagnose rather than deploy. The two resources overlap heavily and cross-dataset accuracy collapses after "
"de-duplication, so neither validates the other. Gene calls depend on a metagenomic predictor and on fragmented "
"assemblies, and three host genomes are flagged rather than removed. Finally, all inference is within the observed "
"phage population; a prospective, non-overlapping, independently curated validation set is the appropriate next "
"step. Future work should add such a set, incorporate receptor and CRISPR evidence where available, and test whether "
"the coding signal transfers across genuinely novel hosts rather than across near neighbors.")

# ---- 5 Conclusion ----
h1(doc, "5", "Conclusion")
body(doc, "On two alignment-free phage-host benchmarks, the dominant effect is the data partition, and a simple "
"logistic regression matches or exceeds the deep models. The bidirectional Mamba2 block and the two-dimensional FCGR "
"layout do not add a stable gain, while ORF-derived coding features improve ranking under leakage control and retain "
"their signal after GC residualization and multiple-comparison correction. The biological interpretation is kept "
"deliberately narrow: codon usage is a predictive and consistent signal, not demonstrated proof of adaptation. "
"Because the two resources overlap and do not survive de-duplication, independent prospective validation remains the "
"principal requirement before strong deployment claims are made.")

# ---- Statements ----
h1(doc, "", "Data and code availability")
body(doc, "The source data are public: the CHERRY phage and host resources are released with that project [5], and the "
"DeepHost taxonomy and processed sequences are released with that project [6]. The reproducibility package contains "
"the curated and processed datasets, all feature and host-FCGR files, the full 94-run provenance, the complete "
"correction list, every analysis script, the training and inference code, and the figures. The code is maintained in "
"the public repository https://github.com/QWERasd12345-ASD/BMKMPHI, which also carries the processed data and "
"per-run results, and the same files are included in this reproducibility package. A versioned release with a "
"persistent archive identifier (for example a Zenodo DOI) will be minted at acceptance, and the release tag and DOI "
"are to be added at that point. A manifest records SHA-256 hashes for all processed files.")

h1(doc, "", "Author contributions")
body(doc, "Yuan Hou: data curation, software, formal analysis, validation, and writing - original draft. Haoran Gu: "
"software, data curation, and validation. Xingshun Song: conceptualization, methodology, resources, and writing - "
"review and editing. Yan Miao: conceptualization, methodology, supervision, funding acquisition, and writing - review "
"and editing. All authors read and approved the final manuscript.")

h1(doc, "", "Funding")
body(doc, "This work was supported by the National Natural Science Foundation of China (grant 62301139) and the "
"Heilongjiang Provincial Natural Science Foundation (grant QC2025F001). The funders had no role in study design, data "
"collection, analysis, interpretation, or the decision to submit the manuscript.")

h1(doc, "", "Conflict of interest")
body(doc, "The authors declare that they have no competing financial interests or personal relationships that could have "
"appeared to influence the work reported in this paper.")

h1(doc, "", "Acknowledgments")
body(doc, "The authors thank the maintainers of the public CHERRY, DeepHost, Mash, and pyrodigal resources, and the "
"reviewers whose comments led to the explicit provenance, correction family, paired inference, and tuning analyses.")

# ---- References ----
h1(doc, "", "References")
refs = [
    "Ondov BD, Treangen TJ, Melsted P, et al. Mash: fast genome and metagenome distance estimation using MinHash. Genome Biology, 2016, 17: 132.",
    "Broder AZ. On the resemblance and containment of documents. In: Proceedings of the Compression and Complexity of Sequences, 1997: 21-29.",
    "Hyatt D, Chen GL, Locascio PF, et al. Prodigal: prokaryotic gene recognition and translation initiation site identification. BMC Bioinformatics, 2010, 11: 119.",
    "Larralde M. pyrodigal: Python bindings and interface to Prodigal, an efficient method for gene prediction in prokaryotes. Journal of Open Source Software, 2022, 7(72): 4296.",
    "Shang J, Sun Y. CHERRY: a Computational metHod for accuratE pRediction of virus-pRokarYotic interactions using a graph encoder-decoder model. Briefings in Bioinformatics, 2022, 23(1): bbac182.",
    "Wang R, Zhang X-L-L, Wang J, Li SC. DeepHost: phage host prediction with convolutional neural network. Briefings in Bioinformatics, 2022, 23(1): bbab385.",
    "Wang T, Yu Z-G, Li J, Lin X. PhageCGRNet: integrating chaos game representation of genomes with convolutional neural network for accurate phage host classification prediction. PLOS Computational Biology, 2026, 22(4): e1014130.",
    "Guo J, Chechik G. On equivalence between loss functions in mining and ranking. In: Proceedings of ICML, PMLR 70, 2017: 1321-1329.",
    "Naeini MP, Turek M, Cooper GF. Obtaining well calibrated probabilities using Bayesian binning. In: Proceedings of AAAI, 2015.",
    "Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of ICML, 2017.",
    "Sharp PM, Li WH. The codon adaptation index - a measure of directional synonymous codon usage bias, and its potential applications. Nucleic Acids Research, 1987, 15(3): 1281-1295.",
    "Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 1979, 6(2): 65-70.",
    "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society Series B, 1995, 57(1): 289-300.",
    "Efron B. Bootstrap methods: another look at the jackknife. The Annals of Statistics, 1979, 7(1): 1-26.",
    "Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed. Hillsdale: Lawrence Erlbaum Associates, 1988.",
    "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research, 2011, 12: 2825-2830.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph(); pf = p.paragraph_format
    pf.line_spacing = 1.3; pf.space_after = Pt(2); pf.left_indent = Cm(0.75); pf.first_line_indent = Cm(-0.75)
    set_font(p.add_run(f"[{i}] {r}"), size=9.5)

doc.save(OUT)
print("saved:", OUT)
print("paragraphs:", len(doc.paragraphs), "tables:", len(doc.tables))
