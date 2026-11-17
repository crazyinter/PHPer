# -*- coding: utf-8 -*-
"""End-to-end workflow diagram of the controlled benchmark."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from plot_style import apply_style, PALETTE

apply_style()
ROOT = Path(r"C:\Users\Admin\Doubao\chats\2026-11-13\new-chat-1\BMKMPHI_repro")
FIG = ROOT / "figures"

fig, ax = plt.subplots(figsize=(13.5, 7.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

C_INPUT = PALETTE["gray"]; C_SEQ = PALETTE["blue"]; C_PROT = PALETTE["green"]
C_FUSE = PALETTE["purple"]; C_OUT = PALETTE["orange"]; C_SPLIT = PALETTE["red"]


def box(x, y, w, h, title, lines, color, tsize=11.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                       linewidth=1.4, edgecolor=color, facecolor=color + "18")
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 4.2, title, ha="center", va="top",
            fontsize=tsize, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 9.5, "\n".join(lines), ha="center", va="top",
            fontsize=9.2, color="#222222", linespacing=1.45)


def arrow(x1, y1, x2, y2, color="#444444", lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=13, linewidth=lw, color=color))


# Column 1 - inputs
box(2, 60, 15, 22, "Phage genomes", ["1,875 (CHERRY)", "2,845 (DeepHost)"], C_INPUT)
box(2, 24, 15, 22, "Host genomes", ["206 / 118 species", "reference candidates"], C_INPUT)

# Column 2 - representations
box(23, 60, 17, 22, "FCGR (k = 6)", ["64 x 64 matrix", "4,096 k-mer channels", "nucleotide composition"], C_SEQ)
box(23, 24, 17, 22, "Protein features", ["pyrodigal ORFs", "96 dimensions", "structure / codon / amino /", "physicochemical"], C_PROT)

# Column 3 - encoders
box(46, 60, 16, 22, "Sequence encoder", ["2-layer CNN", "bidirectional Mamba2", "512-d embedding"], C_SEQ)
box(46, 24, 16, 22, "Protein encoder", ["MLP 96-256-256", "128-d embedding"], C_PROT)

# Column 4 - fusion + objective
box(68, 40, 15, 26, "Fusion & objective", ["concatenation", "hinge contrastive loss", "Euclidean margin = 1", "(cross-entropy control)"], C_FUSE)

# Column 5 - output
box(87, 40, 11, 26, "Host ranking", ["nearest-host", "distance", "top-1 / top-5", "MRR / macro-F1"], C_OUT)

# arrows: inputs -> representations
arrow(17, 71, 23, 71, C_SEQ)
arrow(17, 35, 23, 35, C_PROT)
# representations -> encoders
arrow(40, 71, 46, 71, C_SEQ)
arrow(40, 35, 46, 35, C_PROT)
# encoders -> fusion
arrow(62, 68, 68, 58, C_FUSE)
arrow(62, 38, 68, 50, C_FUSE)
# fusion -> output
arrow(83, 53, 87, 53, C_OUT)

# Bottom band - splitting protocol
p = FancyBboxPatch((2, 5), 96, 11, boxstyle="round,pad=0.4,rounding_size=1.6",
                   linewidth=1.4, edgecolor=C_SPLIT, facecolor=C_SPLIT + "14")
ax.add_patch(p)
ax.text(50, 12.6, "Splitting protocol applied to every model", ha="center",
        va="top", fontsize=11, fontweight="bold", color=C_SPLIT)
ax.text(50, 8.2, "Random split (8:1:1)     vs.     Similarity-aware split: MinHash k=21, Mash distance 0.05 connected components kept together",
        ha="center", va="center", fontsize=9.3, color="#222222")

plt.savefig(FIG / "fig_workflow.png")
print("saved fig_workflow.png")
