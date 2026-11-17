# -*- coding: utf-8 -*-
"""Shared publication-quality plotting style."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Muted, color-blind-friendly, journal-ready palette
PALETTE = {
    "blue":   "#4878A8",
    "orange": "#E08A3C",
    "green":  "#5A9E6F",
    "red":    "#C0504D",
    "purple": "#7E6BA8",
    "teal":   "#5AA8A0",
    "gray":   "#8A8F98",
    "gold":   "#D4A94E",
}
# semantic mappings
SPLIT_COLORS = {"random": PALETTE["blue"], "similarity": PALETTE["orange"]}
REPR_COLORS = {"FCGR": PALETTE["blue"], "Protein": PALETTE["green"],
               "Fusion": PALETTE["purple"]}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#D8DBE0",
        "grid.linewidth": 0.7,
        "grid.linestyle": "--",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def add_err(ax, x, y, err, color="black", lw=1.1, capsize=3.5):
    ax.errorbar(x, y, yerr=err, fmt="none", ecolor=color,
                elinewidth=lw, capthick=lw, capsize=capsize, zorder=5)
