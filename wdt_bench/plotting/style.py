"""Shared matplotlib appearance for the WDT-Bench figures.

Nature-family styling: white background, no grid, left and bottom spines
only, sans-serif base fonts (the paper figures switch to a serif family via
:func:`wdt_bench.plotting.paper_common.apply_font`).
"""
from __future__ import annotations

import matplotlib.pyplot as plt

FIGURE_DPI = 120
SAVEFIG_DPI = 400


def apply_style() -> None:
    """Apply the shared rcParams used by every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": SAVEFIG_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "Helvetica",
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
        }
    )
