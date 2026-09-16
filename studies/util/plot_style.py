"""Shared matplotlib style + small stats helpers for study analysis figures.

Presentation-grade defaults (large axis/tick text) per project convention, and
the per-hidden-size color palette used across the scaling studies so H=64 is the
same green in every figure. Import from a study's analysis script:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "util"))
    from plot_style import apply_presentation_style, HCOLOR, t975

Error-bar convention (Han 2026-07): studies draw **SEM** on scaling figures with
n noted in the figure; the t-based 95% CI half-width (t975(n)*sem) is retained in
the curated JSON/CSV for downstream use but not drawn by default.
"""
from __future__ import annotations

import matplotlib as mpl

# One hue per GRU hidden size (H); reused across studies. 2-way references are
# drawn in the same hue, faded, so a reader never re-checks the legend.
HCOLOR = {16: "#4C72B0", 64: "#55A868", 128: "#C44E52", 256: "#8172B3"}

# Shared species palette: rodents are neutral/cool and primates are warm.
# These Okabe-Ito-inspired colors retain contrast on white backgrounds.
SPECIES_COLORS = {
    "mouse": "#333333",
    "rat": "#0072B2",
    "macaque": "#E69F00",
    "human": "#D55E00",
}

# t_{0.975, n-1} for small-n 95% CI half-width (= t * sem).
_T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
         7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def t975(n: int) -> float:
    """Two-sided t critical value at 0.975 for n-1 dof (normal approx for large n)."""
    return _T975.get(n, 1.96)


def apply_presentation_style() -> None:
    """Set presentation-grade rcParams (large fonts, no top/right spines).

    Font (Han, 2026-08): Helvetica by default, falling back to Arial / DejaVu
    Sans on hosts where Helvetica isn't installed (e.g. Linux compute). SVG
    labels remain editable ``<text>`` elements rather than glyph paths.
    """
    mpl.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "lines.linewidth": 2.4,
    })
