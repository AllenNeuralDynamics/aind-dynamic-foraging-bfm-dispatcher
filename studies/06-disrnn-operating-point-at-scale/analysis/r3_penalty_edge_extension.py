"""r3 producer: does going lighter than mult-d-grid's tested penalty floor help further?

Reads the committed analysis/grid.csv (mult-d-grid, 80-run wave 1) and analysis/grid_wave2.csv
(wave 2's 8-run extended-penalty-range grid) and asks, at each of the two axes wave 2 tested
independently:

  - beta axis (mult=1 fixed): mult-d-grid tested beta in {1e-3, 3e-4}; wave 2 adds beta=1e-4.
  - mult axis (beta=3e-4 fixed): mult-d-grid tested mult in {1,2,5,10}; wave 2 adds mult=0.5.

... whether held-out likelihood keeps improving past mult-d-grid's tested floor (mult=1,
beta=3e-4), and whether the D=614 generalization gap (train LL - held-out LL) keeps shrinking
or turns around, at D in {300, 614} (the two cohorts wave 2 tested; D snapped to nominal per
mult-d-grid's own +/-3 tolerance since resolved subject count varies by seed).

Only state=='finished' rows are trusted (both grids report 100% finished for the cells used
here -- see r3's launch-verification section). No backfilled rows are used from either file
for the cells this report reads (wave2 has none; the 6 mult-d-grid backfilled cells fall
outside the D in {300,614}/beta,mult settings compared here EXCEPT one -- see NOTE below).

    python analysis/r3_penalty_edge_extension.py

OUTPUTS:
  analysis/r3_summary.json               - curated per-(D,mult,beta) stats + provenance
  analysis/reports/fig_r3_penalty_edge_extension.png - the verdict figure
Regenerates the <!-- BEGIN result-1 --> / result-2 blocks in reports/r3-penalty-edge-extension.md.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402

GRID_CSV = HERE / "grid.csv"
WAVE2_CSV = HERE / "grid_wave2.csv"
REPORT = HERE / "reports" / "r3-penalty-edge-extension.md"
FIG = HERE / "reports" / "fig_r3_penalty_edge_extension.png"
WANDB_GROUPS = [
    "mult-d-grid@20260718-151409",
    "wave2-step-budget-and-penalty-extension@20260909-021546",
]

D_NOMINAL = (10, 30, 100, 300, 614)
D_SNAP_TOL = 3
GRU = {10: 0.7219, 30: 0.7250, 100: 0.7262, 300: 0.7267, 614: 0.7268}   # study 01
RL_BASELINE = 0.7170                                                    # study 05 r1


def nominal_d(d: int) -> int:
    for target in D_NOMINAL:
        if abs(d - target) <= D_SNAP_TOL:
            return target
    return d


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_csv_usable(path: Path) -> list[dict]:
    """Rows usable for analysis: state=='finished' OR heldout_backfilled==True with a value."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    usable = []
    for r in rows:
        ll = _f(r.get("heldout_ll"))
        if ll is None:
            continue
        if r["state"] == "finished" or str(r.get("heldout_backfilled")).lower() == "true":
            r["heldout_ll"] = ll
            r["eval_ll"] = _f(r.get("eval_ll"))
            r["D"] = int(float(r["D"]))
            r["D_nom"] = nominal_d(r["D"])
            r["mult"] = round(float(r["mult"]), 4)
            r["beta"] = float(r["beta"])
            usable.append(r)
    return usable


def cell_stats(rows: list[dict], D_nom: int, mult: float, beta: float) -> dict:
    sub = [r for r in rows if r["D_nom"] == D_nom and abs(r["mult"] - mult) < 1e-6
           and abs(r["beta"] - beta) < 1e-9]
    n = len(sub)
    heldout = [r["heldout_ll"] for r in sub]
    gaps = [r["eval_ll"] - r["heldout_ll"] for r in sub if r["eval_ll"] is not None]
    mean = lambda v: sum(v) / len(v) if v else None
    sem = lambda v: (np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
    return {
        "D": D_nom, "mult": mult, "beta": beta, "n": n,
        "heldout_mean": mean(heldout), "heldout_sem": sem(heldout),
        "gap_mean": mean(gaps), "gap_sem": sem(gaps),
    }


def main() -> None:
    orig = read_csv_usable(GRID_CSV)
    wave2 = read_csv_usable(WAVE2_CSV)
    all_rows = orig + wave2

    n_wave2_total, n_wave2_finished = 8, sum(1 for r in wave2 if r["state"] == "finished")

    beta_axis_settings = [(1.0, 0.0010), (1.0, 0.0003), (1.0, 0.0001)]     # heavy -> light
    mult_axis_settings = [(10.0, 0.0003), (5.0, 0.0003), (2.0, 0.0003), (1.0, 0.0003), (0.5, 0.0003)]

    cells = []
    for D in (300, 614):
        for mult, beta in beta_axis_settings:
            cells.append({**cell_stats(all_rows, D, mult, beta), "axis": "beta"})
        for mult, beta in mult_axis_settings:
            if (mult, beta) == (1.0, 0.0003):
                continue  # already added by the beta-axis loop above; avoid double count in JSON
            cells.append({**cell_stats(all_rows, D, mult, beta), "axis": "mult"})

    summary = {
        "_meta": build_meta("analysis/r3_penalty_edge_extension.py", WANDB_GROUPS, study_root=STUDY),
        "note": ("wave2 pulled via W&B GraphQL from the sandbox (wandb.Api() is blocked there); "
                 "see analysis/pull_wave2_grid.py docstring. All 8 wave2 runs state=='finished', "
                 "no backfilled rows used."),
        "progress": {"wave2_total": n_wave2_total, "wave2_finished": n_wave2_finished},
        "cells": cells,
        "gru_reference": GRU,
        "rl_baseline": RL_BASELINE,
    }
    with (HERE / "r3_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {HERE / 'r3_summary.json'} — {len(cells)} cells")

    make_figure(all_rows, beta_axis_settings, mult_axis_settings)
    print(f"wrote {FIG}")


def make_figure(all_rows, beta_axis_settings, mult_axis_settings):
    sys.path.insert(0, str(REPO))  # for the figure_style kernel plugin functions if run as a script
    try:
        apply_figure_style()  # noqa: F821 -- provided by figure-style skill when running interactively
    except NameError:
        plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    def _panel_letter(ax, letter):
        try:
            panel_letter(ax, letter)  # noqa: F821 -- figure-style skill helper, if loaded
        except NameError:
            ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="bottom", ha="left")

    D_COLOR = {300: "#4c72b0", 614: "#1a1a1a"}
    NEW_MARKER = "D"     # wave2's new lighter point
    OLD_MARKER = "o"     # mult-d-grid's tested points

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    def plot_axis(ax_top, ax_bot, settings, xlabel, x_is_beta):
        xs = list(range(len(settings)))
        xticklabels = []
        for mult, beta in settings:
            if x_is_beta:
                xticklabels.append(f"{beta:.0e}".replace("e-0", "e-"))
            else:
                xticklabels.append(f"{mult:g}")
        new_point_idx = len(settings) - 1  # the lightest, newly-added point is always last

        for D in (300, 614):
            heldout_y, heldout_e, gap_y, gap_e = [], [], [], []
            for mult, beta in settings:
                st = cell_stats(all_rows, D, mult, beta)
                heldout_y.append(st["heldout_mean"])
                heldout_e.append(st["heldout_sem"] or 0)
                gap_y.append(st["gap_mean"])
                gap_e.append(st["gap_sem"] or 0)
            c = D_COLOR[D]
            ax_top.errorbar(xs, heldout_y, yerr=heldout_e, color=c, marker=OLD_MARKER,
                             markersize=5, lw=1.4, capsize=2, label=f"D={D}")
            ax_top.plot(xs[new_point_idx], heldout_y[new_point_idx], marker=NEW_MARKER,
                        markersize=8, color=c, markeredgecolor="white", markeredgewidth=0.8,
                        zorder=5)
            ax_bot.errorbar(xs, gap_y, yerr=gap_e, color=c, marker=OLD_MARKER,
                             markersize=5, lw=1.4, capsize=2)
            ax_bot.plot(xs[new_point_idx], gap_y[new_point_idx], marker=NEW_MARKER,
                        markersize=8, color=c, markeredgecolor="white", markeredgewidth=0.8,
                        zorder=5)

        ax_top.set_xticks(xs)
        ax_top.set_xticklabels([])
        ax_bot.set_xticks(xs)
        ax_bot.set_xticklabels(xticklabels)
        ax_bot.set_xlabel(xlabel)
        for ax in (ax_top, ax_bot):
            ax.margins(x=0.12)

    plot_axis(axes[0, 0], axes[1, 0], beta_axis_settings, "global penalty β (mult=1 fixed)", True)
    plot_axis(axes[0, 1], axes[1, 1], mult_axis_settings, "interaction multiplier (β=3e-4 fixed)", False)

    axes[0, 0].set_ylabel("held-out likelihood")
    axes[1, 0].set_ylabel("generalization gap\n(train − held-out)")
    axes[0, 0].set_title("Global penalty (β) axis, one step lighter", loc="left")
    axes[0, 1].set_title("Interaction-multiplier axis, one step lighter", loc="left")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=7)

    # diamond marker legend, once
    from matplotlib.lines import Line2D
    diamond = Line2D([0], [0], marker=NEW_MARKER, color="none", markerfacecolor="grey",
                      markeredgecolor="white", markersize=8, label="wave 2 (new, lighter)")
    axes[0, 1].legend(handles=[diamond], frameon=False, loc="upper right", fontsize=7)

    # §3.6 direction-of-goodness cues, once per row (margin text, not per panel)
    axes[0, 0].text(-0.02, 1.10, "higher = better", transform=axes[0, 0].transAxes,
                     fontsize=6.5, color="#555555", ha="left")
    axes[1, 0].text(-0.02, 1.10, "lower = better", transform=axes[1, 0].transAxes,
                     fontsize=6.5, color="#555555", ha="left")

    _panel_letter(axes[0, 0], "a")
    _panel_letter(axes[0, 1], "b")

    fig.suptitle("Going lighter than mult-d-grid's tested floor: helps on the β axis, reverses on the multiplier axis",
                 fontsize=9, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
