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

`read_csv_usable` deliberately accepts BOTH state=='finished' rows and flagged backfills
(heldout_backfilled==True with a recovered value) -- it is not finished-only. Wave2 has zero
backfilled rows (8/8 finished), but the COMBINED analysis is not backfill-free: four wave-1
(mult-d-grid) rows feeding the D=300 cells used here ARE backfilled (the old tuned point's
seed-1 side, mult=2's seed-0 side, and both seeds of mult=5 -- see the r3 report's caveats
section for run ids and why). D=614 is fully backfill-free across every cell used.

    python analysis/r3_penalty_edge_extension.py

OUTPUTS:
  analysis/r3_summary.json               - curated per-(D,mult,beta) stats + provenance
  analysis/reports/fig_r3_penalty_edge_extension.png - the verdict figure
Regenerates the <!-- BEGIN result-1 --> (beta-axis table) and <!-- BEGIN result-2 -->
(multiplier-axis table) blocks in reports/r3-penalty-edge-extension.md, following
scaling_report.py's (r2) update_report_block pattern -- the two data tables are producer-owned;
everything outside the marker pairs (prose, caveats, the "what it says" discussion) is
hand-written and untouched by a re-run. Re-running this script twice on unchanged
grid.csv/grid_wave2.csv must produce an EMPTY diff on the report file.
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
FIG = HERE / "reports" / "fig_r3_penalty_edge_extension.png"
REPORT = HERE / "reports" / "r3-penalty-edge-extension.md"
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


def _find_cell(cells: list[dict], D: int, mult: float, beta: float) -> dict | None:
    for c in cells:
        if c["D"] == D and abs(c["mult"] - mult) < 1e-6 and abs(c["beta"] - beta) < 1e-9:
            return c
    return None


def update_report_block(cells: list[dict]) -> None:
    """Regenerate the beta-axis (result-1) and multiplier-axis (result-2) tables in
    reports/r3-penalty-edge-extension.md from this run's `cells`. Mirrors
    scaling_report.py's update_report_block (r2) / model_selection.py's (r1): a producer-owned
    <!-- BEGIN result-N --> / <!-- END result-N --> block pair per table; everything outside
    the markers (prose, caveats) is hand-written and untouched.
    """
    if not REPORT.exists():
        return
    text = REPORT.read_text()

    # result-1: beta axis (mult=1 fixed), heavy -> light, with Delta(new-old) and its z-score
    # against the COMBINED SEM of the old (beta=3e-4) and new (beta=1e-4) two-seed cells --
    # never the new cell's own SEM alone (see the Copilot-flagged 2026-09 correction).
    beta_lines = ["| D | β=1e-3 | β=3e-4 (old tuned pt.) | β=1e-4 (new) | Δ (new − old) |",
                  "|---|---|---|---|---|"]
    for D in (300, 614):
        heavy = _find_cell(cells, D, 1.0, 0.0010)
        old = _find_cell(cells, D, 1.0, 0.0003)
        new = _find_cell(cells, D, 1.0, 0.0001)
        delta = new["heldout_mean"] - old["heldout_mean"]
        combined_sem = np.sqrt(old["heldout_sem"] ** 2 + new["heldout_sem"] ** 2)
        z = delta / combined_sem
        beta_lines.append(
            f"| {D} | {heavy['heldout_mean']:.4f} | {old['heldout_mean']:.4f} | "
            f"**{new['heldout_mean']:.4f}** | +{delta:.4f} (z≈{z:.1f}) |"
        )
    block1 = "<!-- BEGIN result-1 -->\n" + "\n".join(beta_lines) + "\n<!-- END result-1 -->"
    text = re.sub(r"<!-- BEGIN result-1 -->.*?<!-- END result-1 -->", block1, text, flags=re.S)

    # result-2: interaction-multiplier axis (beta=3e-4 fixed), mult 10 -> 0.5; mult=1 bolded
    # as the old tuned point (and, per r3's verdict, the interior peak on this axis).
    mult_lines = ["| D | mult=10 | 5 | 2 | 1 (old tuned pt.) | 0.5 (new) |",
                  "|---|---|---|---|---|---|"]
    for D in (300, 614):
        vals = [_find_cell(cells, D, m, 0.0003) for m in (10.0, 5.0, 2.0, 1.0, 0.5)]
        cols = [f"**{v['heldout_mean']:.4f}**" if m == 1.0 else f"{v['heldout_mean']:.4f}"
                for m, v in zip((10.0, 5.0, 2.0, 1.0, 0.5), vals)]
        mult_lines.append(f"| {D} | " + " | ".join(cols) + " |")
    block2 = "<!-- BEGIN result-2 -->\n" + "\n".join(mult_lines) + "\n<!-- END result-2 -->"
    text = re.sub(r"<!-- BEGIN result-2 -->.*?<!-- END result-2 -->", block2, text, flags=re.S)

    REPORT.write_text(text)


def main() -> None:
    orig = read_csv_usable(GRID_CSV)
    wave2 = read_csv_usable(WAVE2_CSV)
    all_rows = orig + wave2

    with WAVE2_CSV.open() as f:
        n_wave2_total = sum(1 for _ in csv.DictReader(f))   # actual pulled row count, not assumed
    n_wave2_finished = sum(1 for r in wave2 if r["state"] == "finished")
    n_wave2_backfilled = sum(1 for r in wave2 if str(r.get("heldout_backfilled")).lower() == "true")
    if n_wave2_total != n_wave2_finished:
        print(f"WARNING: grid_wave2.csv has {n_wave2_total} rows but only {n_wave2_finished} "
              f"finished -- re-run pull_wave2_grid.py once the rest complete before trusting r3's "
              f"numbers as final.")

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

    used_settings = set(beta_axis_settings) | set(mult_axis_settings)
    n_orig_backfilled_used = sum(
        1 for r in orig
        if r["D_nom"] in (300, 614) and (r["mult"], r["beta"]) in used_settings
        and str(r.get("heldout_backfilled")).lower() == "true"
    )

    summary = {
        "_meta": build_meta("analysis/r3_penalty_edge_extension.py", WANDB_GROUPS, study_root=STUDY),
        "note": (f"wave2 pulled via W&B GraphQL from the sandbox (wandb.Api() is blocked there); "
                 f"see analysis/pull_wave2_grid.py docstring. {n_wave2_finished}/{n_wave2_total} "
                 f"wave2 rows state=='finished', {n_wave2_backfilled} wave2 rows backfilled. "
                 f"The COMBINED analysis (wave2 + wave1) is not backfill-free: {n_orig_backfilled_used} "
                 f"wave-1 (mult-d-grid) row(s) used in these cells are heldout_backfilled=True -- see "
                 f"the r3 report's caveats section for which cell and why."),
        "progress": {"wave2_total": n_wave2_total, "wave2_finished": n_wave2_finished,
                     "wave2_backfilled": n_wave2_backfilled,
                     "wave1_backfilled_rows_used": n_orig_backfilled_used},
        "cells": cells,
        "gru_reference": GRU,
        "rl_baseline": RL_BASELINE,
    }
    with (HERE / "r3_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {HERE / 'r3_summary.json'} — {len(cells)} cells")

    make_figure(all_rows, beta_axis_settings, mult_axis_settings)
    print(f"wrote {FIG}")

    update_report_block(cells)
    print(f"updated {REPORT}")


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
