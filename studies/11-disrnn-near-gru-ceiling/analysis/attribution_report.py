"""r1 producer: staged attribution of the residual GRU gap.

Reads the committed analysis/grid.csv (this study's 3 stages) and
analysis/reference_points.csv (study 06 tuned point + study 01 GRU ceiling),
computes the gap-closure attribution table, and renders
analysis/fig_gru_ceiling_attribution.png.

    python analysis/attribution_report.py     # offline; no WANDB_API_KEY needed

Run `python analysis/pull_grid.py` first to refresh both CSVs from W&B.

NOTE on the GRU reference: study 06's analysis/scaling_report.py hardcodes
GRU = {..., 614: 0.7268}. That constant predates a later re-run of study 01's
nxd-grid (produced_at 2026-09-01 per nxd_scaling.json's _meta) and is now STALE.
The live re-pull in reference_points.csv gives 0.7290 (3 seeds, SD<0.0001,
all early-stopped at step 90505) -- this is the number used here. This means
the residual gap this study attributes against is -0.0069, not the commonly
quoted -0.0047 (see the r1 report for the full explanation).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE / "reports" / "r1-near-gru-attribution.md"

RL_BASELINE = 0.7170
FOCAL = "#1f77b4"
COLLAPSED = "#c44e52"
GRU_COLOR = "#333333"
RL_COLOR = "#888888"

# Table-row labels for the r1 report's "Gap-closure attribution" section, parallel to
# `labels` (the plot's multi-line x-tick labels) below. index 1 (stage1) carries the
# footnote marker because it is effectively n=1 (seed 0 excluded, collapsed).
REPORT_ROW_LABELS = [
    "disRNN tuned (study 06, baseline)",
    "+stage1 (interaction penalty \u2192 0)\u1d43",
    "+stage1+2 (+ linear update net)",
    "+stage1+2+3 (+ latent 5\u219232)",
]


def load_grid():
    with (HERE / "grid.csv").open() as f:
        rows = list(csv.DictReader(f))
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r["stage"], {})[int(r["seed"])] = dict(
            heldout=float(r["heldout_ll"]), checkpoint=float(r["checkpoint_ll"]))
    return by_stage


def load_reference():
    with (HERE / "reference_points.csv").open() as f:
        rows = list(csv.DictReader(f))
    s06 = {int(r["seed"]): float(r["heldout_ll"]) for r in rows if r["reference"] == "study06_tuned_D614_mult1_beta3e-4"}
    gru = [float(r["heldout_ll"]) for r in rows if r["reference"] == "gru_ceiling_H256_D614"]
    return s06, gru


def update_report_block(means: list[float], gru_mean: float, baseline_gap: float) -> None:
    """Regenerate the <!-- BEGIN result-1 --> gap-closure table in r1 from computed means.

    Mirrors the pattern in studies/06-disrnn-operating-point-at-scale/analysis/
    model_selection.py:137-148 and scaling_report.py:528-542.
    """
    lines = ["| stage (cumulative) | mean held-out LL | remaining gap to GRU | "
             "gap closed vs. baseline | % of baseline gap closed |",
             "|---|---|---|---|---|"]
    is_baseline = [True, False, False, False]
    for row_label, m, baseline_row in zip(REPORT_ROW_LABELS, means, is_baseline):
        gap = gru_mean - m
        closure = baseline_gap - gap
        pct = closure / baseline_gap * 100
        pct_str = "0%" if baseline_row else f"**{pct:+.1f}%**"
        closure_str = "\u2014" if baseline_row else f"{closure:+.4f}"
        # "remaining gap to GRU" is displayed model-minus-GRU (negative), matching the
        # report's residual-gap sign convention throughout -- i.e. -gap, not +gap.
        lines.append(f"| {row_label} | {m:.4f} | {-gap:+.4f} | {closure_str} | {pct_str} |")
    block = "<!-- BEGIN result-1 -->\n" + "\n".join(lines) + "\n<!-- END result-1 -->"
    if not REPORT.exists():
        return
    text = REPORT.read_text()
    new = re.sub(r"<!-- BEGIN result-1 -->.*?<!-- END result-1 -->", block, text, flags=re.S)
    REPORT.write_text(new)
    print(f"regenerated result-1 block in {REPORT.name}")


def main():
    stages = load_grid()
    s06_seeds, gru_seeds = load_reference()

    s06_mean = float(np.mean(list(s06_seeds.values())))
    gru_mean = float(np.mean(gru_seeds))
    baseline_gap = gru_mean - s06_mean

    stage1 = {k: v["heldout"] for k, v in stages["stage1-no-penalty"].items()}
    stage2 = {k: v["heldout"] for k, v in stages["stage2-linear-update-net"].items()}
    stage3 = {k: v["heldout"] for k, v in stages["stage3-wide-latent"].items()}

    # seed 0 of stage1 collapsed late in training (checkpoint curve verified separately via
    # W&B history: healthy through step 87500, crashes between 87500 and 97500) -- excluded
    # from the summary trend per fig-style S1.1 (an excluded row never enters a plotted summary).
    stage1_healthy = {k: v for k, v in stage1.items() if k != 0}

    means = [s06_mean, np.mean(list(stage1_healthy.values())), np.mean(list(stage2.values())), np.mean(list(stage3.values()))]
    labels = ["disRNN\n(study 06\ntuned pt.)", "+stage1\n(no interaction\npenalty)",
              "+stage1+2\n(linear\nupdate net)", "+stage1+2+3\n(latent 5\u219232)"]
    seed_vals = [list(s06_seeds.values()), list(stage1_healthy.values()), list(stage2.values()), list(stage3.values())]

    print("=== gap-closure attribution (GRU ceiling verified fresh: %.4f) ===" % gru_mean)
    print("baseline gap (study06 tuned vs GRU) = %.4f" % baseline_gap)
    closure_pcts = []
    for lab, m in zip(labels, means):
        gap = gru_mean - m
        closure = baseline_gap - gap
        pct = closure / baseline_gap * 100
        closure_pcts.append(pct)
        print(f"  {lab.replace(chr(10),' '):45s} mean={m:.4f}  gap={gap:+.4f}  closure={closure:+.4f} ({pct:+.1f}%)")
    # Max closure across the three relaxation stages (excludes the baseline's own 0% entry
    # at index 0) -- drives the plot title below so it can't drift out of sync with the data,
    # per Copilot review on PR #163 (attribution_report.py:113 previously hardcoded "~11%"
    # against an actual max of 11.9%, which rounds to ~12%).
    max_closure_pct = max(closure_pcts[1:])

    update_report_block(means, gru_mean, baseline_gap)

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    ax.axhline(gru_mean, ls="--", color=GRU_COLOR, lw=1.4, zorder=1)
    ax.text(3.5, gru_mean + 0.0012, f"GRU ceiling (H=256): {gru_mean:.4f}", va="bottom", ha="right", fontsize=7.5, color=GRU_COLOR)
    ax.axhline(RL_BASELINE, ls=":", color=RL_COLOR, lw=1.2, zorder=1)
    ax.text(3.5, RL_BASELINE - 0.0012, f"RL baseline: {RL_BASELINE:.4f}", va="top", ha="right", fontsize=7.5, color=RL_COLOR)

    ax.plot(x, means, "-o", color=FOCAL, lw=2, ms=7, zorder=4)
    for xi, seeds in zip(x, seed_vals):
        for s in seeds:
            ax.plot([xi], [s], "o", color=FOCAL, ms=4, alpha=0.45, zorder=3)

    ax.plot([1], [stage1[0]], marker="o", mfc="none", mec=COLLAPSED, mew=1.6, ms=8, zorder=5)
    ax.annotate("seed 0: diverged after step\n~90k (excluded from trend)", xy=(1, stage1[0]),
                xytext=(2.05, stage1[0] + 0.004), fontsize=7, color=COLLAPSED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=COLLAPSED, lw=0.9))

    for xi, m in zip(x, means):
        ax.annotate(f"{m:.4f}", xy=(xi, m), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7.8, color=FOCAL)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("held-out eval likelihood")
    ax.set_title(f"Staged relaxation toward GRU capacity closes ~{round(max_closure_pct)}% of the residual gap, not more", loc="left")
    ax.set_ylim(0.672, 0.735)
    ax.set_xlim(-0.4, 4.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = HERE / "fig_gru_ceiling_attribution.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
