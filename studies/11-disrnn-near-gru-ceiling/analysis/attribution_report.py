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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

RL_BASELINE = 0.7170
FOCAL = "#1f77b4"
COLLAPSED = "#c44e52"
GRU_COLOR = "#333333"
RL_COLOR = "#888888"


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
    for lab, m in zip(labels, means):
        gap = gru_mean - m
        closure = baseline_gap - gap
        print(f"  {lab.replace(chr(10),' '):45s} mean={m:.4f}  gap={gap:+.4f}  closure={closure:+.4f} ({closure/baseline_gap*100:+.1f}%)")

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
    ax.set_title("Staged relaxation toward GRU capacity closes ~11% of the residual gap, not more", loc="left")
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
