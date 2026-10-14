"""r1 producer: staged attribution of the residual GRU gap.

Reads the committed analysis/grid.csv (this study's 3 stages) and
analysis/reference_points.csv (study 06 tuned point + study 01 GRU ceiling),
computes the gap-closure attribution table, and renders
analysis/fig_gru_ceiling_attribution.png.

    python analysis/attribution_report.py     # offline; no WANDB_API_KEY needed

Run `python analysis/pull_grid.py` first to refresh both CSVs from W&B.

NOTE on the GRU reference: this study intentionally compares against study 01's
H=256 capacity-grid cell (0.7290, `nxd-grid@20260623-102649`, 3 seeds, SD<0.0001,
all early-stopped at step 90505) rather than study 06's hardcoded `GRU = {...,
614: 0.7268}` constant. The two are DIFFERENT model sizes, not two readings of the
same one: 0.7268 is study 01's original fixed-H128 curve at D=614
(studies/01-gru-scaling-law/README.md), while 0.7290 is the later N x D grid's
separate H=256 cell (analysis/reports/r7-nxd-joint-scaling-grid.md). This study
needs the wider H256 comparator specifically because stage 3's capacity-match
target was always latent_size=256 (the GRU-H256-matched ceiling probe), even
though the executed stage 3 ended up downgraded to latent_size=32 -- see the
study README's downgrade note. Using H256 means the residual gap this study
attributes against is -0.0069 (vs. the H256 comparator), not the -0.0047 commonly
quoted against study 06's old H128 comparator (see the r1 report for the full
explanation).

Everything below `## Gap-closure attribution` and `## Verified numbers`/
`## Generalization gap` in the r1 report is regenerated from the two CSVs
(update_report_block / update_verified_numbers_block / update_generalization_gap_block).
The `## Verdict` prose is NOT regenerated -- written interpretation doesn't fit a
table-fill template -- but `verify_verdict_numbers()` re-derives the handful of
figures the verdict quotes (the ~12%/10.6%/11.9% closure percentages, the ~89%
remaining-gap figure, the -0.0069 baseline gap, and the 0.7218 stage1+2 mean) and
raises if any of them no longer match the text, so a CSV change that would make
the verdict stale fails `make all` instead of silently landing.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_grid import GRU_GROUPS, GRU_PROJECT, S06_GROUP, S06_PROJECT, STUDY10_GROUPS, STUDY10_PROJECT

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
    s06_ckpt = {int(r["seed"]): float(r["checkpoint_ll"]) for r in rows
                if r["reference"] == "study06_tuned_D614_mult1_beta3e-4" and r.get("checkpoint_ll")}
    gru_by_seed = {int(r["seed"]): float(r["heldout_ll"]) for r in rows if r["reference"] == "gru_ceiling_H256_D614"}
    gru = [gru_by_seed[k] for k in sorted(gru_by_seed)]
    return s06, s06_ckpt, gru


def _write_block(tag: str, lines: list[str]) -> None:
    """Rewrite one <!-- BEGIN <tag> --> ... <!-- END <tag> --> block in the r1 report.

    Shared by all three regenerated sections (result-1/2/3) so each stays a one-line
    call instead of repeating the read/sub/write boilerplate.
    """
    block = f"<!-- BEGIN {tag} -->\n" + "\n".join(lines) + f"\n<!-- END {tag} -->"
    if not REPORT.exists():
        return
    text = REPORT.read_text()
    new = re.sub(rf"<!-- BEGIN {tag} -->.*?<!-- END {tag} -->", block, text, flags=re.S)
    if new == text and f"<!-- BEGIN {tag} -->" not in text:
        raise RuntimeError(f"{REPORT.name} has no <!-- BEGIN {tag} --> marker to regenerate")
    REPORT.write_text(new)
    print(f"regenerated {tag} block in {REPORT.name}")


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
    _write_block("result-1", lines)


def update_verified_numbers_block(s06_seeds, s06_ckpt, gru_seeds, baseline_gap,
                                   stage1, stage1_healthy, stage2, stage3) -> None:
    """Regenerate the <!-- BEGIN result-2 --> "Verified numbers" table in r1.

    One row per raw data source this report depends on, so every number in the rest
    of the report is traceable back to a CSV row instead of being hand-typed.
    """
    s06_mean = float(np.mean(list(s06_seeds.values())))
    gru_mean = float(np.mean(gru_seeds))
    stage10_group = STUDY10_GROUPS["stage1-no-penalty"]
    stage20_group = STUDY10_GROUPS["stage2-linear-update-net"]
    stage30_group = STUDY10_GROUPS["stage3-wide-latent"]

    def seed_str(d: dict[int, float], collapsed: set[int] = frozenset()) -> str:
        parts = []
        for k in sorted(d):
            tag = " (COLLAPSED, excluded)" if k in collapsed else ""
            parts.append(f"seed{k}=**{d[k]:.5f}{tag}**" if tag else f"seed{k}={d[k]:.5f}")
        return ", ".join(parts)

    lines = ["| quantity | value | source |", "|---|---|---|"]
    lines.append(f"| study06 tuned point (D=614, mult=1, \u03b2=3e-4, n_steps=100000) | "
                 f"{seed_str(s06_seeds)}, **mean={s06_mean:.5f}** | "
                 f"`{S06_PROJECT.split('/')[1]}` / `{S06_GROUP}` |")
    gru_vals = "/".join(f"{v:.5f}" for v in gru_seeds)
    lines.append(f"| GRU ceiling (H=256, D=614) | {gru_vals}, **mean={gru_mean:.5f}** | "
                 f"`{GRU_PROJECT.split('/')[1]}` / `{GRU_GROUPS[0]}` |")
    lines.append(f"| baseline residual gap | **{-baseline_gap:+.5f}** | study06 tuned \u2212 GRU |")
    lines.append(f"| stage1-no-penalty | {seed_str(stage1, collapsed=set(stage1) - set(stage1_healthy))} | "
                 f"`{STUDY10_PROJECT.split('/')[1]}` / `{stage10_group}` |")
    lines.append(f"| stage2-linear-update-net | {seed_str(stage2)}, mean=**{np.mean(list(stage2.values())):.5f}** | "
                 f"`{STUDY10_PROJECT.split('/')[1]}` / `{stage20_group}` |")
    lines.append(f"| stage3-wide-latent (latent=32) | {seed_str(stage3)}, mean=**{np.mean(list(stage3.values())):.5f}** | "
                 f"`{STUDY10_PROJECT.split('/')[1]}` / `{stage30_group}` |")
    _write_block("result-2", lines)


def update_generalization_gap_block(s06_seeds, s06_ckpt, stages: dict) -> None:
    """Regenerate the <!-- BEGIN result-3 --> generalization-gap table in r1.

    gap = checkpoint_ll - heldout_ll per seed, from the same two CSVs as everything
    else in this report (reference_points.csv now also carries study06's checkpoint_ll
    -- see pull_grid.py -- so this table no longer needs a number nobody can re-derive).
    """
    def gaps(heldout: dict[int, float], checkpoint: dict[int, float]) -> dict[int, float]:
        return {k: checkpoint[k] - heldout[k] for k in heldout if k in checkpoint}

    s06_gaps = gaps(s06_seeds, s06_ckpt)
    rows = [("study06 tuned (baseline)", s06_gaps, None)]
    for stage, label in [("stage1-no-penalty", "stage1-no-penalty"),
                          ("stage2-linear-update-net", "stage2-linear-update-net"),
                          ("stage3-wide-latent", "stage3-wide-latent")]:
        heldout = {k: v["heldout"] for k, v in stages[stage].items()}
        checkpoint = {k: v["checkpoint"] for k, v in stages[stage].items()}
        rows.append((label, gaps(heldout, checkpoint), None))

    lines = ["| stage | seed0 gap | seed1 gap | mean |", "|---|---|---|---|"]
    for label, g, _ in rows:
        g0, g1 = g.get(0), g.get(1)
        if label == "stage1-no-penalty":
            # seed 0 collapsed late in training (see "Verified numbers" / gap-closure
            # tables above) -- its generalization gap is not a meaningful comparison point.
            lines.append(f"| {label} | {g0:.4f} (collapsed run, not meaningful) | {g1:.4f} | "
                         f"\u2014 (use seed1: {g1:.4f}) |")
        else:
            lines.append(f"| {label} | {g0:.4f} | {g1:.4f} | {np.mean([g0, g1]):.4f} |")
    _write_block("result-3", lines)


def verify_verdict_numbers(closure_pcts: list[float], max_closure_pct: float,
                            baseline_gap: float, stage12_mean: float) -> None:
    """Re-derive the handful of figures the (hand-written) Verdict prose quotes and
    raise if any no longer match the committed text.

    The Verdict section is interpretive narrative, not a table fill-in, so it is not
    auto-regenerated like result-1/2/3 (per the Copilot review on PR #163's suggested
    alternative: "generate ... or clearly separate as non-regenerated narrative").
    This check is the enforcement half of that: a CSV change that would make the
    verdict's numbers stale now fails `make all` instead of landing silently.
    """
    if not REPORT.exists():
        return
    full_text = REPORT.read_text()
    # Check ONLY the hand-written Verdict section (between the markers, which sit outside
    # the generated result-N blocks). Searching the whole report would let regenerated
    # table values satisfy the checks while the Verdict prose is stale.
    begin, end = "<!-- BEGIN verdict -->", "<!-- END verdict -->"
    if full_text.count(begin) != 1 or full_text.count(end) != 1 or \
            full_text.index(begin) > full_text.index(end):
        raise AssertionError(
            f"{REPORT.name}: expected exactly one '{begin}' ... '{end}' pair delimiting the "
            "Verdict section; cannot verify verdict numbers without it.")
    text = full_text.split(begin, 1)[1].split(end, 1)[0]
    checks = [
        (f"~{round(max_closure_pct)}% of the residual gap closes", "headline max-closure %"),
        (f"{closure_pcts[3]:.1f}% vs. {closure_pcts[1]:.1f}%", "stage3 vs. stage1 closure % comparison"),
        (f"~{round(100 - closure_pcts[3])}% of the residual gap remains", "remaining-gap % at latent_size=32"),
        (f"{-baseline_gap:.4f}", "baseline residual gap (-0.0069)"),
        (f"{stage12_mean:.4f}", "stage1+2 mean held-out LL (0.7218)"),
    ]
    missing = [(needle, why) for needle, why in checks if needle not in text]
    if missing:
        detail = "; ".join(f"{why}: expected {needle!r} not found in Verdict text" for needle, why in missing)
        raise AssertionError(
            f"{REPORT.name}'s Verdict section is stale relative to the CSVs -- {detail}. "
            "Verdict prose is hand-written (not auto-regenerated); update it by hand to "
            "match the recomputed numbers above, then re-run."
        )
    print("verified Verdict section numbers against recomputed values (still current)")


def main():
    stages = load_grid()
    s06_seeds, s06_ckpt, gru_seeds = load_reference()

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
    update_verified_numbers_block(s06_seeds, s06_ckpt, gru_seeds, baseline_gap,
                                   stage1, stage1_healthy, stage2, stage3)
    update_generalization_gap_block(s06_seeds, s06_ckpt, stages)
    # closure_pcts = [baseline(0%), stage1, stage1+2, stage1+2+3]; the Verdict quotes
    # stage1 (index 1) vs. stage1+2+3 (index 3), and the stage1+2 mean (index 2 of `means`).
    verify_verdict_numbers(closure_pcts, max_closure_pct, baseline_gap, means[2])

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
