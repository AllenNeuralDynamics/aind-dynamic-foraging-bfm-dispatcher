#!/usr/bin/env python
"""Render the frozen-random reservoir comparison from committed JSON."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402
from plot_style import apply_presentation_style  # noqa: E402


REFERENCE = STUDY / "reference" / "study01-trained-gru.json"
FROZEN = STUDY / "analysis" / "reservoir_results.json"
CURVE = STUDY / "analysis" / "reservoir_curve_results.json"
SUMMARY = STUDY / "analysis" / "reservoir_analysis.json"
FIGURE = STUDY / "analysis" / "fig_reservoir_vs_trained.png"
REPORT = STUDY / "analysis" / "reports" / "r1-frozen-random-reservoir.md"
START = "<!-- BEGIN result-1 -->"
END = "<!-- END result-1 -->"
NONINFERIORITY_MARGIN = -0.002
BOOTSTRAP_SEED = 20260907
N_BOOTSTRAP = 10000


def _paired_differences(data: dict) -> tuple[list[str], np.ndarray]:
    reservoir = {item["seed"]: item for item in data["reservoir"]}
    trained = {item["seed"]: item for item in data["trained_gru"]}
    seeds = sorted(set(reservoir) & set(trained))
    per_seed = []
    subject_ids = None
    for seed in seeds:
        reservoir_rows = {row["subject_id"]: row for row in reservoir[seed]["subjects"]}
        trained_rows = {row["subject_id"]: row for row in trained[seed]["subjects"]}
        if set(reservoir_rows) != set(trained_rows):
            raise ValueError(f"seed {seed} subject keys differ")
        current_ids = sorted(reservoir_rows)
        if subject_ids is None:
            subject_ids = current_ids
        elif current_ids != subject_ids:
            raise ValueError("held-out subject keys differ across seeds")
        per_seed.append(
            [
                reservoir_rows[subject_id]["likelihood"]
                - trained_rows[subject_id]["likelihood"]
                for subject_id in current_ids
            ]
        )
    return subject_ids or [], np.asarray(per_seed, dtype=float).mean(axis=0)


def _bootstrap(values: np.ndarray) -> dict:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "fraction_positive": float(np.mean(values > 0)),
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "noninferior": bool(lower > NONINFERIORITY_MARGIN),
        "n_subjects": int(len(values)),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _curve_summary(reference: dict, curve: dict) -> list[dict]:
    trained = defaultdict(dict)
    for cell in reference["cells"]:
        trained[int(cell.get("nominal_D", cell["D"]))][int(cell["seed"])] = float(
            cell["likelihood"]
        )
    reservoir = defaultdict(dict)
    realized_d = defaultdict(list)
    for cell in curve["cells"]:
        reservoir[int(cell["nominal_D"])][int(cell["seed"])] = float(
            cell["heldout_likelihood"]
        )
        realized_d[int(cell["nominal_D"])].append(int(cell["D"]))
    summary = []
    for d in sorted(reservoir):
        if sorted(reservoir[d]) != [0, 1, 2] or sorted(trained[d]) != [0, 1, 2]:
            raise ValueError(f"D={d} is missing a seed-paired curve cell")
        reservoir_values = np.asarray([reservoir[d][seed] for seed in range(3)])
        trained_values = np.asarray([trained[d][seed] for seed in range(3)])
        differences = reservoir_values - trained_values
        summary.append(
            {
                "nominal_D": d,
                "realized_D": realized_d[d],
                "reservoir_values": reservoir_values.tolist(),
                "trained_values": trained_values.tolist(),
                "reservoir_mean": float(reservoir_values.mean()),
                "reservoir_sd": float(reservoir_values.std(ddof=1)),
                "trained_mean": float(trained_values.mean()),
                "trained_sd": float(trained_values.std(ddof=1)),
                "difference_values": differences.tolist(),
                "difference_mean": float(differences.mean()),
                "difference_sd": float(differences.std(ddof=1)),
            }
        )
    return summary


def _make_figure(
    reference: dict,
    data: dict,
    curve_summary: list[dict],
    values: np.ndarray,
    stats: dict,
) -> None:
    apply_presentation_style()
    by_nominal_d = defaultdict(list)
    for cell in reference["cells"]:
        by_nominal_d[int(cell.get("nominal_D", cell["D"]))].append(cell["likelihood"])
    ds = sorted(by_nominal_d)
    means = np.asarray([np.mean(by_nominal_d[d]) for d in ds])
    sds = np.asarray([np.std(by_nominal_d[d], ddof=1) for d in ds])
    curve_by_d = {item["nominal_D"]: item for item in curve_summary}
    completed_ds = sorted(curve_by_d)
    exact_values = np.asarray(
        [
            item["heldout_likelihood"]
            for item in sorted(data["reservoir"], key=lambda x: x["seed"])
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axis = axes[0]
    axis.plot(ds, means, color="#7DB7E8", marker="o", label="Trained GRU (E=4)")
    axis.fill_between(ds, means - sds, means + sds, color="#7DB7E8", alpha=0.18)
    for d in ds:
        axis.scatter(
            [d] * len(by_nominal_d[d]),
            by_nominal_d[d],
            color="#7DB7E8",
            s=20,
            alpha=0.75,
        )
    reservoir_means = [curve_by_d[d]["reservoir_mean"] for d in completed_ds]
    reservoir_sds = [curve_by_d[d]["reservoir_sd"] for d in completed_ds]
    axis.errorbar(
        completed_ds,
        reservoir_means,
        yerr=reservoir_sds,
        marker="D",
        color="#8C2F23",
        markersize=7,
        capsize=4,
        label="Frozen random reservoir (E=4)",
    )
    for d in completed_ds:
        axis.scatter(
            [d] * 3,
            curve_by_d[d]["reservoir_values"],
            color="#CC5A49",
            s=24,
            alpha=0.7,
        )
    axis.scatter(
        [614] * len(exact_values),
        exact_values,
        marker="x",
        color="#7A4EAB",
        s=48,
        linewidths=1.5,
        label="Exact-split D=614 replication",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Source-training subjects (D)")
    axis.set_ylabel("Held-out-subject normalized likelihood")
    axis.set_title("Source-domain generalization")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    violin = axis.violinplot([values], positions=[0], widths=0.7, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#CC5A49")
        body.set_alpha(0.28)
    jitter = np.random.default_rng(17).normal(0, 0.045, size=len(values))
    axis.scatter(jitter, values, s=9, color="#8C2F23", alpha=0.45, linewidths=0)
    axis.errorbar(
        0.18,
        stats["mean"],
        yerr=[[stats["mean"] - stats["lower_95"]], [stats["upper_95"] - stats["mean"]]],
        color="black",
        marker="o",
        capsize=5,
        label="Mean and subject-bootstrap 95% CI",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.axhline(
        NONINFERIORITY_MARGIN,
        color="#555555",
        linestyle="--",
        linewidth=1,
        label="Non-inferiority margin",
    )
    axis.set_xlim(-0.55, 0.55)
    axis.set_xticks([0])
    axis.set_xticklabels(["Reservoir - trained GRU"])
    axis.set_ylabel("Paired normalized-likelihood difference")
    axis.set_title(f"Held-out subjects (n={len(values)})")
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)


def _replace_result(report: str, replacement: str) -> str:
    before, tail = report.split(START, 1)
    _, after = tail.split(END, 1)
    return before + START + "\n" + replacement.rstrip() + "\n" + END + after


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    data = json.loads(FROZEN.read_text())
    curve = json.loads(CURVE.read_text())
    subject_ids, values = _paired_differences(data)
    stats = _bootstrap(values)
    curve_summary = _curve_summary(reference, curve)
    reservoir_values = [item["heldout_likelihood"] for item in data["reservoir"]]
    trained_values = [item["heldout_likelihood"] for item in data["trained_gru"]]
    groups = data["_meta"]["wandb_groups"] + curve["_meta"]["wandb_groups"]
    summary = {
        "_meta": build_meta(
            "analysis/report_reservoir.py", groups, study_root=STUDY
        ),
        "reservoir_seed_likelihoods": reservoir_values,
        "trained_gru_seed_likelihoods": trained_values,
        "reservoir_mean": float(np.mean(reservoir_values)),
        "reservoir_sd": float(np.std(reservoir_values, ddof=1)),
        "trained_gru_mean": float(np.mean(trained_values)),
        "trained_gru_sd": float(np.std(trained_values, ddof=1)),
        "paired_subject_bootstrap": stats,
        "subject_ids": subject_ids,
        "scaling_curve": curve_summary,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    _make_figure(reference, data, curve_summary, values, stats)

    verdict = "non-inferior" if stats["noninferior"] else "not non-inferior"
    curve_rows = "\n".join(
        "| {nominal_D} | {realized_D} | {trained_mean:.6f} | "
        "{reservoir_mean:.6f} | {difference_mean:+.6f} |".format(
            nominal_D=cell["nominal_D"],
            realized_D=", ".join(str(value) for value in cell["realized_D"]),
            trained_mean=cell["trained_mean"],
            reservoir_mean=cell["reservoir_mean"],
            difference_mean=cell["difference_mean"],
        )
        for cell in curve_summary
    )
    result = f"""![Reservoir comparison](../fig_reservoir_vs_trained.png)

| nominal D | realized D by seed | trained GRU | frozen reservoir | reservoir − trained |
|---:|---|---:|---:|---:|
{curve_rows}

| D=614 comparison | mean normalized likelihood | SD across seeds |
|---|---:|---:|
| Trained GRU (H=128, D=614, E=4) | {summary['trained_gru_mean']:.6f} | {summary['trained_gru_sd']:.6f} |
| Frozen reservoir, exact split (H=128, D=614, E=4) | {summary['reservoir_mean']:.6f} | {summary['reservoir_sd']:.6f} |

Across {stats['n_subjects']} paired held-out subjects, reservoir minus trained-GRU likelihood is
**{stats['mean']:+.6f}** on average (subject-bootstrap 95% CI
**[{stats['lower_95']:+.6f}, {stats['upper_95']:+.6f}]**). The predeclared lower-bound
criterion is greater than {NONINFERIORITY_MARGIN:+.3f}; therefore the reservoir is
**{verdict}** in this first-pass source-domain test.

All 15 scaling-curve runs passed the bitwise audit: every frozen GRU and
session-conditioning parameter equals its initialized value, while source subject
embeddings and the readout changed. The D=614 curve cell uses the accepted
three-subject-drift pilot; its exact-split replication is shown separately and is
not counted as three additional independent seeds. Held-out subject keys match the
paired trained-GRU runs for all exact-split D=614 seeds."""
    REPORT.write_text(_replace_result(REPORT.read_text(), result))
    print(f"wrote {SUMMARY}, {FIGURE}, and {REPORT}")


if __name__ == "__main__":
    main()
