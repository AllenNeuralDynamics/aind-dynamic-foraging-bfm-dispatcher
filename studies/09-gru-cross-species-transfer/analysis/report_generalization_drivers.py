"""Render Result 3 from frozen Study 09 generalization-driver estimates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import apply_presentation_style  # noqa: E402


DATA = STUDY / "analysis" / "generalization_drivers.json"
MAIN_FIGURE = STUDY / "analysis" / "fig_generalization_drivers.png"
ROBUSTNESS_FIGURE = STUDY / "analysis" / "fig_generalization_robustness.png"
REPORT = STUDY / "analysis" / "reports" / "r3-generalization-drivers.md"
START = "<!-- BEGIN result-3 -->"
END = "<!-- END result-3 -->"
SPECIES_COLORS = {
    "mouse": "#4C72B0",
    "rat": "#DD8452",
    "macaque": "#C44E52",
    "human": "#8172B3",
}


def _summary(cohort: dict, key: str) -> float:
    return float(cohort["summary"][key]["mean"])


def _seed_values(cohort: dict, key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in cohort["seeds"]])


def _annotate(axis: plt.Axes, x: float, y: float, label: str) -> None:
    axis.annotate(
        label,
        (x, y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8.5,
        alpha=0.9,
    )


def _species_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=9,
            label=species.capitalize(),
        )
        for species, color in SPECIES_COLORS.items()
    ]


def _plot_main(data: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), constrained_layout=True)
    cohorts = data["cohorts"]

    for cohort in cohorts.values():
        color = SPECIES_COLORS[cohort["species"]]
        centroid = _seed_values(cohort, "embedding_centroid_mahalanobis")
        delta = _seed_values(cohort, "gru_d614_minus_q_bits_per_trial")
        q_likelihood = _summary(
            cohort, "q_subject_balanced_normalized_likelihood"
        )
        gru_likelihood = _seed_values(
            cohort, "gru_d614_subject_balanced_normalized_likelihood"
        )
        q_predictability = _summary(cohort, "q_bits_above_chance")

        axes[0].plot(centroid, delta, color=color, alpha=0.20, linewidth=0.9)
        axes[0].scatter(centroid, delta, color=color, alpha=0.32, s=24)
        axes[0].scatter(
            centroid.mean(),
            delta.mean(),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=75,
            zorder=4,
        )
        _annotate(
            axes[0], centroid.mean(), delta.mean(), cohort["label"]
        )

        axes[1].plot(
            [q_likelihood] * 3,
            gru_likelihood,
            color=color,
            alpha=0.20,
            linewidth=0.9,
        )
        axes[1].scatter(
            [q_likelihood] * 3,
            gru_likelihood,
            color=color,
            alpha=0.32,
            s=24,
        )
        axes[1].scatter(
            q_likelihood,
            gru_likelihood.mean(),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=75,
            zorder=4,
        )
        _annotate(
            axes[1], q_likelihood, gru_likelihood.mean(), cohort["label"]
        )

        axes[2].plot(
            [q_predictability] * 3,
            delta,
            color=color,
            alpha=0.20,
            linewidth=0.9,
        )
        axes[2].scatter(
            [q_predictability] * 3,
            delta,
            color=color,
            alpha=0.32,
            s=24,
        )
        axes[2].scatter(
            q_predictability,
            delta.mean(),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            s=75,
            zorder=4,
        )
        _annotate(
            axes[2], q_predictability, delta.mean(), cohort["label"]
        )

    relation = data["relationships"][
        "gru_d614_minus_q_vs_embedding_centroid"
    ]
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xlabel("External-centroid distance from source\n(4D Mahalanobis)")
    axes[0].set_ylabel("GRU D=614 − common Q\n(subject-balanced bits/trial)")
    axes[0].set_title(
        "Transfer advantage vs embedding displacement\n"
        f"Spearman ρ={relation['spearman_rho']:+.2f}, "
        f"permutation p={relation['permutation_p_two_sided']:.3f}"
    )

    all_likelihoods = [
        value
        for cohort in cohorts.values()
        for value in [
            _summary(cohort, "q_subject_balanced_normalized_likelihood"),
            *_seed_values(
                cohort, "gru_d614_subject_balanced_normalized_likelihood"
            ),
        ]
    ]
    lower = min(all_likelihoods) - 0.01
    upper = max(all_likelihoods) + 0.01
    axes[1].plot([lower, upper], [lower, upper], color="#777777", linestyle="--")
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("Common-Q normalized likelihood")
    axes[1].set_ylabel("GRU D=614 normalized likelihood")
    axes[1].set_title("Absolute held-out predictability\n(identity line = equal performance)")

    coupled = data["relationships"][
        "gru_d614_minus_q_vs_common_q_predictability"
    ]
    axes[2].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Common-Q predictability (bits above chance)")
    axes[2].set_ylabel("GRU D=614 − common Q\n(subject-balanced bits/trial)")
    axes[2].set_title(
        "Advantage vs common-Q predictability†\n"
        f"Spearman ρ={coupled['spearman_rho']:+.2f}, "
        f"permutation p={coupled['permutation_p_two_sided']:.3f}"
    )

    fig.legend(
        handles=_species_legend(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Study 09 external transfer: embedding displacement and baseline predictability\n"
        "Large labeled points are cohort means; small points are paired source seeds"
    )
    fig.savefig(MAIN_FIGURE, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness(data: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), constrained_layout=True)
    cohorts = data["cohorts"]

    for cohort in cohorts.values():
        color = SPECIES_COLORS[cohort["species"]]
        delta = _seed_values(cohort, "gru_d614_minus_q_bits_per_trial")
        median_distance = _seed_values(
            cohort, "embedding_median_subject_mahalanobis"
        )
        centroid = _seed_values(cohort, "embedding_centroid_mahalanobis")
        scaling = _seed_values(cohort, "gru_d614_minus_d10_bits_per_trial")
        for axis, x, y in (
            (axes[0], median_distance, delta),
            (axes[1], centroid, scaling),
        ):
            axis.plot(x, y, color=color, alpha=0.20, linewidth=0.9)
            axis.scatter(x, y, color=color, alpha=0.32, s=24)
            axis.scatter(
                x.mean(),
                y.mean(),
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=75,
                zorder=4,
            )
            _annotate(axis, x.mean(), y.mean(), cohort["label"])

    median_relation = data["relationships"][
        "gru_d614_minus_q_vs_embedding_median_subject_distance"
    ]
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Median subject distance from source\n(4D Mahalanobis)")
    axes[0].set_ylabel("GRU D=614 − common Q\n(subject-balanced bits/trial)")
    axes[0].set_title(
        "Robustness: individual-subject distance\n"
        f"Spearman ρ={median_relation['spearman_rho']:+.2f}, "
        f"p={median_relation['permutation_p_two_sided']:.3f}"
    )

    scaling_relation = data["relationships"][
        "gru_d614_minus_d10_vs_embedding_centroid"
    ]
    axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xlabel("External-centroid distance from source\n(4D Mahalanobis)")
    axes[1].set_ylabel("GRU D=614 − GRU D=10\n(subject-balanced bits/trial)")
    axes[1].set_title(
        "Does source-population scaling help distant tasks?\n"
        f"Spearman ρ={scaling_relation['spearman_rho']:+.2f}, "
        f"p={scaling_relation['permutation_p_two_sided']:.3f}"
    )

    fig.legend(
        handles=_species_legend(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Embedding-distance robustness and source-D scaling")
    fig.savefig(ROBUSTNESS_FIGURE, bbox_inches="tight")
    plt.close(fig)


def _fmt_interval(values: list[float]) -> str:
    return f"[{values[0]:+.2f}, {values[1]:+.2f}]"


def _relationship_rows(data: dict) -> list[str]:
    labels = {
        "gru_d614_minus_q_vs_embedding_centroid": (
            "GRU614−Q vs embedding centroid distance"
        ),
        "gru_d614_minus_q_vs_embedding_median_subject_distance": (
            "GRU614−Q vs median subject embedding distance"
        ),
        "gru_d614_minus_q_vs_common_q_predictability": (
            "GRU614−Q vs common-Q predictability†"
        ),
        "gru_d614_minus_d10_vs_embedding_centroid": (
            "GRU614−GRU10 vs embedding centroid distance"
        ),
    }
    rows = []
    for key, label in labels.items():
        result = data["relationships"][key]
        rows.append(
            f"| {label} | {result['spearman_rho']:+.3f} | "
            f"{_fmt_interval(result['bootstrap_95_ci'])} | "
            f"{result['permutation_p_two_sided']:.4f} | "
            f"{_fmt_interval(result['leave_one_out_range'])} |"
        )
    return rows


def _cohort_rows(data: dict) -> list[str]:
    rows = []
    for cohort in data["cohorts"].values():
        rows.append(
            f"| {cohort['label']} | {cohort['species']} | {cohort['n_subjects']} | "
            f"{_summary(cohort, 'q_subject_balanced_normalized_likelihood'):.4f} | "
            f"{_summary(cohort, 'gru_d614_subject_balanced_normalized_likelihood'):.4f} | "
            f"{_summary(cohort, 'gru_d614_minus_q_bits_per_trial'):+.4f} | "
            f"{_summary(cohort, 'gru_d614_minus_q_mean_subject_normalized_likelihood'):+.4f} | "
            f"{_summary(cohort, 'embedding_centroid_mahalanobis'):.2f} | "
            f"{_summary(cohort, 'embedding_median_subject_mahalanobis'):.2f} | "
            f"{_summary(cohort, 'gru_d614_minus_d10_bits_per_trial'):+.4f} |"
        )
    return rows


def _result_block(data: dict) -> str:
    deltas = {
        cohort["label"]: _summary(
            cohort, "gru_d614_minus_q_bits_per_trial"
        )
        for cohort in data["cohorts"].values()
    }
    positive = [name for name, value in deltas.items() if value > 0]
    negative = [name for name, value in deltas.items() if value < 0]
    centroid = data["relationships"][
        "gru_d614_minus_q_vs_embedding_centroid"
    ]
    q_relation = data["relationships"][
        "gru_d614_minus_q_vs_common_q_predictability"
    ]
    scaling = data["relationships"][
        "gru_d614_minus_d10_vs_embedding_centroid"
    ]
    lines = [
        "[regenerated by `analysis/report_generalization_drivers.py` — do not edit by hand]",
        "",
        "## First-pass result",
        "",
        "![Generalization versus embedding distance and common-Q predictability](../fig_generalization_drivers.png)",
        "",
        "Each cohort is one equal-weight cross-study unit. Performance is the arithmetic "
        "mean held-out log likelihood across subjects, converted to bits per trial. "
        "Embedding distance is calculated in the full four-dimensional space, separately "
        "for each source seed. Large labeled points average the three paired seeds; small "
        "points show the seed-specific values. Species is descriptive rather than an "
        "inferential grouping because species, study, and task design are confounded.",
        "",
        f"The D=614 GRU has higher subject-balanced mean log likelihood than common Q in "
        f"{len(positive)} cohorts ({', '.join(positive)}) and lower mean log likelihood in "
        f"{len(negative)} ({', '.join(negative)}). This direction summary does not replace "
        "the paired subject tests in Result 1.",
        "",
        "The additive log-score estimand is primary for cross-task comparison. Zid is the "
        "only direction reversal under the mean subject normalized-likelihood difference: "
        "its log-score difference is positive, whereas its mean normalized-likelihood "
        "difference is negative, consistent with Result 1. Both values are retained below.",
        "",
        f"Across the 13 cohort means, GRU advantage and embedding-centroid distance have "
        f"Spearman ρ={centroid['spearman_rho']:+.3f} "
        f"(bootstrap 95% CI {_fmt_interval(centroid['bootstrap_95_ci'])}; "
        f"two-sided permutation p={centroid['permutation_p_two_sided']:.4f}). "
        f"The leave-one-cohort-out range is "
        f"{_fmt_interval(centroid['leave_one_out_range'])}.",
        "",
        "The identity plot is the primary view of baseline predictability. The right panel "
        "shows the requested GRU-minus-Q value against common Q, but its correlation is "
        "mathematically coupled because Q appears on both axes. It is therefore descriptive, "
        f"even though its observed ρ is {q_relation['spearman_rho']:+.3f}.",
        "",
        "![Robustness and source-population scaling](../fig_generalization_robustness.png)",
        "",
        "Median individual-subject embedding distance tests whether the centroid result is "
        "hiding a dispersed or bimodal cohort. The scaling panel asks whether increasing "
        "the source population from D=10 to D=614 helps cohorts that land farther from the "
        f"source embedding distribution. Its cross-cohort Spearman ρ is "
        f"{scaling['spearman_rho']:+.3f} "
        f"(permutation p={scaling['permutation_p_two_sided']:.4f}).",
        "",
        "### Cohort estimates",
        "",
        "| cohort | species | subjects | common Q likelihood | GRU614 likelihood | GRU614−Q bits/trial | mean subject Δ likelihood | centroid distance | median subject distance | GRU614−GRU10 bits/trial |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_cohort_rows(data),
        "",
        "Normalized likelihoods in this table are `exp(mean subject log likelihood)`, "
        "not trial-pooled values. This prevents large cohorts or long sessions from "
        "dominating a cross-study comparison.",
        "",
        "### Cross-cohort sensitivity",
        "",
        "| relationship | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|",
        *_relationship_rows(data),
        "",
        "† The common-Q relationship shares Q between the horizontal axis and the "
        "GRU-minus-Q vertical axis. Its correlation is not an independent test of whether "
        "intrinsically easier tasks transfer better.",
        "",
        "## What this version can and cannot answer",
        "",
        "This first version tests geometry and baseline predictability using already frozen "
        "results. Embedding distance is measured after embedding-only adaptation, so it can "
        "reflect both task structure and cohort behavior. It is not yet a pure task-distance "
        "measure. Thirteen cohorts are also too few, and too confounded, to estimate a causal "
        "species effect.",
        "",
        "The next revision should add two task-design layers: (1) quantitative reward-schedule "
        "features, including persistence of `p_right - p_left`, arm coupling, switch hazard, "
        "step size, and reward gap; and (2) an evidence-backed categorical matrix for baiting, "
        "schedule family, action versus stimulus choice, restraint, response modality, and "
        "reward modality. Species and split/data-volume variables should remain separate from "
        "the task-distance score. An LLM-derived pairwise rank should be a blinded sensitivity "
        "analysis only after the auditable feature matrix exists.",
        "",
        "## Reproduce",
        "",
        "The report and figures are offline products of the committed frozen summary:",
        "",
        "```bash",
        "make -C studies/09-gru-cross-species-transfer r3",
        "```",
    ]
    return "\n".join(lines)


def main() -> None:
    data = json.loads(DATA.read_text())
    expected = tuple(data["contract"]["cohort_order"])
    if tuple(data["cohorts"]) != expected:
        raise AssertionError("Generalization-driver cohort order drifted")
    if set(cohort["species"] for cohort in data["cohorts"].values()) != set(
        SPECIES_COLORS
    ):
        raise AssertionError("Species color map does not match frozen cohorts")
    _plot_main(data)
    _plot_robustness(data)
    body = _result_block(data)
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + body + "\n" + text[end_start:])
    print(f"Wrote {MAIN_FIGURE}, {ROBUSTNESS_FIGURE}, and {REPORT}")


if __name__ == "__main__":
    main()
