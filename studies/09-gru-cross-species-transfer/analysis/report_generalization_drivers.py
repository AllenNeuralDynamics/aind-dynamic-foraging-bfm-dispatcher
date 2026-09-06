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
TASK_DATA = STUDY / "analysis" / "task_design_features.json"
MAIN_FIGURE = STUDY / "analysis" / "fig_generalization_drivers.png"
ROBUSTNESS_FIGURE = STUDY / "analysis" / "fig_generalization_robustness.png"
TASK_FIGURE = STUDY / "analysis" / "fig_task_design_drivers.png"
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


def _plot_task_design(task_data: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.2), constrained_layout=True)
    columns = (
        (
            "categorical_distance.task_structure",
            "Task-structure distance from AIND",
            "task_structure_distance",
        ),
        (
            "categorical_distance.full_design",
            "Full-design distance from AIND",
            "full_design_distance",
        ),
        (
            "empirical_schedule_distance_to_grossman",
            "Empirical schedule distance from Grossman",
            "empirical_schedule_distance",
        ),
    )
    outcomes = (
        (
            "gru_d614_minus_q_bits_per_trial",
            "GRU D=614 − common Q\n(subject-balanced bits/trial)",
            "gru_d614_minus_q",
        ),
        (
            "embedding_centroid_mahalanobis",
            "External-centroid distance from source\n(4D Mahalanobis)",
            "embedding_centroid",
        ),
    )

    def nested(record: dict, path: str) -> float | None:
        value: object = record
        for key in path.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return None if value is None else float(value)

    for column, (x_path, x_label, relationship_x) in enumerate(columns):
        for row, (outcome, y_label, relationship_y) in enumerate(outcomes):
            axis = axes[row, column]
            for cohort in task_data["cohorts"].values():
                x = nested(cohort, x_path)
                if x is None:
                    continue
                color = SPECIES_COLORS[cohort["species"]]
                y = float(cohort["outcomes"][outcome])
                axis.scatter(
                    x,
                    y,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    s=78,
                    zorder=4,
                )
                _annotate(axis, x, y, cohort["label"])
            relationship = task_data["relationships"][
                f"{relationship_y}_vs_{relationship_x}"
            ]
            if row == 0:
                axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(
                f"n={relationship['n_cohorts']}; "
                f"Spearman ρ={relationship['spearman_rho']:+.2f}; "
                f"p={relationship['permutation_p_two_sided']:.3f}"
            )

    fig.legend(
        handles=_species_legend(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Task-design distance predicts embedding displacement more clearly than transfer advantage\n"
        "Categorical scores are nearest-prototype mismatch; schedule scores use complete trial-wise probabilities"
    )
    fig.savefig(TASK_FIGURE, bbox_inches="tight")
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


def _value_text(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).replace("_", " ") for item in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace("_", " ")


def _task_rows(task_data: dict) -> list[str]:
    rows = []
    for cohort in task_data["cohorts"].values():
        annotation = cohort["annotation"]
        rows.append(
            f"| [{cohort['label']}]({annotation['evidence_url']}) | "
            f"{cohort['species']} | {_value_text(annotation['schedule_family'])} | "
            f"{_value_text(annotation['arm_coupling'])} | "
            f"{_value_text(annotation['baiting'])} | "
            f"{_value_text(annotation['choice_target'])} | "
            f"{_value_text(annotation['physical_context'])} | "
            f"{_value_text(annotation['response_modality'])} | "
            f"{_value_text(annotation['reward_modality'])} | "
            f"{cohort['categorical_distance']['task_structure']:.2f} | "
            f"{cohort['categorical_distance']['full_design']:.2f} |"
        )
    return rows


def _species_rows(task_data: dict) -> list[str]:
    rows = []
    for species in SPECIES_COLORS:
        cohorts = [
            cohort
            for cohort in task_data["cohorts"].values()
            if cohort["species"] == species
        ]
        delta = np.asarray(
            [
                cohort["outcomes"]["gru_d614_minus_q_bits_per_trial"]
                for cohort in cohorts
            ],
            dtype=float,
        )
        embedding = np.asarray(
            [
                cohort["outcomes"]["embedding_centroid_mahalanobis"]
                for cohort in cohorts
            ],
            dtype=float,
        )
        task_distance = np.asarray(
            [
                cohort["categorical_distance"]["task_structure"]
                for cohort in cohorts
            ],
            dtype=float,
        )
        rows.append(
            f"| {species.capitalize()} | {len(cohorts)} | {delta.mean():+.4f} | "
            f"{np.median(delta):+.4f} | [{delta.min():+.4f}, {delta.max():+.4f}] | "
            f"{embedding.mean():.2f} | {task_distance.mean():.2f} |"
        )
    return rows


def _schedule_rows(task_data: dict) -> list[str]:
    labels = {
        "mean_arm_lag1_autocorrelation": "arm lag-1",
        "reward_gap_lag1_autocorrelation": "gap lag-1",
        "cross_arm_correlation": "cross-arm",
        "probability_change_rate": "change rate",
        "mean_absolute_arm_step_on_change": "step size",
        "mean_absolute_reward_gap": "mean |gap|",
        "mean_reward_probability": "mean p(reward)",
        "equal_probability_fraction": "equal-p fraction",
    }
    rows = []
    for name in task_data["contract"]["schedule_cohort_order"]:
        cohort = task_data["cohorts"][name]
        values = [
            cohort["schedule_features"][feature]["subject_balanced_mean"]
            for feature in labels
        ]
        rows.append(
            f"| {cohort['label']} | "
            + " | ".join(f"{value:.3f}" for value in values)
            + f" | {cohort['empirical_schedule_distance_to_grossman']:.2f} |"
        )
    return rows


def _task_relationship_rows(task_data: dict) -> list[str]:
    labels = {
        "gru_d614_minus_q_vs_task_structure_distance": "GRU614−Q vs task-structure distance",
        "embedding_centroid_vs_task_structure_distance": "embedding vs task-structure distance",
        "gru_d614_minus_q_vs_full_design_distance": "GRU614−Q vs full-design distance",
        "embedding_centroid_vs_full_design_distance": "embedding vs full-design distance",
        "gru_d614_minus_q_vs_empirical_schedule_distance": "GRU614−Q vs empirical schedule distance",
        "embedding_centroid_vs_empirical_schedule_distance": "embedding vs empirical schedule distance",
    }
    rows = []
    for key, label in labels.items():
        result = task_data["relationships"][key]
        rows.append(
            f"| {label} | {result['n_cohorts']} | {result['spearman_rho']:+.3f} | "
            f"{_fmt_interval(result['bootstrap_95_ci'])} | "
            f"{result['permutation_p_two_sided']:.4f} ({result['permutation_method']}) | "
            f"{_fmt_interval(result['leave_one_out_range'])} |"
        )
    return rows


def _feature_screen_rows(task_data: dict) -> list[str]:
    labels = {
        "mean_arm_lag1_autocorrelation": "mean arm lag-1 autocorrelation",
        "reward_gap_lag1_autocorrelation": "reward-gap lag-1 autocorrelation",
        "cross_arm_correlation": "contemporaneous cross-arm correlation",
        "probability_change_rate": "probability-change rate",
        "mean_absolute_arm_step_on_change": "mean absolute arm step when changed",
        "mean_absolute_reward_gap": "mean absolute reward gap",
        "mean_reward_probability": "mean reward probability",
        "equal_probability_fraction": "equal-probability fraction",
    }
    return [
        f"| {labels[row['feature']]} | {row['spearman_rho']:+.3f} | "
        f"{row['permutation_p_two_sided']:.4f} | {row['bh_fdr_q']:.4f} |"
        for row in task_data["schedule_feature_screen_vs_gru_d614_minus_q"]
    ]


def _result_block(data: dict, task_data: dict) -> str:
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
    task_performance = task_data["relationships"][
        "gru_d614_minus_q_vs_task_structure_distance"
    ]
    task_embedding = task_data["relationships"][
        "embedding_centroid_vs_task_structure_distance"
    ]
    full_performance = task_data["relationships"][
        "gru_d614_minus_q_vs_full_design_distance"
    ]
    full_embedding = task_data["relationships"][
        "embedding_centroid_vs_full_design_distance"
    ]
    schedule_performance = task_data["relationships"][
        "gru_d614_minus_q_vs_empirical_schedule_distance"
    ]
    schedule_embedding = task_data["relationships"][
        "embedding_centroid_vs_empirical_schedule_distance"
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
        "## Task-design meta-analysis",
        "",
        "![Task-design distance versus transfer and embedding displacement](../fig_task_design_drivers.png)",
        "",
        "The categorical analysis is outcome-blind. Task-structure distance is the equal-weight "
        "mismatch over schedule family, arm coupling, baiting, and what the subject chooses. "
        "Full-design distance adds physical context, response modality, and reward modality. "
        "Each cohort is scored against the nearest of the three task prototypes actually "
        "represented in the AIND source-training snapshot; this preserves source heterogeneity "
        "and makes Grossman a zero-distance anchor.",
        "",
        f"Task-structure distance is associated with adapted embedding-centroid displacement "
        f"(ρ={task_embedding['spearman_rho']:+.3f}, permutation "
        f"p={task_embedding['permutation_p_two_sided']:.4f}) but not with GRU advantage "
        f"(ρ={task_performance['spearman_rho']:+.3f}, "
        f"p={task_performance['permutation_p_two_sided']:.4f}). The same separation is stronger "
        f"for the full-design score: embedding ρ={full_embedding['spearman_rho']:+.3f} "
        f"(p={full_embedding['permutation_p_two_sided']:.4f}), versus GRU advantage "
        f"ρ={full_performance['spearman_rho']:+.3f} "
        f"(p={full_performance['permutation_p_two_sided']:.4f}). Thus the transferred embedding "
        "geometry carries an auditable task/apparatus-distance signal, but categorical closeness "
        "alone does not explain whether GRU beats common Q.",
        "",
        "### Species-stratified description",
        "",
        "| species | cohorts | mean GRU614−Q bits/trial | median | range | mean embedding distance | mean task distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *_species_rows(task_data),
        "",
        "These are equal-cohort descriptive summaries, not species effects. Each species is "
        "represented by only two to six studies, and task design differs systematically by "
        "species. In particular, a species contrast would currently relabel the same design and "
        "apparatus contrasts rather than isolate biology.",
        "",
        f"For the seven cohorts whose canonical adapters expose complete trial-wise probabilities "
        f"for both arms, empirical schedule distance from Grossman is negatively associated with "
        f"GRU advantage (ρ={schedule_performance['spearman_rho']:+.3f}, exact permutation "
        f"p={schedule_performance['permutation_p_two_sided']:.4f}; bootstrap 95% CI "
        f"{_fmt_interval(schedule_performance['bootstrap_95_ci'])}; leave-one-out range "
        f"{_fmt_interval(schedule_performance['leave_one_out_range'])}). Its association with "
        f"embedding-centroid distance is weak (ρ={schedule_embedding['spearman_rho']:+.3f}, "
        f"p={schedule_embedding['permutation_p_two_sided']:.4f}). The performance result is "
        "promising but small-sample: the bootstrap interval crosses zero, and Grossman defines "
        "the schedule-distance origin.",
        "",
        "### Evidence-backed categorical matrix",
        "",
        "| cohort | species | schedule | coupling | baited | choice target | context | response | reward | task distance | full distance |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|",
        *_task_rows(task_data),
        "",
        "Cohort names link to the primary Methods source used for annotation. The complete "
        "evidence note for every row, the three AIND prototypes, and the scoring contract are "
        "frozen in `analysis/task_design_features.json`.",
        "",
        "### Empirical reward-schedule features",
        "",
        "| cohort | arm lag-1 | gap lag-1 | cross-arm | change rate | step size | mean abs(gap) | mean p(reward) | equal-p fraction | distance to Grossman |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_schedule_rows(task_data),
        "",
        "Every metric is computed using only within-session transitions. Subjects are summarized "
        "first and then averaged equally within a cohort. The composite distance is the root-mean-"
        "square standardized Euclidean distance across seven predeclared features; reward-gap "
        "autocorrelation is reported but omitted from the distance because it largely duplicates "
        "the two arm-autocorrelation terms. No missing schedule is imputed.",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_task_relationship_rows(task_data),
        "",
        "### Individual schedule-feature screen",
        "",
        "| schedule feature vs GRU614−Q | Spearman ρ | exact p | BH-FDR q |",
        "|---|---:|---:|---:|",
        *_feature_screen_rows(task_data),
        "",
        "No individual schedule feature survives the eight-feature FDR correction. The composite "
        "distance result should therefore motivate preregistered tests on additional datasets, "
        "not a post-hoc claim that one schedule statistic is the mechanism.",
        "",
        "## What this version can and cannot answer",
        "",
        "Embedding distance is measured after embedding-only adaptation, so it can reflect both "
        "task structure and cohort behavior. Species, study, apparatus, reward schedule, and data "
        "volume remain confounded, and 13 cohorts are too few for a causal species effect or a "
        "stable multivariable regression. Species colors are descriptive only.",
        "",
        "The next defensible extension is to recover explicit schedules for the six currently "
        "excluded adapters, then test whether the schedule-distance relationship replicates. An "
        "LLM pairwise closeness rank remains a blinded sensitivity analysis: prompts and Methods "
        "excerpts should be frozen before exposing the model to GRU outcomes, and agreement with "
        "the auditable matrix should be reported rather than used to replace it.",
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
    task_data = json.loads(TASK_DATA.read_text())
    expected = tuple(data["contract"]["cohort_order"])
    if tuple(data["cohorts"]) != expected:
        raise AssertionError("Generalization-driver cohort order drifted")
    if set(cohort["species"] for cohort in data["cohorts"].values()) != set(
        SPECIES_COLORS
    ):
        raise AssertionError("Species color map does not match frozen cohorts")
    _plot_main(data)
    _plot_robustness(data)
    _plot_task_design(task_data)
    body = _result_block(data, task_data)
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + body + "\n" + text[end_start:])
    print(
        f"Wrote {MAIN_FIGURE}, {ROBUSTNESS_FIGURE}, {TASK_FIGURE}, and {REPORT}"
    )


if __name__ == "__main__":
    main()
