"""Render Result 3 from frozen Study 09 generalization-driver estimates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import rankdata


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import SPECIES_COLORS, apply_presentation_style  # noqa: E402


DATA = {
    4: STUDY / "analysis" / "generalization_drivers.json",
    8: STUDY / "analysis" / "generalization_drivers_e8.json",
}
TASK_DATA = {
    4: STUDY / "analysis" / "task_design_features.json",
    8: STUDY / "analysis" / "task_design_features_e8.json",
}
LLM_DATA = STUDY / "analysis" / "llm_task_similarity_results.json"
MAIN_FIGURE = STUDY / "analysis" / "fig_generalization_drivers.png"
ROBUSTNESS_FIGURE = STUDY / "analysis" / "fig_generalization_robustness.png"
TASK_FIGURE = STUDY / "analysis" / "fig_task_design_drivers.png"
MAIN_FIGURES = {
    "primary": MAIN_FIGURE,
    "all_valid": STUDY / "analysis" / "fig_generalization_drivers_all_valid.png",
}
ROBUSTNESS_FIGURES = {
    "primary": ROBUSTNESS_FIGURE,
    "all_valid": STUDY / "analysis" / "fig_generalization_robustness_all_valid.png",
}
TASK_FIGURES = {
    "primary": TASK_FIGURE,
    "all_valid": STUDY / "analysis" / "fig_task_design_drivers_all_valid.png",
}
E8_MAIN_FIGURES = {
    "primary": STUDY / "analysis" / "fig_generalization_drivers_e8.png",
    "all_valid": STUDY / "analysis" / "fig_generalization_drivers_all_valid_e8.png",
}
R1_SCALE_MAIN_FIGURES = {
    4: {
        "primary": STUDY
        / "analysis"
        / "fig_generalization_drivers_r1_scale.png",
        "all_valid": STUDY
        / "analysis"
        / "fig_generalization_drivers_all_valid_r1_scale.png",
    },
    8: {
        "primary": STUDY
        / "analysis"
        / "fig_generalization_drivers_e8_r1_scale.png",
        "all_valid": STUDY
        / "analysis"
        / "fig_generalization_drivers_all_valid_e8_r1_scale.png",
    },
}
AUTHOR_MAIN_FIGURES = {
    4: {
        "primary": STUDY / "analysis" / "fig_generalization_drivers_author.png",
        "all_valid": STUDY
        / "analysis"
        / "fig_generalization_drivers_all_valid_author.png",
    },
    8: {
        "primary": STUDY / "analysis" / "fig_generalization_drivers_e8_author.png",
        "all_valid": STUDY
        / "analysis"
        / "fig_generalization_drivers_all_valid_e8_author.png",
    },
}
AUTHOR_R1_SCALE_MAIN_FIGURES = {
    dimension: {
        view: path.with_name(path.stem + "_r1_scale.png")
        for view, path in figures.items()
    }
    for dimension, figures in AUTHOR_MAIN_FIGURES.items()
}
E8_ROBUSTNESS_FIGURES = {
    "primary": STUDY / "analysis" / "fig_generalization_robustness_e8.png",
    "all_valid": STUDY
    / "analysis"
    / "fig_generalization_robustness_all_valid_e8.png",
}
E8_TASK_FIGURES = {
    "primary": STUDY / "analysis" / "fig_task_design_drivers_e8.png",
    "all_valid": STUDY / "analysis" / "fig_task_design_drivers_all_valid_e8.png",
}
SLIDE_FIGURE_PNG = STUDY / "analysis" / "fig_slide_r3_e8_generalization.png"
SLIDE_FIGURE_SVG = STUDY / "analysis" / "fig_slide_r3_e8_generalization.svg"
FOCUSED_FIGURE_PNG = STUDY / "analysis" / "fig_slide_e8_author_embedding_llm.png"
FOCUSED_FIGURE_SVG = STUDY / "analysis" / "fig_slide_e8_author_embedding_llm.svg"
SLIDE_PERMUTATIONS = 100_000
SLIDE_RNG_SEED = 20260915
FIGURE_SETS = {
    4: (MAIN_FIGURES, ROBUSTNESS_FIGURES, TASK_FIGURES),
    8: (E8_MAIN_FIGURES, E8_ROBUSTNESS_FIGURES, E8_TASK_FIGURES),
}
REPORT = STUDY / "analysis" / "reports" / "r3-generalization-drivers.md"
START = "<!-- BEGIN result-3 -->"
END = "<!-- END result-3 -->"
VIEW_TIERS = {
    "primary": ("primary",),
    "all_valid": ("primary", "stress_test", "descriptive_only"),
}
VIEW_LABELS = {
    "primary": "Primary-inference",
    "all_valid": "All valid (primary + stress-test)",
}
TIER_MARKERS = {
    "primary": "o",
    "stress_test": "s",
    "descriptive_only": "^",
}
TIER_LEGEND_LABELS = {
    "primary": "Primary",
    "stress_test": "Stress test",
    "descriptive_only": "Descriptive only",
}

MAIN_LABEL_OFFSETS = {
    # Cohort-specific offsets keep the compact comparison panels legible.
    # Each tuple is (left, absolute-predictability, right) in display points.
    "Beron (mouse)": ((4, -13), (4, 4), (4, -13)),
    "Chen (mouse)": ((4, 18), (4, -12), (4, 8)),
    "Costa (macaque)": ((4, 10), (4, -12), (4, 10)),
    "Eckstein (human)": ((4, -13), (4, -12), (4, 8)),
    "Findling (human)": ((4, -13), (4, 5), (4, -13)),
    "Grossman (mouse)": ((4, 7), (4, 4), (4, 7)),
    "Hattori (mouse)": ((4, 8), (4, -12), (4, 8)),
    "Lebedeva (mouse)": ((4, -12), (4, 4), (4, -12)),
    "López-Yépez (mouse)": ((4, 9), (4, -12), (4, 9)),
    "Miller (rat)": ((4, 8), (4, 4), (4, -13)),
    "Zid (human)": ((4, -18), (4, -12), (4, -11)),
}

SLIDE_LABEL_OFFSETS = {
    "Grossman (mouse)": ((4, 7), (5, 13), (4, 7), (5, 13), (5, 7)),
    "Hattori (mouse)": ((4, 18), (5, -16), (4, 17), (5, -16), (5, 10)),
    "Lebedeva (mouse)": ((4, -16), (5, -16), (4, -16), (5, -16), (5, -16)),
    "Beron (mouse)": ((4, -18), (5, -17), (4, -18), (5, -17), (5, -17)),
    "Chen (mouse)": ((4, 17), (5, 11), (4, 17), (5, 11), (5, 11)),
    "Costa (macaque)": ((4, 6), (5, 15), (4, 11), (5, 14), (5, 7)),
    "Miller (rat)": ((4, -17), (5, -16), (4, -16), (5, -16), (5, -16)),
    "Eckstein (human)": ((4, -17), (5, -19), (4, -17), (5, -19), (5, -19)),
    "Zid (human)": ((4, 10), (5, -16), (4, 11), (5, -17), (5, -18)),
    "Findling (human)": ((4, -17), (5, -16), (4, -17), (5, -17), (-105, 13)),
    "Alsiö (rat)": ((4, 11), (5, 17), (4, 11), (5, 17), (5, 22)),
    "López-Yépez (mouse)": ((4, 9), (5, 11), (4, 9), (5, 11), (5, 11)),
}


def _summary(cohort: dict, key: str) -> float:
    return float(cohort["summary"][key]["mean"])


def _seed_values(cohort: dict, key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in cohort["seeds"]])


def _annotate(
    axis: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    color: str | None = None,
    rotation: float = 0,
    offset: tuple[float, float] = (4, 4),
) -> None:
    rotation_options = (
        {"rotation_mode": "anchor", "ha": "left", "va": "bottom"}
        if rotation
        else {}
    )
    axis.annotate(
        label,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.5,
        alpha=0.9,
        color=color,
        rotation=rotation,
        **rotation_options,
    )


def _species_legend(cohorts: list[dict]) -> list[Line2D]:
    present = {cohort["species"] for cohort in cohorts}
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
        if species in present
    ]


def _tier_legend(cohorts: list[dict]) -> list[Line2D]:
    present = {cohort["analysis_tier"] for cohort in cohorts}
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=9,
            label=TIER_LEGEND_LABELS[tier],
        )
        for tier, marker in TIER_MARKERS.items()
        if tier in present
    ]


def _valid_cohorts(data: dict) -> list[dict]:
    return [
        cohort
        for cohort in data["cohorts"].values()
        if cohort["analysis_tier"] != "quarantined"
    ]


def _view_cohorts(data: dict, view: str) -> list[dict]:
    return [
        cohort
        for cohort in data["cohorts"].values()
        if cohort["analysis_tier"] in VIEW_TIERS[view]
    ]


def _relation_source(data: dict, view: str) -> dict | None:
    if view == "primary":
        return data["relationships"]
    if view == "all_valid":
        return data["sensitivity_relationships"]
    return None


def _relation_title(label: str, relation: dict | None, n_cohorts: int) -> str:
    if relation is None:
        return f"{label}\nn={n_cohorts}; descriptive"
    return (
        f"{label}\nn={relation['n_cohorts']}; Spearman ρ={relation['spearman_rho']:+.2f}, "
        f"permutation p={relation['permutation_p_two_sided']:.3f}"
    )


def _plot_seed_mean_sem(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    marker: str,
) -> None:
    def sem(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    axis.errorbar(
        x.mean(),
        y.mean(),
        xerr=sem(x),
        yerr=sem(y),
        fmt=marker,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=8.7,
        elinewidth=1.2,
        capsize=3,
        zorder=4,
    )


def _plot_main(
    data: dict,
    view: str,
    output: Path,
    *,
    r1_scale: bool = False,
    reference: str = "q",
) -> None:
    if reference not in {"q", "author"}:
        raise ValueError("reference must be 'q' or 'author'")
    apply_presentation_style()
    dimension = int(data["contract"]["subject_embedding_size"])
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2), constrained_layout=True)
    cohorts = _view_cohorts(data, view)
    if reference == "author":
        cohorts = [cohort for cohort in cohorts if cohort["author_reference"] is not None]
    relation_source = _relation_source(data, view)
    reference_label = "Bari2019" if reference == "q" else "author model"
    reference_key = f"{reference}_subject_balanced_normalized_likelihood"
    reference_bits_key = f"{reference}_bits_above_chance"
    delta_bits_key = f"gru_d614_minus_{reference}_bits_per_trial"

    for cohort in cohorts:
        color = SPECIES_COLORS[cohort["species"]]
        marker = TIER_MARKERS[cohort["analysis_tier"]]
        centroid = _seed_values(cohort, "embedding_centroid_mahalanobis")
        reference_likelihood = _summary(cohort, reference_key)
        gru_likelihood = _seed_values(
            cohort, "gru_d614_subject_balanced_normalized_likelihood"
        )
        if r1_scale:
            delta = gru_likelihood - reference_likelihood
            reference_predictability = reference_likelihood
        else:
            delta = _seed_values(cohort, delta_bits_key)
            reference_predictability = _summary(cohort, reference_bits_key)

        plot_values = (
            (axes[0], centroid, delta),
            (
                axes[1],
                np.full(len(gru_likelihood), reference_likelihood),
                gru_likelihood,
            ),
            (axes[2], np.full(len(delta), reference_predictability), delta),
        )
        for panel_index, (axis, x, y) in enumerate(plot_values):
            _plot_seed_mean_sem(axis, x, y, color, marker)
            is_left_panel = axis is axes[0]
            offset = MAIN_LABEL_OFFSETS.get(
                cohort["label"], ((4, 4), (4, 4), (4, 4))
            )[panel_index]
            if r1_scale and is_left_panel and cohort["label"] == "Costa (macaque)":
                offset = (4, -24)
            _annotate(
                axis,
                x.mean(),
                y.mean(),
                cohort["label"],
                color=color if is_left_panel else None,
                rotation=30 if is_left_panel else 0,
                offset=offset,
            )

    embedding_relationship_key = (
        f"gru_d614_minus_{reference}_normalized_likelihood_vs_embedding_centroid"
        if r1_scale
        else f"gru_d614_minus_{reference}_vs_embedding_centroid"
    )
    relation = (
        relation_source[embedding_relationship_key]
        if relation_source is not None
        else None
    )
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xlabel(
        f"External-centroid distance from source\n({dimension}D Mahalanobis)"
    )
    delta_label = (
        f"GRU E={dimension}, D=614 − {reference_label}\n"
        + (
            "(subject-balanced normalized likelihood)"
            if r1_scale
            else "(subject-balanced bits/trial)"
        )
    )
    axes[0].set_ylabel(delta_label)
    axes[0].margins(y=0.16)
    axes[0].set_title(
        _relation_title(
            "Transfer advantage vs embedding displacement", relation, len(cohorts)
        )
    )

    all_likelihoods = [
        value
        for cohort in cohorts
        for value in [
            _summary(cohort, reference_key),
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
    axes[1].set_xlabel(f"{reference_label.capitalize()} normalized likelihood")
    axes[1].set_ylabel(f"GRU E={dimension}, D=614 normalized likelihood")
    axes[1].set_title("Absolute held-out predictability\n(identity line = equal performance)")

    coupled_relationship_key = (
        (
            "gru_d614_minus_q_normalized_likelihood_vs_common_q_normalized_likelihood"
            if reference == "q"
            else "gru_d614_minus_author_normalized_likelihood_vs_author_normalized_likelihood"
        )
        if r1_scale
        else (
            "gru_d614_minus_q_vs_common_q_predictability"
            if reference == "q"
            else "gru_d614_minus_author_vs_author_predictability"
        )
    )
    coupled = (
        relation_source[coupled_relationship_key]
        if relation_source is not None
        else None
    )
    axes[2].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_xlabel(
        f"{reference_label.capitalize()} normalized likelihood"
        if r1_scale
        else f"{reference_label.capitalize()} predictability (bits above chance)"
    )
    axes[2].set_ylabel(delta_label)
    axes[2].set_title(
        _relation_title(
            f"Advantage vs {reference_label} predictability†", coupled, len(cohorts)
        )
    )

    fig.legend(
        handles=[*_species_legend(cohorts), *_tier_legend(cohorts)],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    scale_note = "mean ± SEM across three source seeds; " + (
        "R1 normalized-likelihood scale"
        if r1_scale
        else "primary additive-log-score scale"
    )
    fig.suptitle(
        f"Study 09 external transfer vs {reference_label} — E={dimension}, "
        f"{VIEW_LABELS[view]} cohorts\n"
        f"{scale_note}; labeled points are cohort means"
    )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _plot_robustness(data: dict, view: str, output: Path) -> None:
    apply_presentation_style()
    dimension = int(data["contract"]["subject_embedding_size"])
    n_columns = 2 if dimension == 4 else 1
    fig, raw_axes = plt.subplots(
        1,
        n_columns,
        figsize=(13.2 if dimension == 4 else 7.2, 6.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(raw_axes)
    cohorts = _view_cohorts(data, view)
    relation_source = _relation_source(data, view)

    for cohort in cohorts:
        color = SPECIES_COLORS[cohort["species"]]
        marker = TIER_MARKERS[cohort["analysis_tier"]]
        delta = _seed_values(cohort, "gru_d614_minus_q_bits_per_trial")
        median_distance = _seed_values(
            cohort, "embedding_median_subject_mahalanobis"
        )
        centroid = _seed_values(cohort, "embedding_centroid_mahalanobis")
        plot_values = [(axes[0], median_distance, delta)]
        if dimension == 4:
            plot_values.append(
                (
                    axes[1],
                    centroid,
                    _seed_values(cohort, "gru_d614_minus_d10_bits_per_trial"),
                )
            )
        for axis, x, y in plot_values:
            _plot_seed_mean_sem(axis, x, y, color, marker)
            _annotate(axis, x.mean(), y.mean(), cohort["label"])

    median_relation = (
        relation_source["gru_d614_minus_q_vs_embedding_median_subject_distance"]
        if relation_source is not None
        else None
    )
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xlabel(
        f"Median subject distance from source\n({dimension}D Mahalanobis)"
    )
    axes[0].set_ylabel(
        f"GRU E={dimension}, D=614 − Bari2019\n(subject-balanced bits/trial)"
    )
    axes[0].set_title(
        _relation_title(
            "Robustness: individual-subject distance",
            median_relation,
            len(cohorts),
        )
    )

    if dimension == 4:
        scaling_relation = (
            relation_source["gru_d614_minus_d10_vs_embedding_centroid"]
            if relation_source is not None
            else None
        )
        axes[1].axhline(0, color="#777777", linestyle="--", linewidth=1)
        axes[1].set_xlabel(
            "External-centroid distance from source\n(4D Mahalanobis)"
        )
        axes[1].set_ylabel(
            "GRU E=4, D=614 − GRU E=4, D=10\n(subject-balanced bits/trial)"
        )
        axes[1].set_title(
            _relation_title(
                "Does source-population scaling help distant tasks?",
                scaling_relation,
                len(cohorts),
            )
        )

    fig.legend(
        handles=[*_species_legend(cohorts), *_tier_legend(cohorts)],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"Embedding-distance robustness — E={dimension}, {VIEW_LABELS[view]} cohorts"
    )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _plot_task_design(
    task_data: dict,
    generalization_data: dict,
    view: str,
    output: Path,
) -> None:
    apply_presentation_style()
    dimension = int(task_data["contract"]["subject_embedding_size"])
    all_columns = (
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
            "Empirical schedule distance\nfrom Grossman (mouse)",
            "empirical_schedule_distance",
        ),
    )
    columns = all_columns if view == "primary" else all_columns[:2]
    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(18.5 if view == "primary" else 14.5, 11.2),
        constrained_layout=True,
    )
    cohorts = _view_cohorts(task_data, view)
    generalization_by_label = {
        cohort["label"]: cohort
        for cohort in generalization_data["cohorts"].values()
    }
    relation_source = _relation_source(task_data, view)
    outcomes = (
        (
            "gru_d614_minus_q_bits_per_trial",
            f"GRU E={dimension}, D=614 − Bari2019\n(subject-balanced bits/trial)",
            "gru_d614_minus_q",
        ),
        (
            "embedding_centroid_mahalanobis",
            f"External-centroid distance from source\n({dimension}D Mahalanobis)",
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
            for cohort in cohorts:
                x = nested(cohort, x_path)
                if x is None:
                    continue
                color = SPECIES_COLORS[cohort["species"]]
                marker = TIER_MARKERS[cohort["analysis_tier"]]
                y = _seed_values(generalization_by_label[cohort["label"]], outcome)
                x_values = np.full(len(y), x)
                _plot_seed_mean_sem(axis, x_values, y, color, marker)
                _annotate(axis, x, y.mean(), cohort["label"])
            relationship = (
                relation_source[f"{relationship_y}_vs_{relationship_x}"]
                if relation_source is not None
                else None
            )
            if row == 0:
                axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(_relation_title("", relationship, len(cohorts)).lstrip())

    fig.legend(
        handles=[*_species_legend(cohorts), *_tier_legend(cohorts)],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    subtitle = "Categorical scores are nearest-prototype mismatch"
    if view == "primary":
        subtitle += "; schedule scores use complete trial-wise probabilities"
    fig.suptitle(
        f"Task-design distance — E={dimension}, {VIEW_LABELS[view]} cohorts\n"
        f"{subtitle}; outcomes are mean ± SEM across three source seeds"
    )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _slide_relation(x: list[float], y: list[float], seed: int) -> dict:
    """Spearman statistic and two-sided permutation p for slide-only relations."""
    x_rank = rankdata(np.asarray(x, dtype=float))
    y_rank = rankdata(np.asarray(y, dtype=float))
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if len(x_rank) < 3 or denominator == 0:
        raise AssertionError("Slide correlation requires paired nonconstant values")
    observed = float((x_centered @ y_centered) / denominator)
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = SLIDE_PERMUTATIONS
    while remaining:
        batch_size = min(10_000, remaining)
        indices = np.argsort(rng.random((batch_size, len(y_rank))), axis=1)
        correlations = ((y_rank[indices] - y_rank.mean()) @ x_centered) / denominator
        extreme += int(np.sum(np.abs(correlations) >= abs(observed)))
        remaining -= batch_size
    return {
        "n_cohorts": len(x_rank),
        "spearman_rho": observed,
        "permutation_p_two_sided": (extreme + 1) / (SLIDE_PERMUTATIONS + 1),
    }


def _plot_slide_synthesis(data: dict, task_data: dict) -> None:
    """Render the requested three-row E8 synthesis for presentation use."""
    apply_presentation_style()
    if int(data["contract"]["subject_embedding_size"]) != 8:
        raise AssertionError("Slide synthesis requires E8 generalization results")
    names = tuple(data["contract"]["all_valid_sensitivity_cohorts"])
    if len(names) != 12:
        raise AssertionError("Slide synthesis expects the 12 valid cohorts")
    author_names = tuple(
        name for name in names if data["cohorts"][name]["author_reference"] is not None
    )
    if len(author_names) != 11:
        raise AssertionError("Slide author comparison expects 11 aligned references")

    full_design = {
        name: float(task_data["cohorts"][name]["categorical_distance"]["full_design"])
        for name in names
    }
    q_full_relation = _slide_relation(
        [full_design[name] for name in names],
        [
            _summary(
                data["cohorts"][name],
                "gru_d614_minus_q_subject_balanced_normalized_likelihood",
            )
            for name in names
        ],
        SLIDE_RNG_SEED,
    )
    author_full_relation = _slide_relation(
        [full_design[name] for name in author_names],
        [
            _summary(
                data["cohorts"][name],
                "gru_d614_minus_author_subject_balanced_normalized_likelihood",
            )
            for name in author_names
        ],
        SLIDE_RNG_SEED + 1,
    )

    fig = plt.figure(figsize=(16.5, 18), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 0.95))
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, :]),
    )

    def draw(
        axis: plt.Axes,
        cohort_names: tuple[str, ...],
        x_key: str,
        y_key: str,
        relation: dict,
        title: str,
        x_label: str,
        y_label: str,
        *,
        panel_index: int,
        zero_line: bool,
    ) -> None:
        for name in cohort_names:
            cohort = data["cohorts"][name]
            y = _seed_values(cohort, y_key)
            x = (
                _seed_values(cohort, "embedding_centroid_mahalanobis")
                if x_key == "embedding"
                else np.full(len(y), full_design[name])
            )
            color = SPECIES_COLORS[cohort["species"]]
            marker = TIER_MARKERS[cohort["analysis_tier"]]
            _plot_seed_mean_sem(axis, x, y, color, marker)
            offset = SLIDE_LABEL_OFFSETS.get(cohort["label"], ((4, 4),) * 5)[
                panel_index
            ]
            _annotate(
                axis,
                float(x.mean()),
                float(y.mean()),
                cohort["label"],
                color=color,
                rotation=24 if x_key == "embedding" else 0,
                offset=offset,
            )
        if zero_line:
            axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(_relation_title(title, relation, len(cohort_names)))
        axis.margins(x=0.08, y=0.16)

    q_delta = "gru_d614_minus_q_subject_balanced_normalized_likelihood"
    author_delta = "gru_d614_minus_author_subject_balanced_normalized_likelihood"
    embedding_label = "External-centroid distance from source\n(8D Mahalanobis)"
    design_label = "Full-design distance from AIND\n(equal-weight categorical mismatch)"
    draw(
        axes[0],
        names,
        "embedding",
        q_delta,
        data["sensitivity_relationships"][
            "gru_d614_minus_q_normalized_likelihood_vs_embedding_centroid"
        ],
        "A  GRU−Bari2019 vs embedding distance",
        embedding_label,
        "GRU E=8, D=614 − Bari2019\n(normalized likelihood)",
        panel_index=0,
        zero_line=True,
    )
    draw(
        axes[1],
        names,
        "full_design",
        q_delta,
        q_full_relation,
        "B  GRU−Bari2019 vs full-design distance",
        design_label,
        "GRU E=8, D=614 − Bari2019\n(normalized likelihood)",
        panel_index=1,
        zero_line=True,
    )
    draw(
        axes[2],
        author_names,
        "embedding",
        author_delta,
        data["sensitivity_relationships"][
            "gru_d614_minus_author_normalized_likelihood_vs_embedding_centroid"
        ],
        "C  GRU−author model vs embedding distance",
        embedding_label,
        "GRU E=8, D=614 − author model\n(normalized likelihood)",
        panel_index=2,
        zero_line=True,
    )
    draw(
        axes[3],
        author_names,
        "full_design",
        author_delta,
        author_full_relation,
        "D  GRU−author model vs full-design distance",
        design_label,
        "GRU E=8, D=614 − author model\n(normalized likelihood)",
        panel_index=3,
        zero_line=True,
    )
    draw(
        axes[4],
        names,
        "full_design",
        "embedding_centroid_mahalanobis",
        task_data["sensitivity_relationships"][
            "embedding_centroid_vs_full_design_distance"
        ],
        "E  Embedding distance vs full-design distance",
        design_label,
        embedding_label,
        panel_index=4,
        zero_line=False,
    )
    for axis in axes:
        axis.set_box_aspect(1)
    fig.legend(
        handles=[
            *_species_legend([data["cohorts"][name] for name in names]),
            *_tier_legend([data["cohorts"][name] for name in names]),
        ],
        loc="outside lower center",
        ncol=6,
        frameon=False,
    )
    fig.suptitle(
        "Study 09: what predicts frozen-core GRU transfer?\n"
        "E8, D=614; all 12 valid cohorts; mean ± SEM across three source seeds",
        fontsize=19,
    )
    fig.savefig(SLIDE_FIGURE_PNG, bbox_inches="tight", dpi=220)
    plt.rcParams["svg.hashsalt"] = "study09-r3-e8-generalization"
    plt.rcParams["svg.fonttype"] = "none"
    fig.savefig(SLIDE_FIGURE_SVG, bbox_inches="tight", metadata={"Date": None})
    SLIDE_FIGURE_SVG.write_text(
        "\n".join(line.rstrip() for line in SLIDE_FIGURE_SVG.read_text().splitlines())
        + "\n"
    )
    plt.close(fig)


def _plot_focused_author_embedding_llm(data: dict, llm_data: dict) -> None:
    """Render the focused E8 author-model and LLM-distance slide figure."""
    apply_presentation_style()
    if int(data["contract"]["subject_embedding_size"]) != 8:
        raise AssertionError("Focused synthesis requires E8 generalization results")

    valid_names = tuple(data["contract"]["all_valid_sensitivity_cohorts"])
    author_names = tuple(
        name for name in valid_names if data["cohorts"][name]["author_reference"] is not None
    )
    if len(author_names) != 11:
        raise AssertionError("Focused author panel expects 11 aligned references")
    llm_rows = {
        row["cohort"]: row
        for row in llm_data["ranking"]
        if row["cohort"] in valid_names
    }
    if set(llm_rows) != set(valid_names):
        raise AssertionError("LLM ranking must cover every valid E8 cohort")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), constrained_layout=True)
    left, right = axes
    author_delta = "gru_d614_minus_author_subject_balanced_normalized_likelihood"
    llm_label_offsets = {
        # The top-right human/rat pair is close in both coordinates; separate
        # their rotated labels while preserving the point locations.
        "Alsiö (rat)": (-74, 15),
        "Zid (human)": (5, -26),
    }

    for name in author_names:
        cohort = data["cohorts"][name]
        x = _seed_values(cohort, "embedding_centroid_mahalanobis")
        y = _seed_values(cohort, author_delta)
        color = SPECIES_COLORS[cohort["species"]]
        _plot_seed_mean_sem(left, x, y, color, TIER_MARKERS[cohort["analysis_tier"]])
        _annotate(
            left,
            float(x.mean()),
            float(y.mean()),
            cohort["label"],
            color=color,
            rotation=30,
            offset=SLIDE_LABEL_OFFSETS.get(cohort["label"], ((4, 4),) * 5)[2],
        )

    for name in valid_names:
        cohort = data["cohorts"][name]
        row = llm_rows[name]
        x = _seed_values(cohort, "embedding_centroid_mahalanobis")
        y = np.full(len(x), float(row["llm_task_distance_to_aind"]))
        color = SPECIES_COLORS[cohort["species"]]
        _plot_seed_mean_sem(right, x, y, color, TIER_MARKERS[cohort["analysis_tier"]])
        _annotate(
            right,
            float(x.mean()),
            float(y.mean()),
            cohort["label"],
            color=color,
            rotation=30,
            offset=llm_label_offsets.get(
                cohort["label"],
                SLIDE_LABEL_OFFSETS.get(cohort["label"], ((4, 4),) * 5)[4],
            ),
        )

    left_relation = data["sensitivity_relationships"][
        "gru_d614_minus_author_normalized_likelihood_vs_embedding_centroid"
    ]
    right_relation = llm_data["relationships"]["embedding_distance_vs_llm_task_distance"]
    left.axhline(0, color="#777777", linestyle="--", linewidth=1, zorder=1)
    left.set_xlabel("External-centroid distance from source\n(E8 Mahalanobis)")
    left.set_ylabel("GRU E8, D=614 − author model\n(normalized likelihood)")
    left.set_title(
        _relation_title("A  GRU−author model versus embedding distance", left_relation, len(author_names))
    )
    right.set_xlabel("External-centroid distance from source\n(E8 Mahalanobis)")
    right.set_ylabel("LLM task distance to AIND")
    right.set_title(
        _relation_title("B  Embedding distance versus LLM task distance", right_relation, len(valid_names))
    )
    for axis in axes:
        axis.set_box_aspect(1)
        axis.margins(x=0.10, y=0.17)
    fig.legend(
        handles=[
            *_species_legend([data["cohorts"][name] for name in valid_names]),
            *_tier_legend([data["cohorts"][name] for name in valid_names]),
        ],
        loc="outside lower center",
        ncol=6,
        frameon=False,
    )
    fig.suptitle(
        "Study 09: embedding displacement, author-model advantage, and task distance\n"
        "E8, D=614; markers are means ± SEM across three source seeds; LLM distance has one judge",
        fontsize=18,
    )
    fig.savefig(FOCUSED_FIGURE_PNG, bbox_inches="tight", dpi=220)
    plt.rcParams["svg.hashsalt"] = "study09-e8-author-embedding-llm"
    fig.savefig(FOCUSED_FIGURE_SVG, bbox_inches="tight", metadata={"Date": None})
    FOCUSED_FIGURE_SVG.write_text(
        "\n".join(line.rstrip() for line in FOCUSED_FIGURE_SVG.read_text().splitlines())
        + "\n"
    )
    plt.close(fig)


def _fmt_interval(values: list[float]) -> str:
    return f"[{values[0]:+.2f}, {values[1]:+.2f}]"


def _relationship_rows(relationships: dict) -> list[str]:
    labels = {
        "gru_d614_minus_q_vs_embedding_centroid": (
            "GRU614−Q vs embedding centroid distance"
        ),
        "gru_d614_minus_q_vs_embedding_median_subject_distance": (
            "GRU614−Q vs median subject embedding distance"
        ),
        "gru_d614_minus_q_vs_common_q_predictability": (
            "GRU614−Bari2019 vs Bari2019 predictability†"
        ),
        "gru_d614_minus_author_vs_embedding_centroid": (
            "GRU614−author vs embedding centroid distance"
        ),
        "gru_d614_minus_author_vs_author_predictability": (
            "GRU614−author vs author predictability†"
        ),
        "gru_d614_minus_q_normalized_likelihood_vs_embedding_centroid": (
            "Normalized-likelihood GRU614−Bari2019 vs embedding centroid distance"
        ),
        "gru_d614_minus_q_normalized_likelihood_vs_common_q_normalized_likelihood": (
            "Normalized-likelihood GRU614−Bari2019 vs Bari2019 likelihood†"
        ),
        "gru_d614_minus_author_normalized_likelihood_vs_embedding_centroid": (
            "Normalized-likelihood GRU614−author vs embedding centroid distance"
        ),
        "gru_d614_minus_author_normalized_likelihood_vs_author_normalized_likelihood": (
            "Normalized-likelihood GRU614−author vs author likelihood†"
        ),
        "gru_d614_minus_d10_vs_embedding_centroid": (
            "GRU614−GRU10 vs embedding centroid distance"
        ),
    }
    rows = []
    for key, label in labels.items():
        if key not in relationships:
            continue
        result = relationships[key]
        rows.append(
            f"| {label} | {result['n_cohorts']} | {result['spearman_rho']:+.3f} | "
            f"{_fmt_interval(result['bootstrap_95_ci'])} | "
            f"{result['permutation_p_two_sided']:.4f} | "
            f"{_fmt_interval(result['leave_one_out_range'])} |"
        )
    return rows


def _cohort_rows(data: dict, dimension: int) -> list[str]:
    rows = []
    for cohort in _valid_cohorts(data):
        if cohort["author_reference"] is None:
            author_values = "— | — | —"
        else:
            author_values = (
                f"{cohort['author_reference']} | "
                f"{_summary(cohort, 'author_subject_balanced_normalized_likelihood'):.4f} | "
                f"{_summary(cohort, 'gru_d614_minus_author_bits_per_trial'):+.4f}"
            )
        rows.append(
            f"| E={dimension} | {cohort['label']} | {cohort['analysis_tier'].replace('_', ' ')} | "
            f"{cohort['n_subjects']} | "
            f"{_summary(cohort, 'q_subject_balanced_normalized_likelihood'):.4f} | "
            f"{_summary(cohort, 'gru_d614_subject_balanced_normalized_likelihood'):.4f} | "
            f"{_summary(cohort, 'gru_d614_minus_q_bits_per_trial'):+.4f} | "
            f"{_summary(cohort, 'gru_d614_minus_q_mean_subject_normalized_likelihood'):+.4f} | "
            f"{author_values} | "
            f"{_summary(cohort, 'embedding_centroid_mahalanobis'):.2f} | "
            f"{_summary(cohort, 'embedding_median_subject_mahalanobis'):.2f} | "
            + (
                f"{_summary(cohort, 'gru_d614_minus_d10_bits_per_trial'):+.4f} |"
                if dimension == 4
                else "— |"
            )
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
    for name, cohort in task_data["cohorts"].items():
        if name == "tang":
            continue
        annotation = cohort["annotation"]
        rows.append(
            f"| [{cohort['label']}]({annotation['evidence_url']}) | "
            f"{cohort['analysis_tier'].replace('_', ' ')} | "
            f"{_value_text(annotation['schedule_family'])} | "
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


def _species_rows(task_data: dict, dimension: int) -> list[str]:
    rows = []
    for species in SPECIES_COLORS:
        cohorts = [
            cohort
            for cohort in _valid_cohorts(task_data)
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
            f"| E={dimension} | {species.capitalize()} | {len(cohorts)} | {delta.mean():+.4f} | "
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


def _task_relationship_rows(relationships: dict) -> list[str]:
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
        if key not in relationships:
            continue
        result = relationships[key]
        rows.append(
            f"| {label} | {result['n_cohorts']} | {result['spearman_rho']:+.3f} | "
            f"{_fmt_interval(result['bootstrap_95_ci'])} | "
            f"{result['permutation_p_two_sided']:.4f} ({result['permutation_method']}) | "
            f"{_fmt_interval(result['leave_one_out_range'])} |"
        )
    return rows


def _feature_screen_rows(task_data: dict, dimension: int) -> list[str]:
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
        f"| E={dimension} | {labels[row['feature']]} | {row['spearman_rho']:+.3f} | "
        f"{row['permutation_p_two_sided']:.4f} | {row['bh_fdr_q']:.4f} |"
        for row in task_data["schedule_feature_screen_vs_gru_d614_minus_q"]
    ]


def _result_block(
    data_by_dimension: dict[int, dict], task_data_by_dimension: dict[int, dict]
) -> str:
    data = data_by_dimension[4]
    data_e8 = data_by_dimension[8]
    task_data = task_data_by_dimension[4]
    task_data_e8 = task_data_by_dimension[8]
    primary_names = data["contract"]["primary_inference_cohorts"]
    primary_cohorts = [data["cohorts"][name] for name in primary_names]
    deltas = {
        cohort["label"]: _summary(
            cohort, "gru_d614_minus_q_bits_per_trial"
        )
        for cohort in primary_cohorts
    }
    positive = [name for name, value in deltas.items() if value > 0]
    negative = [name for name, value in deltas.items() if value < 0]
    centroid = data["relationships"][
        "gru_d614_minus_q_vs_embedding_centroid"
    ]
    q_relation = data["relationships"][
        "gru_d614_minus_q_vs_common_q_predictability"
    ]
    centroid_e8 = data_e8["relationships"][
        "gru_d614_minus_q_vs_embedding_centroid"
    ]
    q_relation_e8 = data_e8["relationships"][
        "gru_d614_minus_q_vs_common_q_predictability"
    ]
    author_centroid = data["relationships"][
        "gru_d614_minus_author_vs_embedding_centroid"
    ]
    author_relation = data["relationships"][
        "gru_d614_minus_author_vs_author_predictability"
    ]
    author_centroid_e8 = data_e8["relationships"][
        "gru_d614_minus_author_vs_embedding_centroid"
    ]
    author_relation_e8 = data_e8["relationships"][
        "gru_d614_minus_author_vs_author_predictability"
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
    task_performance_e8 = task_data_e8["relationships"][
        "gru_d614_minus_q_vs_task_structure_distance"
    ]
    task_embedding_e8 = task_data_e8["relationships"][
        "embedding_centroid_vs_task_structure_distance"
    ]
    full_performance_e8 = task_data_e8["relationships"][
        "gru_d614_minus_q_vs_full_design_distance"
    ]
    full_embedding_e8 = task_data_e8["relationships"][
        "embedding_centroid_vs_full_design_distance"
    ]
    schedule_performance = task_data["relationships"][
        "gru_d614_minus_q_vs_empirical_schedule_distance"
    ]
    schedule_embedding = task_data["relationships"][
        "embedding_centroid_vs_empirical_schedule_distance"
    ]
    schedule_performance_e8 = task_data_e8["relationships"][
        "gru_d614_minus_q_vs_empirical_schedule_distance"
    ]
    schedule_embedding_e8 = task_data_e8["relationships"][
        "embedding_centroid_vs_empirical_schedule_distance"
    ]
    valid_n = len(data["contract"]["all_valid_sensitivity_cohorts"])
    primary_n = len(primary_names)
    schedule_n = len(task_data["contract"]["schedule_cohort_order"])
    lines = [
        "[regenerated by `analysis/report_generalization_drivers.py` — do not edit by hand]",
        "",
        "## First-pass result",
        "",
        "### Slide-ready E8 synthesis",
        "",
        "![E8 normalized-likelihood transfer synthesis](../fig_slide_r3_e8_generalization.png)",
        "",
        "[SVG for slides](../fig_slide_r3_e8_generalization.svg)",
        "",
        "### Focused author-model and LLM-distance view",
        "",
        "![E8 author-model advantage and LLM task distance](../fig_slide_e8_author_embedding_llm.png)",
        "",
        "[SVG for slides](../fig_slide_e8_author_embedding_llm.svg)",
        "",
        "The left panel includes the 11 cohorts with cohort-aligned author-model references; "
        "the right panel includes all 12 valid cohorts. Performance and embedding bars are "
        "SEM across three source seeds; the LLM rank has one judge and therefore no sampling bar.",
        "",
        "The Bari2019 and embedding-versus-design panels include all 12 valid cohorts. "
        "The author-model panels include 11 because Alsiö (rat) has no cohort-aligned "
        "author reference. Every title reports the cohort-level Spearman ρ and two-sided "
        "permutation p for the exact quantities plotted.",
        "",
        "### Detailed analysis figures",
        "",
        "Result 3 contains 24 detailed figures: 16 main comparison panels (Bari2019 or author "
        "reference × bits/trial or normalized-likelihood scale × E=4 or E=8 × primary "
        "or all-valid inclusion), four robustness panels, and four task-design panels. "
        "All cohort markers are means across three source seeds with SEM error bars. "
        "A fixed baseline or task-design coordinate has zero horizontal SEM by design.",
        "",
        "### Primary-inference cohorts",
        "",
        "**E=4**",
        "",
        "![Primary cohorts: generalization versus embedding distance and Bari2019 predictability](../fig_generalization_drivers.png)",
        "",
        "**E=8**",
        "",
        "![Primary cohorts, E8: generalization versus embedding distance and Bari2019 predictability](../fig_generalization_drivers_e8.png)",
        "",
        f"Primary inference uses {primary_n} equal-weight cross-study cohorts. Performance is the arithmetic "
        "mean held-out log likelihood across subjects, converted to bits per trial. "
        "Embedding distance is calculated separately in the full E=4 or E=8 space for each "
        "source seed. Every marker is the mean across the three paired source seeds, with "
        "horizontal and vertical SEM bars. Inclusion tiers are shown in separate figures, "
        f"so secondary cohorts no longer obscure the {primary_n}-cohort inference. All {valid_n} valid cohorts "
        "remain included in the numerical sensitivity table. Species is descriptive rather than an inferential "
        "grouping because species, study, and task design are confounded.",
        "",
        "### All valid cohorts (primary + stress-test)",
        "",
        "**E=4**",
        "",
        "![All valid cohorts: generalization versus embedding distance and Bari2019 predictability](../fig_generalization_drivers_all_valid.png)",
        "",
        "**E=8**",
        "",
        "![All valid cohorts, E8: generalization versus embedding distance and Bari2019 predictability](../fig_generalization_drivers_all_valid_e8.png)",
        "",
        f"This cumulative sensitivity view adds Alsiö (rat), Costa (macaque), and "
        f"López-Yépez (mouse), for {valid_n} cohorts total. No descriptive-only cohort "
        "remains after Tang (macaque) was removed, so primary + stress-test and all valid "
        "are the same set and are shown only once.",
        "",
        "### R1-scale companion: normalized-likelihood difference",
        "",
        "These companion plots use the same normalized-likelihood units as "
        "Result 1. They retain Result 3's equal-subject aggregation: each source seed's "
        "GRU value is `exp(mean subject log likelihood)` minus the matched Bari2019 value. "
        "Each point is the three-seed mean, with horizontal and vertical SEM bars. The "
        "Bari2019 baseline is shared across source seeds, so its horizontal SEM is zero in "
        "the two Bari2019-axis panels. Panel titles report cross-cohort Spearman ρ and "
        "two-sided permutation p for the exact plotted normalized-likelihood quantities. "
        "The bits-per-trial plots above remain primary for additive cross-task inference.",
        "",
        "#### Primary-inference cohorts",
        "",
        "**E=4**",
        "",
        "![Primary cohorts on the R1 normalized-likelihood scale](../fig_generalization_drivers_r1_scale.png)",
        "",
        "**E=8**",
        "",
        "![Primary cohorts, E8, on the R1 normalized-likelihood scale](../fig_generalization_drivers_e8_r1_scale.png)",
        "",
        "#### All valid cohorts (primary + stress-test)",
        "",
        "**E=4**",
        "",
        "![All valid cohorts on the R1 normalized-likelihood scale](../fig_generalization_drivers_all_valid_r1_scale.png)",
        "",
        "**E=8**",
        "",
        "![All valid cohorts, E8, on the R1 normalized-likelihood scale](../fig_generalization_drivers_all_valid_e8_r1_scale.png)",
        "",
        "### Author-model companion",
        "",
        "These panels repeat the same cross-cohort views with GRU minus the strongest "
        "model marked `author_selected` for each cohort. When a paper has multiple "
        "author-selected co-winners, the stronger trial-pooled held-out refit is used as "
        "a conservative comparator. Sensitivity-only models are excluded: Costa (macaque) "
        "therefore uses dual-rate RL plus fitted shape-choice bias, while the additional "
        "CK1 model remains a separately labeled mechanism sensitivity in Result 1. "
        "Alsiö (rat) has no cohort-aligned author model and is omitted from author-reference "
        "panels.",
        "",
        "#### Primary-inference cohorts",
        "",
        "**E=4, bits/trial**",
        "",
        "![Primary cohorts relative to author models](../fig_generalization_drivers_author.png)",
        "",
        "**E=8, bits/trial**",
        "",
        "![Primary cohorts, E8, relative to author models](../fig_generalization_drivers_e8_author.png)",
        "",
        "**E=4, normalized-likelihood difference**",
        "",
        "![Primary cohorts relative to author models on the R1 scale](../fig_generalization_drivers_author_r1_scale.png)",
        "",
        "**E=8, normalized-likelihood difference**",
        "",
        "![Primary cohorts, E8, relative to author models on the R1 scale](../fig_generalization_drivers_e8_author_r1_scale.png)",
        "",
        "#### All valid cohorts (primary + stress-test)",
        "",
        "**E=4, bits/trial**",
        "",
        "![All valid cohorts relative to author models](../fig_generalization_drivers_all_valid_author.png)",
        "",
        "**E=8, bits/trial**",
        "",
        "![All valid cohorts, E8, relative to author models](../fig_generalization_drivers_all_valid_e8_author.png)",
        "",
        "**E=4, normalized-likelihood difference**",
        "",
        "![All valid cohorts relative to author models on the R1 scale](../fig_generalization_drivers_all_valid_author_r1_scale.png)",
        "",
        "**E=8, normalized-likelihood difference**",
        "",
        "![All valid cohorts, E8, relative to author models on the R1 scale](../fig_generalization_drivers_all_valid_e8_author_r1_scale.png)",
        "",
        f"Across the primary cohorts with author references, GRU-minus-author advantage "
        f"versus embedding-centroid distance has Spearman ρ={author_centroid['spearman_rho']:+.3f} "
        f"for E=4 and ρ={author_centroid_e8['spearman_rho']:+.3f} for E=8. "
        f"The mathematically coupled GRU-minus-author versus author-predictability "
        f"relationships are ρ={author_relation['spearman_rho']:+.3f} and "
        f"ρ={author_relation_e8['spearman_rho']:+.3f}, respectively.",
        "",
        f"For E=4, the D=614 GRU has higher subject-balanced mean log likelihood than Bari2019 common Q in "
        f"{len(positive)} cohorts ({', '.join(positive)}) and lower mean log likelihood in "
        f"{len(negative)} ({', '.join(negative)}). This direction summary does not replace "
        "the paired subject tests in Result 1.",
        "",
        "The additive log-score estimand is primary for cross-task comparison. Zid (human) is the "
        "only direction reversal under the mean subject normalized-likelihood difference: "
        "its log-score difference is positive, whereas its mean normalized-likelihood "
        "difference is negative, consistent with Result 1. Both values are retained below.",
        "",
        f"Across the {primary_n} primary cohort means, GRU advantage and embedding-centroid distance have "
        f"Spearman ρ={centroid['spearman_rho']:+.3f} "
        f"(bootstrap 95% CI {_fmt_interval(centroid['bootstrap_95_ci'])}; "
        f"two-sided permutation p={centroid['permutation_p_two_sided']:.4f}). "
        f"The leave-one-cohort-out range is "
        f"{_fmt_interval(centroid['leave_one_out_range'])}.",
        f"For E=8, the corresponding association is ρ={centroid_e8['spearman_rho']:+.3f} "
        f"(bootstrap 95% CI {_fmt_interval(centroid_e8['bootstrap_95_ci'])}; "
        f"two-sided permutation p={centroid_e8['permutation_p_two_sided']:.4f}; "
        f"leave-one-out {_fmt_interval(centroid_e8['leave_one_out_range'])}).",
        "",
        "The identity plot is the primary view of baseline predictability. The right panel "
        "shows the requested GRU-minus-Bari2019 value against Bari2019, but its correlation is "
        "mathematically coupled because Q appears on both axes. It is therefore descriptive, "
        f"The observed ρ is {q_relation['spearman_rho']:+.3f} for E=4 and "
        f"{q_relation_e8['spearman_rho']:+.3f} for E=8.",
        "",
        "### Primary robustness and source-population scaling",
        "",
        "**E=4**",
        "",
        "![Primary cohorts: robustness and source-population scaling](../fig_generalization_robustness.png)",
        "",
        "**E=8**",
        "",
        "![Primary cohorts, E8: robustness](../fig_generalization_robustness_e8.png)",
        "",
        "Median individual-subject embedding distance tests whether the centroid result is "
        "hiding a dispersed or bimodal cohort. The scaling panel asks whether increasing "
        "the source population from D=10 to D=614 helps cohorts that land farther from the "
        f"source embedding distribution. Its cross-cohort Spearman ρ is "
        f"{scaling['spearman_rho']:+.3f} "
        f"(permutation p={scaling['permutation_p_two_sided']:.4f}).",
        "",
        "### All-valid robustness and scaling (primary + stress-test)",
        "",
        "**E=4**",
        "",
        "![All valid cohorts: robustness and source-population scaling](../fig_generalization_robustness_all_valid.png)",
        "",
        "**E=8**",
        "",
        "![All valid cohorts, E8: robustness](../fig_generalization_robustness_all_valid_e8.png)",
        "",
        "E8 was run only at D=614, so the source-population scaling panel is available "
        "only for E4; the E8 robustness figures therefore contain only the matched "
        "individual-subject-distance analysis.",
        "",
        "### Valid cohort estimates",
        "",
        "| space | cohort | tier | subjects | Bari2019 likelihood | GRU614 likelihood | GRU614−Bari2019 bits/trial | mean subject Δ likelihood | author reference | author likelihood | GRU614−author bits/trial | centroid distance | median subject distance | GRU614−GRU10 bits/trial |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        *_cohort_rows(data, 4),
        *_cohort_rows(data_e8, 8),
        "",
        "Normalized likelihoods in this table are `exp(mean subject log likelihood)`, "
        "not trial-pooled values. This prevents large cohorts or long sessions from "
        "dominating a cross-study comparison.",
        "",
        "### Primary cross-cohort inference — E=4",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_relationship_rows(data["relationships"]),
        "",
        "### Primary cross-cohort inference — E=8",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_relationship_rows(data_e8["relationships"]),
        "",
        "### All-valid sensitivity — E=4",
        "",
        "This sensitivity adds Alsiö (rat), Costa (macaque), and López-Yépez (mouse). "
        "It still excludes quarantined Kwak (mouse).",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_relationship_rows(data["sensitivity_relationships"]),
        "",
        "### All-valid sensitivity — E=8",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_relationship_rows(data_e8["sensitivity_relationships"]),
        "",
        "† In baseline-predictability panels, the same Bari2019 or author-model value appears on "
        "the horizontal axis and inside the GRU-minus-baseline vertical axis. These correlations "
        "are mathematically coupled and are not independent tests of whether intrinsically easier "
        "tasks transfer better.",
        "",
        "## Task-design meta-analysis",
        "",
        "### Primary task-design view",
        "",
        "**E=4**",
        "",
        "![Primary cohorts: task-design distance versus transfer and embedding displacement](../fig_task_design_drivers.png)",
        "",
        "**E=8**",
        "",
        "![Primary cohorts, E8: task-design distance versus transfer and embedding displacement](../fig_task_design_drivers_e8.png)",
        "",
        "The categorical analysis is outcome-blind. Task-structure distance is the equal-weight "
        "mismatch over schedule family, arm coupling, baiting, and what the subject chooses. "
        "Full-design distance adds physical context, response modality, and reward modality. "
        "Each cohort is scored against the nearest of the three task prototypes actually "
        "represented in the AIND source-training snapshot; this preserves source heterogeneity "
        "and makes Grossman (mouse) a zero-distance anchor.",
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
        "alone does not explain whether GRU beats Bari2019 common Q.",
        f"For E=8, task-structure distance has ρ={task_embedding_e8['spearman_rho']:+.3f} "
        f"with embedding displacement and ρ={task_performance_e8['spearman_rho']:+.3f} "
        f"with GRU advantage. Full-design distance has ρ={full_embedding_e8['spearman_rho']:+.3f} "
        f"with embedding displacement and ρ={full_performance_e8['spearman_rho']:+.3f} "
        "with GRU advantage.",
        "",
        "### All-valid task-design view (primary + stress-test)",
        "",
        "**E=4**",
        "",
        "![All valid cohorts: task-design distance versus transfer and embedding displacement](../fig_task_design_drivers_all_valid.png)",
        "",
        "**E=8**",
        "",
        "![All valid cohorts, E8: task-design distance versus transfer and embedding displacement](../fig_task_design_drivers_all_valid_e8.png)",
        "",
        "### Species-stratified all-valid description",
        "",
        "| space | species | cohorts | mean GRU614−Q bits/trial | median | range | mean embedding distance | mean task distance |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *_species_rows(task_data, 4),
        *_species_rows(task_data_e8, 8),
        "",
        "These are equal-cohort descriptive summaries, not species effects. Each species is "
        "represented by only two to six studies, and task design differs systematically by "
        "species. In particular, a species contrast would currently relabel the same design and "
        "apparatus contrasts rather than isolate biology.",
        "",
        f"For the {schedule_n} primary cohorts whose canonical adapters expose complete trial-wise probabilities "
        f"for both arms, empirical schedule distance from Grossman (mouse) is negatively associated with "
        f"GRU advantage (ρ={schedule_performance['spearman_rho']:+.3f}, exact permutation "
        f"p={schedule_performance['permutation_p_two_sided']:.4f}; bootstrap 95% CI "
        f"{_fmt_interval(schedule_performance['bootstrap_95_ci'])}; leave-one-out range "
        f"{_fmt_interval(schedule_performance['leave_one_out_range'])}). Its association with "
        f"embedding-centroid distance is weak (ρ={schedule_embedding['spearman_rho']:+.3f}, "
        f"p={schedule_embedding['permutation_p_two_sided']:.4f}). The performance result is "
        "promising but small-sample: the bootstrap interval crosses zero, and Grossman (mouse) defines "
        "the schedule-distance origin.",
        f"For E=8, schedule distance versus GRU advantage is ρ={schedule_performance_e8['spearman_rho']:+.3f} "
        f"(p={schedule_performance_e8['permutation_p_two_sided']:.4f}), while schedule distance "
        f"versus embedding-centroid distance is ρ={schedule_embedding_e8['spearman_rho']:+.3f} "
        f"(p={schedule_embedding_e8['permutation_p_two_sided']:.4f}).",
        "",
        "### Evidence-backed categorical matrix",
        "",
        "| cohort | tier | schedule | coupling | baited | choice target | context | response | reward | task distance | full distance |",
        "|---|---|---|---|---|---|---|---|---|---:|---:|",
        *_task_rows(task_data),
        "",
        "Cohort names link to the primary Methods source used for annotation. The complete "
        "evidence note for every row, the three AIND prototypes, and the scoring contract are "
        "frozen in `analysis/task_design_features.json`.",
        "",
        "### Empirical reward-schedule features",
        "",
        "| cohort | arm lag-1 | gap lag-1 | cross-arm | change rate | step size | mean abs(gap) | mean p(reward) | equal-p fraction | distance to Grossman (mouse) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_schedule_rows(task_data),
        "",
        "Every metric is computed using only within-session transitions. Subjects are summarized "
        "first and then averaged equally within a cohort. The composite distance is the root-mean-"
        "square standardized Euclidean distance across seven predeclared features; reward-gap "
        "autocorrelation is reported but omitted from the distance because it largely duplicates "
        "the two arm-autocorrelation terms. No missing schedule is imputed.",
        "",
        "#### E=4",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_task_relationship_rows(task_data["relationships"]),
        "",
        "#### E=8",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_task_relationship_rows(task_data_e8["relationships"]),
        "",
        "### All-valid categorical sensitivity — E=4",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_task_relationship_rows(task_data["sensitivity_relationships"]),
        "",
        "### All-valid categorical sensitivity — E=8",
        "",
        "| relationship | n | Spearman ρ | cohort-bootstrap 95% CI | permutation p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|---:|",
        *_task_relationship_rows(task_data_e8["sensitivity_relationships"]),
        "",
        "### Individual schedule-feature screen",
        "",
        "| space | schedule feature vs GRU614−Q | Spearman ρ | exact p | BH-FDR q |",
        "|---|---|---:|---:|---:|",
        *_feature_screen_rows(task_data, 4),
        *_feature_screen_rows(task_data_e8, 8),
        "",
        "No individual schedule feature survives the eight-feature FDR correction. The composite "
        "distance result should therefore motivate preregistered tests on additional datasets, "
        "not a post-hoc claim that one schedule statistic is the mechanism.",
        "",
        "## What this version can and cannot answer",
        "",
        "Embedding distance is measured after embedding-only adaptation, so it can reflect both "
        "task structure and cohort behavior. Species, study, apparatus, reward schedule, and data "
        f"volume remain confounded, and even the {valid_n}-cohort sensitivity set is too small for a causal species effect or a "
        "stable multivariable regression. Species colors are descriptive only.",
        "",
        "### Quarantined result",
        "",
        "Kwak (mouse) is excluded from every plot, correlation, ranking, direction count, and "
        "species summary. The frozen split adapts on CNO sessions and tests on DMSO sessions, "
        "so it mixes treatment transfer with subject adaptation. The required replacement keeps "
        "only DMSO/control sessions, adapts on chronological odd DMSO sessions, and tests on "
        "chronological even DMSO sessions. No rerun was launched in this cleanup.",
        "",
        "The next defensible extension is to recover explicit schedules for the remaining "
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
    data_by_dimension = {
        dimension: json.loads(path.read_text()) for dimension, path in DATA.items()
    }
    task_data_by_dimension = {
        dimension: json.loads(path.read_text())
        for dimension, path in TASK_DATA.items()
    }
    llm_data = json.loads(LLM_DATA.read_text())
    expected = tuple(data_by_dimension[4]["contract"]["cohort_order"])
    for dimension, data in data_by_dimension.items():
        task_data = task_data_by_dimension[dimension]
        if tuple(data["cohorts"]) != expected:
            raise AssertionError("Generalization-driver cohort order drifted")
        if tuple(task_data["cohorts"]) != expected:
            raise AssertionError("Task-design cohort order drifted")
        if int(data["contract"]["subject_embedding_size"]) != dimension:
            raise AssertionError("Generalization embedding-dimension contract drifted")
        if int(task_data["contract"]["subject_embedding_size"]) != dimension:
            raise AssertionError("Task-design embedding-dimension contract drifted")
        if any(len(cohort["seeds"]) != 3 for cohort in data["cohorts"].values()):
            raise AssertionError("Each cohort must contain exactly three source seeds")
        if any(
            data["cohorts"][name]["label"] != task_data["cohorts"][name]["label"]
            for name in expected
        ):
            raise AssertionError("Generalization and task-design cohort labels drifted")
        if set(cohort["species"] for cohort in data["cohorts"].values()) != set(
            SPECIES_COLORS
        ):
            raise AssertionError("Species color map does not match frozen cohorts")
        main_figures, robustness_figures, task_figures = FIGURE_SETS[dimension]
        for view, output in main_figures.items():
            _plot_main(data, view, output)
        for view, output in R1_SCALE_MAIN_FIGURES[dimension].items():
            _plot_main(data, view, output, r1_scale=True)
        for view, output in AUTHOR_MAIN_FIGURES[dimension].items():
            _plot_main(data, view, output, reference="author")
        for view, output in AUTHOR_R1_SCALE_MAIN_FIGURES[dimension].items():
            _plot_main(
                data,
                view,
                output,
                r1_scale=True,
                reference="author",
            )
        for view, output in robustness_figures.items():
            _plot_robustness(data, view, output)
        for view, output in task_figures.items():
            _plot_task_design(
                task_data_by_dimension[dimension],
                data,
                view,
                output,
            )
    _plot_slide_synthesis(data_by_dimension[8], task_data_by_dimension[8])
    _plot_focused_author_embedding_llm(data_by_dimension[8], llm_data)
    body = _result_block(data_by_dimension, task_data_by_dimension)
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + body + "\n" + text[end_start:])
    outputs = [
        *(
            output
            for figure_sets in FIGURE_SETS.values()
            for figures in figure_sets
            for output in figures.values()
        ),
        *(
            output
            for figures in R1_SCALE_MAIN_FIGURES.values()
            for output in figures.values()
        ),
        *(
            output
            for figure_map in (AUTHOR_MAIN_FIGURES, AUTHOR_R1_SCALE_MAIN_FIGURES)
            for figures in figure_map.values()
            for output in figures.values()
        ),
        SLIDE_FIGURE_PNG,
        SLIDE_FIGURE_SVG,
        FOCUSED_FIGURE_PNG,
        FOCUSED_FIGURE_SVG,
        REPORT,
    ]
    print("Wrote " + ", ".join(str(output) for output in outputs))


if __name__ == "__main__":
    main()
