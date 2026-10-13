"""Render the frozen Study 09 LLM task-distance analysis."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import rankdata, spearmanr


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402
from plot_style import SPECIES_COLORS, apply_presentation_style  # noqa: E402


JUDGMENTS = STUDY / "analysis" / "llm_task_similarity_judgments.json"
CARDS = STUDY / "analysis" / "llm_task_cards.json"
GENERALIZATION = STUDY / "analysis" / "generalization_drivers_e8.json"
TASK_FEATURES = STUDY / "analysis" / "task_design_features_e8.json"
RESULTS = STUDY / "analysis" / "llm_task_similarity_results.json"
FIGURE_PNG = STUDY / "analysis" / "fig_llm_task_similarity.png"
FIGURE_SVG = STUDY / "analysis" / "fig_llm_task_similarity.svg"
CROSSCHECK_PNG = STUDY / "analysis" / "fig_task_distance_crosscheck.png"
CROSSCHECK_SVG = STUDY / "analysis" / "fig_task_distance_crosscheck.svg"
FULL_DESIGN_PNG = STUDY / "analysis" / "fig_full_design_vs_llm_task_distance.png"
FULL_DESIGN_SVG = STUDY / "analysis" / "fig_full_design_vs_llm_task_distance.svg"
REPORT = STUDY / "analysis" / "reports" / "r6-llm-task-similarity.md"
START = "<!-- BEGIN result-6 -->"
END = "<!-- END result-6 -->"

RNG_SEED = 20260915
N_PERMUTATIONS = 100_000
TIER_MARKERS = {"primary": "o", "stress_test": "s", "descriptive_only": "^"}


def _semantic_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text())
    payload.pop("_meta", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sem(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _pair_outcome(row: dict) -> tuple[str, str, float]:
    first, second = row["forward"]["order"]
    choice = row["forward"]["choice"]
    return first, second, {"A": 1.0, "B": 0.0, "tie": 0.5}[choice]


def _pairwise_win_rates(
    card_ids: list[str], comparisons: list[tuple[str, str, float]]
) -> np.ndarray:
    index = {card_id: idx for idx, card_id in enumerate(card_ids)}
    wins = np.zeros(len(card_ids), dtype=float)
    appearances = np.zeros(len(card_ids), dtype=float)
    for first, second, outcome in comparisons:
        first_index = index[first]
        second_index = index[second]
        wins[first_index] += outcome
        wins[second_index] += 1 - outcome
        appearances[first_index] += 1
        appearances[second_index] += 1
    if np.any(appearances == 0):
        raise AssertionError("every card must appear in the sampled tournament")
    return wins / appearances


def _spearman_permutation(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> dict:
    rho = float(spearmanr(x, y).statistic)
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if len(x) <= 8:
        permutations = np.asarray(list(itertools.permutations(y_centered)))
        null = permutations @ x_centered / denominator
        p_value = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
        n_permutations = len(permutations)
        permutation_method = "exact"
    else:
        permutations = np.asarray(
            [rng.permutation(y_centered) for _ in range(N_PERMUTATIONS)]
        )
        null = permutations @ x_centered / denominator
        p_value = float(
            (np.count_nonzero(np.abs(null) >= abs(rho) - 1e-12) + 1)
            / (N_PERMUTATIONS + 1)
        )
        n_permutations = N_PERMUTATIONS
        permutation_method = "monte_carlo"
    leave_one_out = [
        float(spearmanr(np.delete(x, index), np.delete(y, index)).statistic)
        for index in range(len(x))
    ]
    return {
        "n_cohorts": int(len(x)),
        "spearman_rho": rho,
        "permutation_p_two_sided": p_value,
        "permutation_method": permutation_method,
        "n_permutations": n_permutations,
        "leave_one_cohort_out_range": [min(leave_one_out), max(leave_one_out)],
    }


def _relation_title(label: str, relation: dict) -> str:
    return (
        f"{label}\nn={relation['n_cohorts']}; Spearman ρ={relation['spearman_rho']:+.2f}, "
        f"permutation p={relation['permutation_p_two_sided']:.3f}"
    )


def _plot_point(
    axis: plt.Axes,
    row: dict,
    x: float,
    y_values: np.ndarray,
) -> None:
    color = SPECIES_COLORS[row["species"]]
    marker = TIER_MARKERS[row["analysis_tier"]]
    y = float(np.mean(y_values))
    axis.errorbar(
        x,
        y,
        yerr=_sem(y_values),
        fmt=marker,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=9,
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )
    axis.annotate(
        row["label"].split(" (")[0],
        (x, y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
    )


def _annotation_values(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, list) else (value,)


def _mixed_feature_geometry(task_features: dict, row_by_cohort: dict[str, dict]) -> dict:
    """Build the seven-cohort outcome-blind mixed-feature PCA geometry."""
    contract = task_features["contract"]
    cohort_names = list(contract["schedule_available_cohorts"])
    if len(cohort_names) != 7:
        raise AssertionError("The complete-case task geometry must contain seven cohorts")
    if any(task_features["cohorts"][name]["schedule_features"] is None for name in cohort_names):
        raise AssertionError("Every complete-case cohort must have empirical schedule fields")

    categorical_axes = (
        list(contract["categorical_distance"]["task_structure_axes"])
        + list(contract["categorical_distance"]["apparatus_axes"])
    )
    empirical_features = list(contract["schedule_distance_features"])
    prototypes = contract["aind_source_prototypes"]
    annotations = {
        name: task_features["cohorts"][name]["annotation"] for name in cohort_names
    }
    annotations.update({f"aind::{name}": value for name, value in prototypes.items()})

    levels: dict[str, list[object]] = {}
    for axis in categorical_axes:
        values = {
            value
            for annotation in annotations.values()
            for value in _annotation_values(annotation[axis])
        }
        levels[axis] = sorted(values, key=lambda value: str(value))

    feature_names = []
    for axis in categorical_axes:
        feature_names.extend(f"categorical::{axis}::{value}" for value in levels[axis])
    feature_names.extend(f"empirical::{name}" for name in empirical_features)

    grossman = task_features["cohorts"]["grossman"]
    grossman_empirical = {
        name: grossman["schedule_features"][name]["subject_balanced_mean"]
        for name in empirical_features
    }
    standardization = task_features["schedule_standardization"]

    def encode(annotation: dict, empirical: dict[str, float]) -> np.ndarray:
        categorical = []
        for axis in categorical_axes:
            selected = set(_annotation_values(annotation[axis]))
            categorical.extend(float(level in selected) for level in levels[axis])
        categorical_values = np.asarray(categorical, dtype=float)
        categorical_values /= np.sqrt(2 * len(categorical_axes))
        empirical_values = np.asarray(
            [
                (empirical[name] - standardization[name]["mean"])
                / standardization[name]["population_sd"]
                for name in empirical_features
            ],
            dtype=float,
        )
        empirical_values /= np.sqrt(len(empirical_features))
        return np.concatenate([categorical_values, empirical_values]) / np.sqrt(2)

    row_ids = list(cohort_names) + [f"aind::{name}" for name in prototypes]
    matrix_rows = []
    raw_rows = {}
    for row_id in row_ids:
        if row_id.startswith("aind::"):
            annotation = annotations[row_id]
            empirical = grossman_empirical
            label = row_id.removeprefix("aind::")
            row_type = "aind_proxy"
        else:
            cohort = task_features["cohorts"][row_id]
            annotation = cohort["annotation"]
            empirical = {
                name: cohort["schedule_features"][name]["subject_balanced_mean"]
                for name in empirical_features
            }
            label = row_by_cohort[row_id]["label"]
            row_type = "external_cohort"
        matrix_rows.append(encode(annotation, empirical))
        raw_rows[row_id] = {
            "label": label,
            "row_type": row_type,
            "annotation": {axis: annotation[axis] for axis in categorical_axes},
            "empirical_schedule_features": empirical,
        }

    matrix = np.vstack(matrix_rows)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular_values**2
    explained = variance / variance.sum()
    cumulative = np.cumsum(explained)
    retained = max(2, int(np.searchsorted(cumulative, 0.90) + 1))
    scores = centered @ right_vectors.T
    prototype_indices = np.arange(len(cohort_names), len(row_ids))

    rows = {}
    for index, row_id in enumerate(row_ids):
        row = {
            **raw_rows[row_id],
            "pc_scores": [float(value) for value in scores[index]],
        }
        if row_id in cohort_names:
            retained_distances = np.linalg.norm(
                scores[index, :retained] - scores[prototype_indices, :retained], axis=1
            )
            full_distances = np.linalg.norm(
                scores[index] - scores[prototype_indices], axis=1
            )
            nearest_index = int(np.argmin(retained_distances))
            nearest_name = list(prototypes)[nearest_index]
            row.update(
                {
                    "feature_task_distance_to_aind_proxy": float(
                        retained_distances[nearest_index]
                    ),
                    "full_pc_space_distance_to_aind_proxy": float(
                        full_distances[int(np.argmin(full_distances))]
                    ),
                    "nearest_aind_proxy": nearest_name,
                    "empirical_schedule_distance_to_grossman": float(
                        task_features["cohorts"][row_id][
                            "empirical_schedule_distance_to_grossman"
                        ]
                    ),
                }
            )
        rows[row_id] = row

    loading_rows = []
    for component in range(min(3, len(explained))):
        order = np.argsort(np.abs(right_vectors[component]))[::-1][:6]
        loading_rows.append(
            {
                "component": component + 1,
                "explained_variance_ratio": float(explained[component]),
                "top_loadings": [
                    {
                        "feature": feature_names[index],
                        "loading": float(right_vectors[component, index]),
                    }
                    for index in order
                ],
            }
        )

    return {
        "contract": {
            "cohorts": cohort_names,
            "categorical_axes": categorical_axes,
            "empirical_schedule_features": empirical_features,
            "encoding": (
                "Categorical axes are one-hot encoded so a one-axis mismatch has unit "
                "squared distance before block weighting. Empirical cohort means are "
                "z-scored with the frozen seven-cohort population mean and SD. The "
                "categorical and empirical blocks each receive one-half total weight."
            ),
            "aind_proxy": (
                "The three categorical AIND prototypes are exact. Their empirical schedule "
                "coordinates use Grossman (mouse), because source-curriculum empirical "
                "schedule summaries are not yet frozen. Interpret this as an AIND proxy."
            ),
            "pca": (
                "PCA is fit outcome-blind to seven complete-case cohorts plus three AIND "
                "proxy rows. Distance uses the smallest leading-PC subspace explaining at "
                "least 90% of feature variance; all-PC distance is retained as sensitivity."
            ),
            "retained_components": retained,
            "retained_explained_variance": float(cumulative[retained - 1]),
        },
        "feature_names": feature_names,
        "explained_variance_ratio": [float(value) for value in explained],
        "principal_component_loadings": loading_rows,
        "rows": rows,
    }


def _plot_xy_point(
    axis: plt.Axes,
    row: dict,
    x: float,
    y: float,
    *,
    y_values: np.ndarray | None = None,
    label_offset: tuple[float, float] = (4, 4),
) -> None:
    color = SPECIES_COLORS[row["species"]]
    marker = TIER_MARKERS[row["analysis_tier"]]
    yerr = _sem(y_values) if y_values is not None and len(y_values) > 1 else None
    axis.errorbar(
        x,
        y,
        yerr=yerr,
        fmt=marker,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        markersize=9,
        elinewidth=1.1,
        capsize=3,
        zorder=4,
    )
    axis.annotate(
        row["label"].split(" (")[0],
        (x, y),
        xytext=label_offset,
        textcoords="offset points",
        fontsize=8.5,
        color=color,
    )


def _replace_report(text: str) -> None:
    report = REPORT.read_text()
    if START not in report or END not in report:
        raise RuntimeError("Result 6 report markers are missing")
    before, tail = report.split(START, 1)
    _, after = tail.split(END, 1)
    REPORT.write_text(before + START + "\n" + text.rstrip() + "\n" + END + after)


def main() -> None:
    judgments = json.loads(JUDGMENTS.read_text())
    generalization = json.loads(GENERALIZATION.read_text())
    task_features = json.loads(TASK_FEATURES.read_text())
    mapping = judgments["card_to_cohort_unblinded_after_judgment"]
    card_ids = sorted(mapping)
    comparisons = [_pair_outcome(row) for row in judgments["comparisons"]]

    rng = np.random.default_rng(RNG_SEED)
    distances = 1.0 - _pairwise_win_rates(card_ids, comparisons)

    rows = []
    for index, card_id in enumerate(card_ids):
        name = mapping[card_id]
        cohort = generalization["cohorts"][name]
        seeds = cohort["seeds"]
        author_values = [
            float(seed["gru_d614_minus_author_subject_balanced_normalized_likelihood"])
            for seed in seeds
            if seed.get("gru_d614_minus_author_subject_balanced_normalized_likelihood")
            is not None
        ]
        rows.append(
            {
                "card_id": card_id,
                "cohort": name,
                "label": cohort["label"],
                "species": cohort["species"],
                "analysis_tier": cohort["analysis_tier"],
                "llm_task_distance_to_aind": float(distances[index]),
                "full_design_distance_to_aind": float(
                    task_features["cohorts"][name]["categorical_distance"][
                        "full_design"
                    ]
                ),
                "gru_minus_bari_normalized_likelihood_by_seed": [
                    float(seed["gru_d614_minus_q_subject_balanced_normalized_likelihood"])
                    for seed in seeds
                ],
                "gru_minus_author_normalized_likelihood_by_seed": author_values,
                "embedding_centroid_mahalanobis_by_seed": [
                    float(seed["embedding_centroid_mahalanobis"]) for seed in seeds
                ],
            }
        )
    rows.sort(key=lambda row: (row["llm_task_distance_to_aind"], row["label"]))

    x_all = np.asarray([row["llm_task_distance_to_aind"] for row in rows])
    q_all = np.asarray(
        [np.mean(row["gru_minus_bari_normalized_likelihood_by_seed"]) for row in rows]
    )
    embedding_all = np.asarray(
        [np.mean(row["embedding_centroid_mahalanobis_by_seed"]) for row in rows]
    )
    author_rows = [row for row in rows if row["gru_minus_author_normalized_likelihood_by_seed"]]
    author_x = np.asarray([row["llm_task_distance_to_aind"] for row in author_rows])
    author_y = np.asarray(
        [np.mean(row["gru_minus_author_normalized_likelihood_by_seed"]) for row in author_rows]
    )
    relationships = {
        "gru_minus_bari_vs_llm_task_distance": _spearman_permutation(x_all, q_all, rng),
        "gru_minus_author_vs_llm_task_distance": _spearman_permutation(author_x, author_y, rng),
        "embedding_distance_vs_llm_task_distance": _spearman_permutation(
            x_all, embedding_all, rng
        ),
        "full_design_distance_vs_llm_task_distance": _spearman_permutation(
            np.asarray([row["full_design_distance_to_aind"] for row in rows]),
            x_all,
            rng,
        ),
    }

    row_by_cohort = {row["cohort"]: row for row in rows}
    feature_geometry = _mixed_feature_geometry(task_features, row_by_cohort)
    complete_names = feature_geometry["contract"]["cohorts"]
    complete_rows = [row_by_cohort[name] for name in complete_names]
    geometry_rows = feature_geometry["rows"]
    feature_distance = np.asarray(
        [geometry_rows[name]["feature_task_distance_to_aind_proxy"] for name in complete_names]
    )
    full_feature_distance = np.asarray(
        [geometry_rows[name]["full_pc_space_distance_to_aind_proxy"] for name in complete_names]
    )
    empirical_distance = np.asarray(
        [geometry_rows[name]["empirical_schedule_distance_to_grossman"] for name in complete_names]
    )
    llm_distance = np.asarray(
        [row_by_cohort[name]["llm_task_distance_to_aind"] for name in complete_names]
    )
    complete_q = np.asarray(
        [
            np.mean(row_by_cohort[name]["gru_minus_bari_normalized_likelihood_by_seed"])
            for name in complete_names
        ]
    )
    complete_author = np.asarray(
        [
            np.mean(row_by_cohort[name]["gru_minus_author_normalized_likelihood_by_seed"])
            for name in complete_names
        ]
    )
    complete_embedding = np.asarray(
        [
            np.mean(row_by_cohort[name]["embedding_centroid_mahalanobis_by_seed"])
            for name in complete_names
        ]
    )
    complete_case_relationships = {
        "feature_distance_vs_llm_task_distance": _spearman_permutation(
            feature_distance, llm_distance, rng
        ),
        "empirical_grossman_distance_vs_llm_task_distance": _spearman_permutation(
            empirical_distance, llm_distance, rng
        ),
        "feature_distance_vs_empirical_grossman_distance": _spearman_permutation(
            feature_distance, empirical_distance, rng
        ),
        "retained_vs_all_pc_feature_distance": _spearman_permutation(
            feature_distance, full_feature_distance, rng
        ),
        "gru_minus_bari_vs_feature_distance": _spearman_permutation(
            feature_distance, complete_q, rng
        ),
        "gru_minus_bari_vs_llm_task_distance": _spearman_permutation(
            llm_distance, complete_q, rng
        ),
        "gru_minus_author_vs_feature_distance": _spearman_permutation(
            feature_distance, complete_author, rng
        ),
        "gru_minus_author_vs_llm_task_distance": _spearman_permutation(
            llm_distance, complete_author, rng
        ),
        "embedding_distance_vs_feature_distance": _spearman_permutation(
            feature_distance, complete_embedding, rng
        ),
        "embedding_distance_vs_llm_task_distance": _spearman_permutation(
            llm_distance, complete_embedding, rng
        ),
    }

    for row in complete_rows:
        geometry = geometry_rows[row["cohort"]]
        row["complete_case_feature_geometry"] = {
            key: geometry[key]
            for key in (
                "feature_task_distance_to_aind_proxy",
                "full_pc_space_distance_to_aind_proxy",
                "nearest_aind_proxy",
                "empirical_schedule_distance_to_grossman",
                "pc_scores",
            )
        }

    output = {
        "_meta": build_meta(
            "analysis/report_llm_task_similarity.py",
            generalization["_meta"]["wandb_groups"],
            study_root=STUDY,
        ),
        "inputs": {
            "analysis/llm_task_cards.json": _semantic_json_sha256(CARDS),
            "analysis/llm_task_similarity_judgments.json": _semantic_json_sha256(
                JUDGMENTS
            ),
            "analysis/generalization_drivers_e8.json": _semantic_json_sha256(
                GENERALIZATION
            ),
            "analysis/task_design_features_e8.json": _semantic_json_sha256(
                TASK_FEATURES
            ),
        },
        "contract": {
            "llm_task_distance_to_aind": (
                "One minus the fraction of pairwise comparisons judged closer to AIND; "
                "ties contribute one-half win. Each candidate is compared with all 11 "
                "alternatives, so zero means closest and one means farthest."
            ),
            "full_design_distance_to_aind": (
                "The previous seven-axis categorical Hamming distance to the nearest "
                "of the three AIND task prototypes."
            ),
            "uncertainty": (
                "The LLM task distance has no error bar because this first pass has one judge. "
                "Performance and embedding error bars are SEM across three source seeds."
            ),
            "performance_metric": (
                "E8 D=614 three-source-seed mean subject-balanced normalized-likelihood difference."
            ),
            "correlation": (
                "Cohort-level Spearman correlation with exact two-sided permutation for "
                "n<=8 and deterministic 100,000-draw permutation otherwise."
            ),
            "rng_seed": RNG_SEED,
        },
        "judge": judgments["judge"],
        "audit": judgments["audit"],
        "ranking": rows,
        "relationships": relationships,
        "complete_case_feature_geometry": feature_geometry,
        "complete_case_relationships": complete_case_relationships,
    }
    RESULTS.write_text(json.dumps(output, indent=2) + "\n")

    apply_presentation_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 14), constrained_layout=True)
    rank_axis, q_axis, author_axis, embedding_axis = axes.flat
    ordered = list(reversed(rows))
    y_positions = np.arange(len(ordered))
    for y_position, row in zip(y_positions, ordered, strict=True):
        distance = row["llm_task_distance_to_aind"]
        rank_axis.barh(
            y_position,
            distance,
            color=SPECIES_COLORS[row["species"]],
            alpha=0.88,
            edgecolor="white",
            hatch="//" if row["analysis_tier"] == "stress_test" else None,
            capsize=3,
        )
    rank_axis.set_yticks(y_positions, [row["label"] for row in ordered])
    for tick, row in zip(rank_axis.get_yticklabels(), ordered, strict=True):
        tick.set_color(SPECIES_COLORS[row["species"]])
        tick.set_fontsize(11)
    rank_axis.set_xlim(0, 1)
    rank_axis.set_xlabel("LLM task distance to AIND")
    rank_axis.set_title(
        "Prompt-blinded task-design rank\n66 implied pairs from one frozen ordinal judge"
    )
    rank_axis.set_box_aspect(1)

    for row in rows:
        distance = row["llm_task_distance_to_aind"]
        _plot_point(
            q_axis,
            row,
            distance,
            np.asarray(row["gru_minus_bari_normalized_likelihood_by_seed"]),
        )
        if row["gru_minus_author_normalized_likelihood_by_seed"]:
            _plot_point(
                author_axis,
                row,
                distance,
                np.asarray(row["gru_minus_author_normalized_likelihood_by_seed"]),
            )
        _plot_point(
            embedding_axis,
            row,
            distance,
            np.asarray(row["embedding_centroid_mahalanobis_by_seed"]),
        )

    panels = (
        (
            q_axis,
            "GRU E8 D=614 − Bari2019\n(normalized likelihood)",
            relationships["gru_minus_bari_vs_llm_task_distance"],
        ),
        (
            author_axis,
            "GRU E8 D=614 − author model\n(normalized likelihood)",
            relationships["gru_minus_author_vs_llm_task_distance"],
        ),
        (
            embedding_axis,
            "E8 external-centroid distance\n(Mahalanobis)",
            relationships["embedding_distance_vs_llm_task_distance"],
        ),
    )
    for axis, ylabel, relationship in panels:
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1, zorder=1)
        axis.set_xlim(0, 1)
        axis.set_xlabel("LLM task distance to AIND")
        axis.set_ylabel(ylabel)
        axis.set_title(_relation_title(ylabel.split("\n")[0], relationship))
        axis.set_box_aspect(1)

    species_handles = [
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
    tier_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#777777",
            markersize=9,
            label=tier.replace("_", " ").title(),
        )
        for tier, marker in TIER_MARKERS.items()
        if any(row["analysis_tier"] == tier for row in rows)
    ]
    q_axis.legend(handles=species_handles + tier_handles, loc="best", fontsize=10)
    fig.suptitle(
        "Prompt-blinded LLM task distance versus external transfer",
        fontsize=22,
        fontweight="bold",
    )
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(FIGURE_SVG, bbox_inches="tight")
    FIGURE_SVG.write_text(
        "\n".join(line.rstrip() for line in FIGURE_SVG.read_text().splitlines())
        + "\n"
    )
    fig.savefig(FIGURE_PNG, bbox_inches="tight")
    plt.close(fig)

    full_design_relation = relationships[
        "full_design_distance_vs_llm_task_distance"
    ]
    full_design_fig, full_design_axis = plt.subplots(
        1, 1, figsize=(8.5, 8.5), constrained_layout=True
    )
    duplicate_offsets = {
        "grossman": -0.008,
        "hattori": 0.008,
        "chen": -0.008,
        "miller": 0.008,
    }
    label_offsets = {
        "grossman": (-4, -18),
        "hattori": (4, 6),
        "chen": (-34, -18),
        "miller": (4, 6),
    }
    for row in rows:
        _plot_xy_point(
            full_design_axis,
            row,
            row["full_design_distance_to_aind"]
            + duplicate_offsets.get(row["cohort"], 0.0),
            row["llm_task_distance_to_aind"],
            label_offset=label_offsets.get(row["cohort"], (4, 4)),
        )
    full_design_axis.set_xlabel("Previous full-design distance to AIND family")
    full_design_axis.set_ylabel("LLM task distance to AIND")
    full_design_axis.set_xlim(-0.04, 0.76)
    full_design_axis.set_ylim(-0.04, 1.06)
    full_design_axis.set_title(
        "Categorical full-design versus LLM task distance\n"
        f"n={full_design_relation['n_cohorts']}; Spearman "
        f"ρ={full_design_relation['spearman_rho']:+.2f}, permutation p<0.001"
    )
    full_design_axis.set_box_aspect(1)
    full_design_axis.text(
        0.02,
        0.98,
        "Tiny horizontal offsets separate exact overlaps; statistics use exact values.",
        transform=full_design_axis.transAxes,
        va="top",
        fontsize=8.5,
        color="#666666",
    )
    full_design_axis.legend(
        handles=species_handles + tier_handles,
        loc="lower right",
        fontsize=9,
    )
    with plt.rc_context({"svg.fonttype": "none"}):
        full_design_fig.savefig(FULL_DESIGN_SVG, bbox_inches="tight")
    FULL_DESIGN_SVG.write_text(
        "\n".join(line.rstrip() for line in FULL_DESIGN_SVG.read_text().splitlines())
        + "\n"
    )
    full_design_fig.savefig(FULL_DESIGN_PNG, bbox_inches="tight")
    plt.close(full_design_fig)

    cross_fig, cross_axes = plt.subplots(
        5, 2, figsize=(15, 34), constrained_layout=True
    )
    (
        pca_axis,
        agreement_axis,
        empirical_llm_axis,
        empirical_feature_axis,
        q_feature_axis,
        q_llm_axis,
        author_feature_axis,
        author_llm_axis,
        embedding_feature_axis,
        embedding_llm_axis,
    ) = cross_axes.flat

    for row in complete_rows:
        pc_scores = geometry_rows[row["cohort"]]["pc_scores"]
        _plot_xy_point(pca_axis, row, pc_scores[0], pc_scores[1])
    prototype_labels = {
        "uncoupled_baiting": "AIND uncoupled baiting",
        "uncoupled_no_baiting": "AIND uncoupled no-baiting",
        "coupled_baiting": "AIND coupled baiting",
    }
    prototype_offsets = {
        "uncoupled_baiting": (8, 16),
        "uncoupled_no_baiting": (8, -12),
        "coupled_baiting": (-105, 4),
    }
    for prototype, label in prototype_labels.items():
        pc_scores = geometry_rows[f"aind::{prototype}"]["pc_scores"]
        pca_axis.scatter(
            pc_scores[0],
            pc_scores[1],
            marker="*",
            s=180,
            facecolors="none",
            edgecolor="#222222",
            linewidth=1.4,
            zorder=5,
        )
        pca_axis.annotate(
            label,
            (pc_scores[0], pc_scores[1]),
            xytext=prototype_offsets[prototype],
            textcoords="offset points",
            fontsize=8.5,
            color="#222222",
        )
    explained = feature_geometry["explained_variance_ratio"]
    pca_axis.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
    pca_axis.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)")
    pca_axis.set_title(
        "A  Outcome-blind mixed task-feature PCA\n"
        f"{feature_geometry['contract']['retained_components']} PCs retain "
        f"{100 * feature_geometry['contract']['retained_explained_variance']:.1f}%"
    )
    pca_axis.set_box_aspect(1)

    relation = complete_case_relationships["feature_distance_vs_llm_task_distance"]
    for row in complete_rows:
        name = row["cohort"]
        _plot_xy_point(
            agreement_axis,
            row,
            geometry_rows[name]["feature_task_distance_to_aind_proxy"],
            row["llm_task_distance_to_aind"],
        )
    agreement_axis.set_xlabel("Feature-based task distance to AIND proxy")
    agreement_axis.set_ylabel("LLM task distance to AIND")
    agreement_axis.set_title(_relation_title("B  Feature versus LLM distance", relation))
    agreement_axis.set_box_aspect(1)

    relation = complete_case_relationships[
        "empirical_grossman_distance_vs_llm_task_distance"
    ]
    for index, row in enumerate(complete_rows):
        _plot_xy_point(
            empirical_llm_axis,
            row,
            float(empirical_distance[index]),
            float(llm_distance[index]),
        )
    empirical_llm_axis.set_xlabel("Empirical schedule distance to Grossman")
    empirical_llm_axis.set_ylabel("LLM task distance to AIND")
    empirical_llm_axis.set_title(
        _relation_title("C  Empirical versus LLM distance", relation)
    )
    empirical_llm_axis.set_box_aspect(1)

    relation = complete_case_relationships[
        "feature_distance_vs_empirical_grossman_distance"
    ]
    for index, row in enumerate(complete_rows):
        _plot_xy_point(
            empirical_feature_axis,
            row,
            float(empirical_distance[index]),
            float(feature_distance[index]),
        )
    empirical_feature_axis.set_xlabel("Empirical schedule distance to Grossman")
    empirical_feature_axis.set_ylabel("Feature-based task distance to AIND proxy")
    empirical_feature_axis.set_title(
        _relation_title("D  Empirical versus mixed-feature distance", relation)
    )
    empirical_feature_axis.set_box_aspect(1)

    outcome_panels = (
        (
            q_feature_axis,
            feature_distance,
            complete_q,
            "Feature-based task distance to AIND proxy",
            "GRU E8 D=614 − Bari2019\n(normalized likelihood)",
            "E  GRU−Bari2019 versus feature distance",
            complete_case_relationships["gru_minus_bari_vs_feature_distance"],
            "gru_minus_bari_normalized_likelihood_by_seed",
        ),
        (
            q_llm_axis,
            llm_distance,
            complete_q,
            "LLM task distance to AIND",
            "GRU E8 D=614 − Bari2019\n(normalized likelihood)",
            "F  GRU−Bari2019 versus LLM distance",
            complete_case_relationships["gru_minus_bari_vs_llm_task_distance"],
            "gru_minus_bari_normalized_likelihood_by_seed",
        ),
        (
            author_feature_axis,
            feature_distance,
            complete_author,
            "Feature-based task distance to AIND proxy",
            "GRU E8 D=614 − author model\n(normalized likelihood)",
            "G  GRU−author versus feature distance",
            complete_case_relationships["gru_minus_author_vs_feature_distance"],
            "gru_minus_author_normalized_likelihood_by_seed",
        ),
        (
            author_llm_axis,
            llm_distance,
            complete_author,
            "LLM task distance to AIND",
            "GRU E8 D=614 − author model\n(normalized likelihood)",
            "H  GRU−author versus LLM distance",
            complete_case_relationships["gru_minus_author_vs_llm_task_distance"],
            "gru_minus_author_normalized_likelihood_by_seed",
        ),
        (
            embedding_feature_axis,
            feature_distance,
            complete_embedding,
            "Feature-based task distance to AIND proxy",
            "E8 external-centroid distance\n(Mahalanobis)",
            "I  Embedding versus feature distance",
            complete_case_relationships["embedding_distance_vs_feature_distance"],
            "embedding_centroid_mahalanobis_by_seed",
        ),
        (
            embedding_llm_axis,
            llm_distance,
            complete_embedding,
            "LLM task distance to AIND",
            "E8 external-centroid distance\n(Mahalanobis)",
            "J  Embedding versus LLM distance",
            complete_case_relationships["embedding_distance_vs_llm_task_distance"],
            "embedding_centroid_mahalanobis_by_seed",
        ),
    )
    for axis, x_values, y_values, xlabel, ylabel, title, relation, seed_key in outcome_panels:
        for index, row in enumerate(complete_rows):
            _plot_xy_point(
                axis,
                row,
                float(x_values[index]),
                float(y_values[index]),
                y_values=np.asarray(row[seed_key]),
            )
        if "normalized likelihood" in ylabel:
            axis.axhline(0, color="#777777", linestyle="--", linewidth=1, zorder=1)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(_relation_title(title, relation))
        axis.set_box_aspect(1)

    cross_species_handles = [
        handle
        for handle in species_handles
        if any(handle.get_label().lower() == row["species"] for row in complete_rows)
    ]
    cross_species_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#222222",
            markersize=13,
            label="AIND proxy",
        )
    )
    pca_axis.legend(handles=cross_species_handles, loc="best", fontsize=9)
    cross_fig.suptitle(
        "Complete-case feature-based and LLM task-distance cross-check",
        fontsize=22,
        fontweight="bold",
    )
    with plt.rc_context({"svg.fonttype": "none"}):
        cross_fig.savefig(CROSSCHECK_SVG, bbox_inches="tight")
    CROSSCHECK_SVG.write_text(
        "\n".join(line.rstrip() for line in CROSSCHECK_SVG.read_text().splitlines())
        + "\n"
    )
    cross_fig.savefig(CROSSCHECK_PNG, bbox_inches="tight")
    plt.close(cross_fig)

    top = ", ".join(row["label"] for row in rows[:3])
    bottom = ", ".join(row["label"] for row in rows[-3:])
    q_relation = relationships["gru_minus_bari_vs_llm_task_distance"]
    author_relation = relationships["gru_minus_author_vs_llm_task_distance"]
    embedding_relation = relationships["embedding_distance_vs_llm_task_distance"]
    full_design_llm_relation = relationships[
        "full_design_distance_vs_llm_task_distance"
    ]
    feature_llm_relation = complete_case_relationships[
        "feature_distance_vs_llm_task_distance"
    ]
    empirical_llm_relation = complete_case_relationships[
        "empirical_grossman_distance_vs_llm_task_distance"
    ]
    feature_empirical_relation = complete_case_relationships[
        "feature_distance_vs_empirical_grossman_distance"
    ]
    feature_embedding_relation = complete_case_relationships[
        "embedding_distance_vs_feature_distance"
    ]
    llm_embedding_complete_relation = complete_case_relationships[
        "embedding_distance_vs_llm_task_distance"
    ]
    pca_sensitivity_relation = complete_case_relationships[
        "retained_vs_all_pc_feature_distance"
    ]
    table = [
        "| rank | study | tier | LLM task distance to AIND |",
        "|---:|---|---|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        table.append(
            f"| {rank} | {row['label']} | {row['analysis_tier'].replace('_', ' ')} | "
            f"{row['llm_task_distance_to_aind']:.3f} |"
        )
    complete_table = [
        "| study | feature distance to AIND proxy | LLM task distance to AIND | empirical distance to Grossman | nearest AIND proxy |",
        "|---|---:|---:|---:|---|",
    ]
    for row in complete_rows:
        geometry = geometry_rows[row["cohort"]]
        complete_table.append(
            f"| {row['label']} | {geometry['feature_task_distance_to_aind_proxy']:.3f} | "
            f"{row['llm_task_distance_to_aind']:.3f} | "
            f"{geometry['empirical_schedule_distance_to_grossman']:.3f} | "
            f"{geometry['nearest_aind_proxy'].replace('_', ' ')} |"
        )

    relationship_labels = (
        ("Feature distance vs LLM distance", "feature_distance_vs_llm_task_distance"),
        (
            "Empirical Grossman distance vs LLM distance",
            "empirical_grossman_distance_vs_llm_task_distance",
        ),
        (
            "Feature distance vs empirical Grossman distance",
            "feature_distance_vs_empirical_grossman_distance",
        ),
        ("GRU−Bari2019 vs feature distance", "gru_minus_bari_vs_feature_distance"),
        ("GRU−Bari2019 vs LLM distance", "gru_minus_bari_vs_llm_task_distance"),
        ("GRU−author vs feature distance", "gru_minus_author_vs_feature_distance"),
        ("GRU−author vs LLM distance", "gru_minus_author_vs_llm_task_distance"),
        ("Embedding distance vs feature distance", "embedding_distance_vs_feature_distance"),
        ("Embedding distance vs LLM distance", "embedding_distance_vs_llm_task_distance"),
    )
    relationship_table = [
        "| complete-case relationship | n | Spearman ρ | exact p | leave-one-cohort-out ρ |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in relationship_labels:
        relation = complete_case_relationships[key]
        loo = relation["leave_one_cohort_out_range"]
        relationship_table.append(
            f"| {label} | {relation['n_cohorts']} | {relation['spearman_rho']:+.3f} | "
            f"{relation['permutation_p_two_sided']:.4f} | [{loo[0]:+.2f}, {loo[1]:+.2f}] |"
        )
    block = "\n".join(
        [
            "[regenerated by `analysis/report_llm_task_similarity.py` — do not edit by hand]",
            "",
            "![Blinded LLM task-distance synthesis](../fig_llm_task_similarity.png)",
            "",
            "[Editable SVG](../fig_llm_task_similarity.svg)",
            "",
            "## Frozen judge protocol",
            "",
            "The judge saw 12 opaque task cards and the three AIND source-task prototypes. "
            "Every actor was called **subject**: species, study and author names, cohort size, "
            "GRU results, embeddings, and the hand-coded distance were absent. Response and "
            "reward apparatus remained visible as secondary task features. The frozen "
            "ordinal output implies all 66 unordered pairwise outcomes. Each was encoded in "
            "both A/B directions as an invariance audit, not as an independent model call.",
            "",
            "This is an exploratory **single-Codex-judge sensitivity analysis**. The pair "
            "rank therefore has no sampling-error bar. An independent multi-model replication "
            "would be required before treating this as a primary covariate. Because the judge "
            "ran in the existing analysis session, the blinding is at the prompt-card level, "
            "not an isolated-context guarantee. The visible performance and embedding error "
            "bars are SEM across three source seeds.",
            "",
            "## Ranking",
            "",
            *table,
            "",
            f"The three closest judged tasks are {top}; the three most distant are {bottom}.",
            "",
            "## Association with E8 transfer",
            "",
            f"Across all 12 valid cohorts, LLM task distance to AIND versus GRU−Bari2019 normalized-"
            f"likelihood advantage has Spearman ρ={q_relation['spearman_rho']:+.3f} "
            f"(permutation p={q_relation['permutation_p_two_sided']:.4f}). Among the "
            f"{author_relation['n_cohorts']} cohorts with an author-model reference, the "
            f"GRU−author association is ρ={author_relation['spearman_rho']:+.3f} "
            f"(p={author_relation['permutation_p_two_sided']:.4f}).",
            "",
            f"LLM task distance to AIND versus E8 embedding-centroid distance has Spearman "
            f"ρ={embedding_relation['spearman_rho']:+.3f} "
            f"(permutation p={embedding_relation['permutation_p_two_sided']:.4f}). Species "
            "colors are added only after unblinding and are descriptive; species was not "
            "available to the judge.",
            "",
            "## Consistency with the previous full-design distance",
            "",
            "![Previous full-design distance versus LLM task distance](../fig_full_design_vs_llm_task_distance.png)",
            "",
            "[Editable SVG](../fig_full_design_vs_llm_task_distance.svg)",
            "",
            "This comparison includes all 12 valid cohorts. The previous full-design score "
            "is the seven-axis categorical distance to the nearest AIND task prototype; the "
            "LLM distance is the frozen outcome-blind pairwise rank. They agree at Spearman "
            f"ρ={full_design_llm_relation['spearman_rho']:+.3f} "
            f"(permutation p={full_design_llm_relation['permutation_p_two_sided']:.4f}; "
            f"leave-one-cohort-out range "
            f"[{full_design_llm_relation['leave_one_cohort_out_range'][0]:+.2f}, "
            f"{full_design_llm_relation['leave_one_cohort_out_range'][1]:+.2f}]).",
            "",
            "## Complete-case mixed-feature cross-check",
            "",
            "![Complete-case feature-based and LLM task-distance cross-check](../fig_task_distance_crosscheck.png)",
            "",
            "[Editable SVG](../fig_task_distance_crosscheck.svg)",
            "",
            "This outcome-blind cross-check uses the seven cohorts whose canonical adapters "
            "expose complete trial-wise arm probabilities: Grossman (mouse), Chen (mouse), "
            "Zid (human), Lebedeva (mouse), Beron (mouse), Miller (rat), and Hattori (mouse). "
            "It combines seven categorical task/apparatus axes with seven empirical schedule "
            "metrics. Categorical axes are one-hot encoded, empirical metrics are standardized "
            "using the frozen seven-cohort population moments, and the two feature blocks each "
            "receive one-half total weight. PCA is fit without outcomes, embeddings, or species. "
            f"The first {feature_geometry['contract']['retained_components']} PCs explain "
            f"{100 * feature_geometry['contract']['retained_explained_variance']:.1f}% of feature variance.",
            "",
            "The three AIND categorical prototypes are exact, but their empirical coordinates "
            "are provisionally anchored to Grossman (mouse), because empirical schedule "
            "summaries stratified by the three AIND source curricula have not yet been frozen. "
            "Accordingly, this is a **feature-based distance to an AIND proxy**, not a final "
            "objective AIND-family distance.",
            "",
            *complete_table,
            "",
            *relationship_table,
            "",
            f"Feature-based distance and LLM task distance agree at ρ="
            f"{feature_llm_relation['spearman_rho']:+.3f} "
            f"(exact p={feature_llm_relation['permutation_p_two_sided']:.4f}). By contrast, "
            f"the empirical schedule-only distance to Grossman (mouse) relates to LLM task "
            f"distance at ρ={empirical_llm_relation['spearman_rho']:+.3f} "
            f"(exact p={empirical_llm_relation['permutation_p_two_sided']:.4f}). This contrast "
            "tests whether the LLM primarily tracks the categorical task description rather "
            "than the realized reward-probability dynamics.",
            "",
            f"The mixed feature distance itself closely tracks the empirical Grossman-centered "
            f"distance (ρ={feature_empirical_relation['spearman_rho']:+.3f}, exact "
            f"p={feature_empirical_relation['permutation_p_two_sided']:.4f}); its leading PCs "
            "are dominated by reward-schedule statistics despite equal block weighting. "
            "Hattori (mouse) is the clearest disagreement: the LLM places its coupled-baiting "
            "task close to the AIND family, while the provisional feature geometry penalizes "
            "its empirical schedule relative to the Grossman (mouse) anchor. The retained-PC "
            f"and all-PC feature-distance rankings are identical (ρ="
            f"{pca_sensitivity_relation['spearman_rho']:+.3f}), so this conclusion is not "
            "caused by the 90%-variance truncation.",
            "",
            f"Within these seven cohorts, embedding displacement relates more strongly to LLM "
            f"task distance (ρ={llm_embedding_complete_relation['spearman_rho']:+.3f}, exact "
            f"p={llm_embedding_complete_relation['permutation_p_two_sided']:.4f}) than to the "
            f"feature-based proxy (ρ={feature_embedding_relation['spearman_rho']:+.3f}, exact "
            f"p={feature_embedding_relation['permutation_p_two_sided']:.4f}). This is exploratory "
            "at n=7 and should not be interpreted as a formal difference between correlations.",
        ]
    )
    _replace_report(block)


if __name__ == "__main__":
    main()
