"""Render the frozen Study 09 LLM task-similarity analysis."""

from __future__ import annotations

import hashlib
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
from plot_style import apply_presentation_style  # noqa: E402


JUDGMENTS = STUDY / "analysis" / "llm_task_similarity_judgments.json"
CARDS = STUDY / "analysis" / "llm_task_cards.json"
GENERALIZATION = STUDY / "analysis" / "generalization_drivers_e8.json"
RESULTS = STUDY / "analysis" / "llm_task_similarity_results.json"
FIGURE_PNG = STUDY / "analysis" / "fig_llm_task_similarity.png"
FIGURE_SVG = STUDY / "analysis" / "fig_llm_task_similarity.svg"
REPORT = STUDY / "analysis" / "reports" / "r6-llm-task-similarity.md"
START = "<!-- BEGIN result-6 -->"
END = "<!-- END result-6 -->"

RNG_SEED = 20260915
N_PERMUTATIONS = 100_000
SPECIES_COLORS = {
    "mouse": "#4C72B0",
    "rat": "#DD8452",
    "macaque": "#C44E52",
    "human": "#8172B3",
}
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
    permutations = np.asarray(
        [rng.permutation(y_centered) for _ in range(N_PERMUTATIONS)]
    )
    null = permutations @ x_centered / denominator
    p_value = float((np.count_nonzero(np.abs(null) >= abs(rho)) + 1) / (N_PERMUTATIONS + 1))
    return {
        "n_cohorts": int(len(x)),
        "spearman_rho": rho,
        "permutation_p_two_sided": p_value,
        "n_permutations": N_PERMUTATIONS,
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
        },
        "contract": {
            "llm_task_distance_to_aind": (
                "One minus the fraction of pairwise comparisons judged closer to AIND; "
                "ties contribute one-half win. Each candidate is compared with all 11 "
                "alternatives, so zero means closest and one means farthest."
            ),
            "uncertainty": (
                "The LLM task distance has no error bar because this first pass has one judge. "
                "Performance and embedding error bars are SEM across three source seeds."
            ),
            "performance_metric": (
                "E8 D=614 three-source-seed mean subject-balanced normalized-likelihood difference."
            ),
            "correlation": (
                "Cohort-level Spearman correlation with deterministic 100,000-draw two-sided "
                "permutation p-value."
            ),
            "rng_seed": RNG_SEED,
        },
        "judge": judgments["judge"],
        "audit": judgments["audit"],
        "ranking": rows,
        "relationships": relationships,
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

    top = ", ".join(row["label"] for row in rows[:3])
    bottom = ", ".join(row["label"] for row in rows[-3:])
    q_relation = relationships["gru_minus_bari_vs_llm_task_distance"]
    author_relation = relationships["gru_minus_author_vs_llm_task_distance"]
    embedding_relation = relationships["embedding_distance_vs_llm_task_distance"]
    table = [
        "| rank | study | tier | LLM task distance to AIND |",
        "|---:|---|---|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        table.append(
            f"| {rank} | {row['label']} | {row['analysis_tier'].replace('_', ' ')} | "
            f"{row['llm_task_distance_to_aind']:.3f} |"
        )
    block = "\n".join(
        [
            "[regenerated by `analysis/report_llm_task_similarity.py` — do not edit by hand]",
            "",
            "![Blinded LLM task-similarity synthesis](../fig_llm_task_similarity.png)",
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
        ]
    )
    _replace_report(block)


if __name__ == "__main__":
    main()
