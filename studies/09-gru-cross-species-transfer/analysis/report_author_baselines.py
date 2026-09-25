"""Render the Stage-A external-transfer decision report from frozen results."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import statistics
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import apply_presentation_style  # noqa: E402


AUTHOR_DATA = STUDY / "analysis" / "author_baseline_results.json"
MATCHED_DATA = STUDY / "analysis" / "matched_half_results.json"
EXAMPLE_DATA = STUDY / "analysis" / "example_behavior_sessions.json"
VALIDATION_DATA = STUDY / "analysis" / "dataset_suite_validation.json"
EMBEDDING_DIMENSION_DATA = STUDY / "analysis" / "embedding_dimension_results.json"
EMBEDDING_DIMENSION_EXPANSION_DATA = (
    STUDY / "analysis" / "embedding_dimension_expansion_results.json"
)
TASK_DESIGN_DATA = STUDY / "analysis" / "task_design_annotations.json"
SURVEY = STUDY / "DATASET_SURVEY.md"
FIGURE = STUDY / "analysis" / "fig_author_baseline_likelihood.png"
SUBJECT_FIGURE = STUDY / "analysis" / "fig_subject_baseline_likelihood.png"
GRU_Q_SUBJECT_FIGURE = STUDY / "analysis" / "fig_subject_gru_minus_q_likelihood.png"
REPORT = STUDY / "analysis" / "reports" / "r1-author-aligned-baselines.md"
START = "<!-- BEGIN result-1 -->"
END = "<!-- END result-1 -->"
DS = (10, 30, 100, 300, 614)
ALL_DATASET_ORDER = (
    "grossman",
    "chen",
    "zid",
    "lebedeva",
    "beron",
    "kwak",
    "miller",
    "findling",
    "tang",
    "alsio",
    "eckstein",
    "costa",
    "lopez_mouse",
)
DATASET_ORDER = tuple(name for name in ALL_DATASET_ORDER if name != "kwak")
TIER_ORDER = ("primary", "stress_test", "descriptive_only")
TIER_LABELS = {
    "primary": "Primary",
    "stress_test": "Stress test",
    "descriptive_only": "Descriptive",
}
SPECIES_COLORS = {
    "mouse": "#4C72B0",
    "rat": "#DD8452",
    "macaque": "#C44E52",
    "human": "#8172B3",
}
E4_COLOR = "#6BAED6"
E8_COLOR = "#17365D"
LABELS = {
    "grossman": "Grossman (mouse)",
    "chen": "Chen (mouse)",
    "zid": "Zid (human)",
    "lebedeva": "Lebedeva (mouse)",
    "beron": "Beron (mouse)",
    "kwak": "Kwak (mouse)",
    "miller": "Miller (rat)",
    "findling": "Findling (human)",
    "tang": "Tang (macaque)",
    "alsio": "Alsiö (rat)",
    "eckstein": "Eckstein (human)",
    "costa": "Costa (macaque)",
    "lopez_mouse": "López-Yépez (mouse)",
}
TASKS = {
    "grossman": "blockwise dynamic foraging",
    "chen": "restless random walk",
    "zid": "restless random walk",
    "lebedeva": "80/20 probabilistic reversal",
    "beron": "nonstationary bandit",
    "kwak": "dynamic bandit under D1/D2 manipulation",
    "miller": "large dynamic bandit",
    "findling": "variable-volatility reversal",
    "tang": "blockwise action/object values",
    "alsio": "discrimination and reversal",
    "eckstein": "developmental stochastic reversal",
    "costa": "stochastic stimulus reversal",
    "lopez_mouse": "baited variable-interval matching",
}
AUTHOR_LABELS = {
    "grossman-meta-learning": "meta-learning RL",
    "chen-rlck": "4-parameter RLCK",
    "zid-traditional-rlck": "traditional RLCK",
    "zid-history-kernel-foraging": "HK2 foraging RL",
    "lebedeva-pr": "PR",
    "beron-rflr": "RFLR",
    "miller-rhg": "RHG",
    "findling-weber-imprecision": "Weber-imprecision BI",
    "findling-weber-imprecision-64p": "Weber BI (64-particle fit)",
    "eckstein-rl": "counterfactual RL",
    "eckstein-bi": "Bayesian inference",
}
AUTHOR_REFERENCE_PANELS = (
    ("grossman", "grossman-meta-learning"),
    ("chen", "chen-rlck"),
    ("zid", "zid-history-kernel-foraging"),
    ("lebedeva", "lebedeva-pr"),
    ("beron", "beron-rflr"),
    ("miller", "miller-rhg"),
    ("findling", "findling-weber-imprecision"),
    ("eckstein", "eckstein-rl"),
    ("eckstein", "eckstein-bi"),
)
EXAMPLE_CATEGORIES = (
    ("lower", "Lower tail"),
    ("median", "Median"),
    ("upper", "Upper tail"),
)
PRIMARY_AUTHOR_COHORTS = (
    "lebedeva",
    "beron",
    "miller",
    "findling",
    "eckstein",
)


def _metric(record: dict) -> float:
    return float(record["metrics"]["normalized_likelihood"])


def _gru_for_d(dataset: dict, d: int) -> list[dict]:
    rows = [row for row in dataset["gru"] if int(row["nominal_D"]) == d]
    if len(rows) != 3:
        raise AssertionError(f"Expected three GRU seeds for D={d}")
    return rows


def _validation_map(validation: dict) -> dict[str, dict]:
    rows = {row["dataset"]: row for row in validation["datasets"]}
    if tuple(rows) != ALL_DATASET_ORDER:
        raise AssertionError("Validation dataset order or membership drifted")
    return rows


def _subject_differences(dataset: dict, d: int) -> list[float]:
    q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
    seeds = [
        row["metrics"]["per_subject_mean_log_likelihood_nats"]
        for row in _gru_for_d(dataset, d)
    ]
    if any(set(seed) != set(q) for seed in seeds):
        raise AssertionError("GRU/Q subject keys do not align")
    return [
        math.exp(statistics.mean(float(seed[key]) for seed in seeds))
        - math.exp(float(q[key]))
        for key in sorted(q)
    ]


def _wilcoxon(values: list[float]) -> float:
    if not any(value != 0 for value in values):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def _p(value: float) -> str:
    return "<.001" if value < 0.001 else f"={value:.3f}".replace("0.", ".")


def _summary_order(
    author_data: dict,
    matched: dict,
    analysis_tiers: dict[str, list[str]],
) -> tuple[tuple[str, str], ...]:
    records = author_data["records"]
    ordered: list[tuple[str, str]] = []
    for tier in TIER_ORDER:
        names = [name for name in analysis_tiers[tier] if name in DATASET_ORDER]

        def advantage(dataset_name: str) -> float:
            dataset = matched["datasets"][dataset_name]
            gru_e4 = statistics.mean(
                _metric(row) for row in _gru_for_d(dataset, 614)
            )
            author_values = [
                _metric(record)
                for record in records.values()
                if record["dataset"] == dataset_name and record["author_selected"]
            ]
            reference = max(author_values) if author_values else _metric(dataset["q"])
            return gru_e4 - reference

        ordered.extend(
            (tier, name) for name in sorted(names, key=advantage, reverse=True)
        )
    if {name for _, name in ordered} != set(DATASET_ORDER):
        raise AssertionError("Analysis tiers do not cover every displayed dataset")
    return tuple(ordered)


def _plot_summary(
    author_data: dict,
    matched: dict,
    validation: dict[str, dict],
    embedding_dimension: dict,
    analysis_tiers: dict[str, list[str]],
) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    records = author_data["records"]
    panel_order = _summary_order(author_data, matched, analysis_tiers)
    for axis, (tier, dataset_name) in zip(axes.flat, panel_order):
        dataset = matched["datasets"][dataset_name]
        means = []
        sds = []
        for d in DS:
            values = [_metric(row) for row in _gru_for_d(dataset, d)]
            means.append(statistics.mean(values))
            sds.append(statistics.stdev(values))
            axis.scatter([d] * 3, values, s=18, color=E4_COLOR, alpha=0.4)
        axis.errorbar(
            DS,
            means,
            yerr=sds,
            marker="o",
            color=E4_COLOR,
            capsize=2,
            linewidth=1.6,
            label="E4 GRU mean ± SD",
        )
        if dataset_name in embedding_dimension["datasets"]:
            e8_values = [
                _metric(row)
                for row in embedding_dimension["datasets"][dataset_name]["e8"]
            ]
            axis.scatter(
                [614] * len(e8_values),
                e8_values,
                s=28,
                color=E8_COLOR,
                alpha=0.45,
                zorder=4,
            )
            axis.errorbar(
                [614],
                [statistics.mean(e8_values)],
                yerr=[statistics.stdev(e8_values)],
                marker="D",
                markersize=5,
                color=E8_COLOR,
                capsize=3,
                linewidth=1.8,
                label="E8 GRU D=614 mean ± SD",
                zorder=5,
            )
        axis.axhline(
            _metric(dataset["q"]), color="#222222", linestyle="--", label="common Q"
        )
        for baseline, record in records.items():
            if record["dataset"] != dataset_name:
                continue
            selected = bool(record["author_selected"])
            axis.axhline(
                _metric(record),
                color="#C44E52" if selected else "#DD8452",
                linestyle="-" if selected else ":",
                linewidth=1.8,
                label=AUTHOR_LABELS[baseline],
            )
        audit = validation[dataset_name]
        details = (
            f"v{audit['schema_version']} · "
            f"n={audit['num_subjects']}, sessions={audit['num_sessions']}, "
            f"test trials={audit['num_test_trials']:,}"
        )
        axis.set_title(
            f"{TIER_LABELS[tier]} · {LABELS[dataset_name]}\n"
            f"{textwrap.fill(TASKS[dataset_name], 34)}\n{details}",
            fontsize=9,
            color=SPECIES_COLORS[audit["species"]],
        )
        axis.set_xscale("log")
        axis.set_xticks(DS, [str(value) for value in DS])
        axis.set_xlabel("Source subjects D")
        axis.set_ylabel("Held-out normalized likelihood")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=6.5, loc="best")
    for axis in axes.flat[len(panel_order) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Frozen-core GRU transfer versus matched common Q and author models\n"
        "Tiered cohorts; within each tier, descending E4 D=614 advantage",
        fontsize=17,
    )
    fig.savefig(FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _author_subject_conditions(
    dataset_name: str,
    selected_baseline: str,
    author_data: dict,
    dataset: dict,
    embedding_dataset: dict,
) -> tuple[
    str,
    list[str],
    list[list[float]],
    list[str],
    list[float],
    tuple[float, float],
]:
    records = [
        (key, record)
        for key, record in author_data["records"].items()
        if record["dataset"] == dataset_name
    ]
    selected_record = dict(records).get(selected_baseline)
    if selected_record is None or not selected_record["author_selected"]:
        raise AssertionError(
            "Requested author reference is unavailable or not selected"
        )
    comparators = sorted(
        (key, record) for key, record in records if key != selected_baseline
    )
    q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
    subjects = sorted(q)
    reference = selected_record["metrics"]["per_subject_mean_log_likelihood_nats"]
    if set(reference) != set(q):
        raise AssertionError("Author and Q per-subject metric sets do not align")

    labels = ["Common Q"]
    log_values = [[float(q[subject]) for subject in subjects]]
    colors = ["#666666"]
    for baseline, record in comparators:
        values = record["metrics"]["per_subject_mean_log_likelihood_nats"]
        if set(values) != set(q):
            raise AssertionError("Author and Q per-subject metric sets do not align")
        labels.append(AUTHOR_LABELS[baseline])
        log_values.append([float(values[subject]) for subject in subjects])
        colors.append("#DD8452")
    for d in DS:
        seeds = [
            row["metrics"]["per_subject_mean_log_likelihood_nats"]
            for row in _gru_for_d(dataset, d)
        ]
        if any(set(seed) != set(reference) for seed in seeds):
            raise AssertionError("Author and GRU per-subject metric sets do not align")
        labels.append(f"GRU D={d}")
        log_values.append(
            [
                statistics.mean(float(seed[subject]) for seed in seeds)
                for subject in subjects
            ]
        )
        colors.append(E4_COLOR)

    e8_seeds = [
        row["metrics"]["per_subject_mean_log_likelihood_nats"]
        for row in embedding_dataset["e8"]
    ]
    if any(set(seed) != set(reference) for seed in e8_seeds):
        raise AssertionError("Author and E8 GRU per-subject metric sets do not align")
    labels.append("E8 GRU D=614")
    log_values.append(
        [
            statistics.mean(float(seed[subject]) for seed in e8_seeds)
            for subject in subjects
        ]
    )
    colors.append(E8_COLOR)

    reference_likelihood = [math.exp(float(reference[subject])) for subject in subjects]
    differences = [
        [
            math.exp(value) - reference_value
            for value, reference_value in zip(values, reference_likelihood)
        ]
        for values in log_values
    ]
    p_values = [_wilcoxon(values) for values in differences]
    correlations = (
        float(np.corrcoef(reference_likelihood, differences[-2])[0, 1]),
        float(np.corrcoef(reference_likelihood, differences[-1])[0, 1]),
    )
    return (
        AUTHOR_LABELS[selected_baseline],
        labels,
        differences,
        colors,
        p_values,
        correlations,
    )


def _plot_author_subjects(
    author_data: dict,
    matched: dict,
    embedding_dimension: dict,
) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)
    for axis, (dataset_name, selected_baseline) in zip(
        axes.flat, AUTHOR_REFERENCE_PANELS
    ):
        reference_label, labels, values, colors, p_values, correlations = (
            _author_subject_conditions(
                dataset_name,
                selected_baseline,
                author_data,
                matched["datasets"][dataset_name],
                embedding_dimension["datasets"][dataset_name],
            )
        )
        positions = np.arange(len(labels))
        n_subjects = len(values[0])
        jitter = np.asarray([((index % 17) - 8) / 80 for index in range(n_subjects)])
        for subject_index in range(n_subjects):
            axis.plot(
                positions + jitter[subject_index],
                [condition[subject_index] for condition in values],
                color="#777777",
                alpha=0.08,
                linewidth=0.45,
                zorder=1,
            )
        violins = axis.violinplot(
            values,
            positions=positions,
            widths=0.72,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)
        violins["cmedians"].set_color(colors)
        violins["cmedians"].set_linewidth(1.8)
        for position, condition, color, p_value in zip(
            positions, values, colors, p_values
        ):
            axis.scatter(
                position + jitter,
                condition,
                s=8,
                color=color,
                alpha=0.34,
                linewidths=0,
                zorder=3,
            )
            axis.scatter(
                position,
                statistics.mean(condition),
                s=38,
                marker="D",
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=5,
            )
            axis.text(
                position,
                0.985,
                f"p{_p(p_value)}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 1,
                },
            )
        tick_labels = [
            label.replace("4-parameter ", "4-param\n")
            .replace("traditional ", "traditional\n")
            .replace("Common Q", "Common\nQ")
            .replace("counterfactual RL", "counterfactual\nRL")
            .replace("Bayesian inference", "Bayesian\ninference")
            .replace("Weber BI (64-particle fit)", "Weber BI\n(64-particle fit)")
            .replace("GRU ", "GRU\n")
            for label in labels
        ]
        axis.set_xticks(positions, tick_labels)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_title(
            f"{LABELS[dataset_name]} (n={n_subjects})\n"
            f"{TASKS[dataset_name]}\n"
            f"Reference: {reference_label}\n"
            "D=614 corr(author likelihood, GRU Δ)\n"
            f"E4 r={correlations[0]:+.2f}, E8 r={correlations[1]:+.2f}",
            fontsize=10,
        )
        axis.axhline(0, color="#C44E52", linewidth=1.6, alpha=0.8, zorder=0)
        axis.set_yscale("symlog", linthresh=0.01)
        axis.set_ylabel("Δ subject held-out normalized likelihood")
        axis.grid(axis="y", alpha=0.2)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#333333", linewidth=1.8, label="median"),
            Line2D(
                [0],
                [0],
                marker="D",
                markerfacecolor="white",
                markeredgecolor="#333333",
                color="none",
                label="mean",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Subject likelihood relative to each paper's author-selected model",
        fontsize=17,
    )
    fig.savefig(SUBJECT_FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_gru_q_subjects(
    matched: dict,
    validation: dict[str, dict],
    embedding_dimension: dict,
) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)
    rng = np.random.default_rng(29)
    for axis, dataset_name in zip(axes.flat, DATASET_ORDER):
        dataset = matched["datasets"][dataset_name]
        q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
        subjects = sorted(q)
        e8_seeds = [
            row["metrics"]["per_subject_mean_log_likelihood_nats"]
            for row in embedding_dimension["datasets"][dataset_name]["e8"]
        ]
        if any(set(seed) != set(q) for seed in e8_seeds):
            raise AssertionError("E8 GRU/Q subject keys do not align")
        e8_values = [
            math.exp(statistics.mean(float(seed[subject]) for seed in e8_seeds))
            - math.exp(float(q[subject]))
            for subject in subjects
        ]
        values = [_subject_differences(dataset, d) for d in DS] + [e8_values]
        colors = [E4_COLOR] * len(DS) + [E8_COLOR]
        positions = np.arange(len(values))
        n_subjects = len(values[0])
        for subject_index in range(n_subjects):
            axis.plot(
                positions,
                [condition[subject_index] for condition in values],
                color="#777777",
                alpha=min(0.14, 5 / n_subjects),
                linewidth=0.35,
                zorder=1,
            )
        violins = axis.violinplot(
            values,
            positions=positions,
            widths=0.72,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.22)
        violins["cmedians"].set_color(colors)
        violins["cmedians"].set_linewidth(1.5)
        for position, condition, color in zip(positions, values, colors):
            jitter = rng.uniform(-0.15, 0.15, len(condition))
            axis.scatter(
                position + jitter,
                condition,
                s=6,
                color=color,
                alpha=min(0.38, 12 / n_subjects),
                linewidths=0,
                zorder=2,
            )
            axis.scatter(
                position,
                statistics.mean(condition),
                s=30,
                marker="D",
                facecolor="white",
                edgecolor=color,
                linewidth=1.2,
                zorder=4,
            )
            axis.text(
                position,
                0.98,
                f"p{_p(_wilcoxon(condition))}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.3,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.7,
                    "pad": 0.5,
                },
            )
        audit = validation[dataset_name]
        axis.axhline(0, color="#222222", linewidth=1)
        axis.set_xticks(
            positions,
            [f"E4\n{value}" for value in DS] + ["E8\n614"],
        )
        axis.set_title(
            f"{LABELS[dataset_name]} · "
            f"v{audit['schema_version']} · n={n_subjects}\n"
            f"{textwrap.fill(TASKS[dataset_name], 34)}",
            fontsize=9,
        )
        axis.set_ylabel("Subject GRU − Q likelihood")
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(DATASET_ORDER) :]:
        axis.set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], color=E4_COLOR, linewidth=4, label="E4"),
            Line2D([0], [0], color=E8_COLOR, linewidth=4, label="E8"),
            Line2D(
                [0],
                [0],
                marker="D",
                markerfacecolor="white",
                markeredgecolor="#333333",
                color="none",
                label="mean",
            ),
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Paired subject-level GRU improvement over common Q\n"
        "Dots are subjects; thin lines connect the same subject across conditions",
        fontsize=17,
    )
    fig.savefig(GRU_Q_SUBJECT_FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_examples(example_data: dict) -> list[Path]:
    module = importlib.import_module(
        "aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session"
    )
    if (
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        != example_data["plotting"]["source_sha256"]
    ):
        raise AssertionError(
            "Installed plot_foraging_session.py differs from frozen source"
        )
    plot_foraging_session = module.plot_foraging_session
    paths = []
    for dataset_name in DATASET_ORDER:
        examples = example_data["datasets"][dataset_name]["examples"]
        for category, category_label in EXAMPLE_CATEGORIES:
            selected = [
                example for example in examples if example["selection_slug"] == category
            ]
            if not selected:
                continue
            apply_presentation_style()
            fig, holders = plt.subplots(
                len(selected),
                1,
                figsize=(15, 2.9 * len(selected)),
                dpi=160,
                squeeze=False,
            )
            legend_handles = legend_labels = None
            for holder, example in zip(holders[:, 0], selected):
                probability = np.asarray(
                    [
                        example["reward_probability_arm_0"],
                        example["reward_probability_arm_1"],
                    ],
                    dtype=float,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    _, axes = plot_foraging_session(
                        choice_history=np.asarray(example["choice"], dtype=float),
                        reward_history=np.asarray(example["reward"], dtype=bool),
                        p_reward=probability,
                        smooth_factor=9,
                        ax=holder,
                        vertical=False,
                        plot_list=(
                            ["choice", "reward_prob"]
                            if example["reward_probability_available"]
                            else ["choice"]
                        ),
                    )
                current_legend = axes[0].get_legend()
                if legend_handles is None:
                    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
                if current_legend is not None:
                    current_legend.remove()
                delta = example["gru_d614_minus_q_normalized_likelihood"]
                axes[0].set_title(
                    f"{example['subject_id']} · {example['session_id']} · "
                    f"GRU−Q={delta:+.3f}",
                    fontsize=9,
                    loc="left",
                )
                boundary = example["adapt_prefix_trials"]
                if boundary is not None:
                    for axis in axes:
                        axis.axvline(
                            boundary + 0.5, color="#7B3294", linestyle="--", lw=1
                        )
                if not example["reward_probability_available"]:
                    axes[1].text(
                        0.5,
                        0.5,
                        "reward schedule not available in public release",
                        transform=axes[1].transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#555555",
                    )
            if legend_handles:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    fontsize=7,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.965),
                    ncol=4,
                    frameon=True,
                )
            fig.suptitle(
                f"{LABELS[dataset_name]}: {category_label} held-out sessions",
                fontsize=14,
                y=0.995,
            )
            fig.subplots_adjust(
                left=0.1,
                right=0.98,
                bottom=0.06,
                top=0.88,
                hspace=0.62,
            )
            path = (
                STUDY
                / "analysis"
                / f"fig_example_sessions_{dataset_name}_{category}.png"
            )
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def _survey_section(start_heading: str, end_heading: str | None = None) -> str:
    text = SURVEY.read_text()
    start = text.index(start_heading) + len(start_heading)
    end = text.index(end_heading, start) if end_heading else len(text)
    return text[start:end].strip()


def _mean_sd(values: list[float]) -> str:
    return f"{statistics.mean(values):.5f} ± {statistics.stdev(values):.5f}"


def _author_rows(
    author_data: dict,
    matched: dict,
    embedding_dimension: dict,
) -> tuple[list[str], list[str]]:
    rows = []
    correlations = []
    for baseline, record in author_data["records"].items():
        dataset_name = record["dataset"]
        dataset = matched["datasets"][dataset_name]
        d614 = _gru_for_d(dataset, 614)
        e8 = embedding_dimension["datasets"][dataset_name]["e8"]
        role = (
            "author-selected"
            if record["author_selected"]
            else record.get("comparison_role", "paper comparator")
        )
        rows.append(
            f"| {LABELS[dataset_name]} | {AUTHOR_LABELS[baseline]} | "
            f"{role} | "
            f"{_metric(dataset['q']):.5f} | {_metric(record):.5f} | "
            f"{_mean_sd([_metric(row) for row in d614])} | "
            f"{_mean_sd([_metric(row) for row in e8])} |"
        )
        if record["author_selected"]:
            author = record["metrics"]["per_subject_mean_log_likelihood_nats"]
            e4_seeds = [
                row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in d614
            ]
            e8_seeds = [
                row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in e8
            ]
            if any(
                set(seed) != set(author) for seed in [*e4_seeds, *e8_seeds]
            ):
                raise AssertionError("Author/GRU subject keys do not align")
            reference = [math.exp(float(author[key])) for key in sorted(author)]
            correlations_by_dimension = []
            for seeds in (e4_seeds, e8_seeds):
                improvement = [
                    math.exp(statistics.mean(float(seed[key]) for seed in seeds))
                    - math.exp(float(author[key]))
                    for key in sorted(author)
                ]
                correlations_by_dimension.append(
                    float(np.corrcoef(reference, improvement)[0, 1])
                )
            correlations.append(
                f"| {LABELS[dataset_name]} | {AUTHOR_LABELS[baseline]} | "
                f"{len(reference)} | {correlations_by_dimension[0]:+.2f} | "
                f"{correlations_by_dimension[1]:+.2f} |"
            )
    return rows, correlations


def _author_subject_rows(
    author_data: dict,
    matched: dict,
    embedding_dimension: dict,
) -> list[str]:
    rows = []
    for dataset_name, selected_baseline in AUTHOR_REFERENCE_PANELS:
        reference_label, labels, values, _, p_values, _ = _author_subject_conditions(
            dataset_name,
            selected_baseline,
            author_data,
            matched["datasets"][dataset_name],
            embedding_dimension["datasets"][dataset_name],
        )
        for label, differences, p_value in zip(labels, values, p_values):
            rows.append(
                f"| {LABELS[dataset_name]} | {reference_label} | {label} | "
                f"{statistics.median(differences):+.5f} | "
                f"{statistics.mean(differences):+.5f} | {p_value:.3g} |"
            )
    return rows


def _primary_author_read(author_data: dict, matched: dict) -> list[str]:
    lines = [
        "### Primary author-model scientific read",
        "",
        "Trial-pooled held-out ranking under the matched-half protocol:",
        "",
    ]
    for dataset_name in PRIMARY_AUTHOR_COHORTS:
        dataset = matched["datasets"][dataset_name]
        scores = [
            ("common Q", _metric(dataset["q"])),
            (
                "GRU D=614",
                statistics.mean(_metric(row) for row in _gru_for_d(dataset, 614)),
            ),
        ]
        scores.extend(
            (AUTHOR_LABELS[key], _metric(record))
            for key, record in author_data["records"].items()
            if record["dataset"] == dataset_name and record["author_selected"]
        )
        ranking = " > ".join(
            f"{label} ({value:.5f})"
            for label, value in sorted(scores, key=lambda item: item[1], reverse=True)
        )
        lines.append(f"- **{LABELS[dataset_name]}:** {ranking}")

    primary = _metric(author_data["records"]["findling-weber-imprecision"])
    sensitivity = _metric(
        author_data["records"]["findling-weber-imprecision-64p"]
    )
    q_score = _metric(matched["datasets"]["findling"]["q"])
    lines += [
        "",
        "For Findling (human), increasing only the fit particle count from 2 to 64 "
        f"raises held-out likelihood from {primary:.5f} to {sensitivity:.5f} "
        f"(Δ={sensitivity - primary:+.5f}), nearly reaching common Q ({q_score:.5f}). "
        "The released two-particle fitting objective therefore contributes material "
        "Monte Carlo instability. The two-particle result remains the primary "
        "author-code-parity reference; the 64-particle result is a sensitivity, not a "
        "replacement author-selected model.",
        "",
        "These rankings concern held-out prediction after equal adaptation data. They do "
        "not recreate the papers' original full-data, hierarchical, or information-criterion "
        "model-selection analyses.",
    ]
    return lines


def _stage_a_read(matched: dict) -> list[str]:
    results = {}
    for dataset_name in DATASET_ORDER:
        dataset = matched["datasets"][dataset_name]
        values = _subject_differences(dataset, 614)
        pooled = statistics.mean(
            _metric(row) for row in _gru_for_d(dataset, 614)
        ) - _metric(dataset["q"])
        results[dataset_name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p": _wilcoxon(values),
            "fraction": sum(value > 0 for value in values) / len(values),
            "pooled": pooled,
        }

    def describe(names: list[str]) -> str:
        return ", ".join(
            f"**{LABELS[name]}** (mean Δ={results[name]['mean']:+.5f}, "
            f"p={results[name]['p']:.3g})"
            for name in names
        )

    gru_better = [
        name
        for name in DATASET_ORDER
        if results[name]["p"] < 0.05 and results[name]["mean"] > 0
    ]
    q_better = [
        name
        for name in DATASET_ORDER
        if results[name]["p"] < 0.05 and results[name]["mean"] < 0
    ]
    unresolved = [name for name in DATASET_ORDER if results[name]["p"] >= 0.05]
    direction_mismatch = [
        name
        for name in DATASET_ORDER
        if results[name]["pooled"] * results[name]["mean"] < 0
    ]
    scaling = {
        name: statistics.mean(
            _metric(row) for row in _gru_for_d(matched["datasets"][name], 614)
        )
        - statistics.mean(
            _metric(row) for row in _gru_for_d(matched["datasets"][name], 10)
        )
        for name in DATASET_ORDER
    }
    lines = [
        "### Stage-A scientific read",
        "",
        "At D=614, the exploratory unadjusted subject-paired Wilcoxon result favors GRU "
        f"for {describe(gru_better)}.",
        "",
        "It favors common Q for " + describe(q_better) + ".",
        "",
        "The remaining cohorts are unresolved at the 0.05 level: "
        + describe(unresolved)
        + ". Tang (macaque) has only two subjects, so its inferential result is especially limited.",
        "",
        f"Every cohort improves in trial-pooled GRU likelihood from D=10 to D=614. "
        f"The largest gains are "
        + ", ".join(
            f"**{LABELS[name]}** ({value:+.5f})"
            for name, value in sorted(
                scaling.items(), key=lambda item: item[1], reverse=True
            )[:4]
        )
        + ". Several curves peak at D=100 or D=300, so the evidence supports scaling "
        "the source population but not a universal optimum at the largest D.",
    ]
    if direction_mismatch:
        name = direction_mismatch[0]
        result = results[name]
        lines += [
            "",
            f"**Aggregation warning — {LABELS[name]}.** The trial-pooled D=614 score "
            f"favors GRU by {result['pooled']:+.5f}, while the arithmetic mean subject "
            f"difference is {result['mean']:+.5f} (median {result['median']:+.5f}; "
            f"{result['fraction']:.0%} of subjects favor GRU; p={result['p']:.3g}). "
            "All Zid (human) subjects contribute the same 150 held-out trials, so this reversal is "
            "not unequal trial weighting. It reflects the nonlinear difference between a "
            "geometric pooled likelihood and arithmetic per-subject likelihood differences "
            "in the heterogeneous Zid (human) distribution. The subject-paired result is primary for "
            "claims about a typical subject; the pooled score remains descriptive of total "
            "trial prediction.",
        ]
    lines += [
        "",
        "This screen therefore supports broad transfer, but not universal superiority over "
        "a fitted subject-level Q model. López-Yépez (mouse), Grossman (mouse), Lebedeva "
        "(mouse), and Chen (mouse) are the positive-transfer cases; Findling (human), "
        "Eckstein (human), Miller (rat), and subject-balanced Zid (human) are the main "
        "valid stress tests that motivated the primary-set author-model reproductions "
        "reported below.",
    ]
    return lines


def _result_block(
    author_data: dict,
    matched: dict,
    examples: dict,
    validation: dict[str, dict],
    embedding_dimension: dict,
) -> str:
    author_rows, correlations = _author_rows(
        author_data,
        matched,
        embedding_dimension,
    )
    author_subject_rows = _author_subject_rows(
        author_data,
        matched,
        embedding_dimension,
    )
    actual_ds = {
        d: sorted(
            {
                int(row["actual_D"])
                for dataset in matched["datasets"].values()
                for row in _gru_for_d(dataset, d)
            }
        )
        for d in DS
    }
    lines = [
        "[regenerated by `analysis/report_author_baselines.py` — do not edit by hand]",
        "",
        "## Stage-A decision result",
        "",
        "![GRU, common Q, and available author baselines](../fig_author_baseline_likelihood.png)",
        "",
        "Every model uses the same immutable adaptation and held-out observations. "
        "Panels are grouped as primary, stress test, and descriptive, then ordered within "
        "each tier by descending E4 D=614 GRU advantage over the strongest available "
        "author-selected model. Stress-test and descriptive cohorts without a reproduced "
        "author model use common Q as the ordering reference. Panel-title color encodes "
        "species using the same palette as the task-design figures. Light-blue GRU points "
        "and curves are the historical E4 screen; dark-blue D=614 overlays are E8 and now "
        "appear for every displayed cohort. The five-cohort diagnostic E8 values have a "
        "paired current-code E4 comparator in Result 4; the seven expansion values are "
        "shown against historical E4 here and should not be interpreted as an isolated "
        "embedding-dimension effect. "
        "GRU points are the three source-training seeds; summaries are their mean ± SD. "
        "Common Q is fitted independently per target subject on the identical adaptation half. "
        "Author-model lines include the existing Grossman (mouse), Chen (mouse), and "
        "Zid (human) fits plus the primary-set reproductions for Lebedeva (mouse), "
        "Beron (mouse), Miller (rat), Findling (human), and both Eckstein (human) "
        "co-winners.",
        "",
        "Kwak (mouse) is omitted from every figure, table, direction count, and inference in "
        "this report. Its frozen manifest adapts on CNO sessions and tests on DMSO sessions, "
        "which confounds subject adaptation with treatment transfer. Readmission requires a "
        "new DMSO/control-only run using chronological odd DMSO sessions for adaptation and "
        "chronological even DMSO sessions for testing.",
        "",
        "![Subject-level likelihood relative to the author-selected model](../fig_subject_baseline_likelihood.png)",
        "",
        "Every displayed subject likelihood is relative to the author model named in "
        "that panel. Eckstein (human) has separate panels for its two co-winners. The "
        "red zero line is the author reference; "
        "positive values favor the displayed model. The panel title reports the correlation "
        "between author-model likelihood and D=614 GRU improvement for E4 and E8. "
        "Light blue denotes E4 and dark blue denotes E8. This preserves the "
        "author-relative comparison from the completed first-round report.",
        "",
        "![Paired subject-level GRU minus common-Q likelihood](../fig_subject_gru_minus_q_likelihood.png)",
        "",
        "Each dot is a subject's normalized likelihood under the three-seed mean GRU minus "
        "that subject's common-Q likelihood. Thin lines connect the same subject across "
        "the five E4 D values and E8 D=614; "
        "the short bar is the median and the hollow diamond is the arithmetic mean. Panel "
        "p-values are unadjusted two-sided paired Wilcoxon signed-rank tests against zero.",
        "",
        "### Cohort summary and trial-pooled likelihood",
        "",
        "| cohort | species | split | subjects | sessions | held-out trials | common Q | GRU D=10 | D=30 | D=100 | D=300 | D=614 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_ORDER:
        audit = validation[dataset_name]
        dataset = matched["datasets"][dataset_name]
        gru = [_mean_sd([_metric(row) for row in _gru_for_d(dataset, d)]) for d in DS]
        lines.append(
            f"| {LABELS[dataset_name]} — {TASKS[dataset_name]} | {audit['species']} | "
            f"v{audit['schema_version']} | {audit['num_subjects']} | "
            f"{audit['num_sessions']} | {audit['num_test_trials']:,} | "
            f"{_metric(dataset['q']):.5f} | " + " | ".join(gru) + " |"
        )

    lines += [
        "",
        "### Paired GRU minus common-Q result",
        "",
        "| cohort | space | D | median Δ likelihood | mean Δ likelihood | subjects GRU better | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_ORDER:
        dataset = matched["datasets"][dataset_name]
        for d in DS:
            values = _subject_differences(dataset, d)
            lines.append(
                f"| {LABELS[dataset_name]} | E4 | {d} | {statistics.median(values):+.5f} | "
                f"{statistics.mean(values):+.5f} | "
                f"{sum(value > 0 for value in values) / len(values):.0%} "
                f"({sum(value > 0 for value in values)}/{len(values)}) | "
                f"{_wilcoxon(values):.3g} |"
            )
        q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
        subjects = sorted(q)
        e8_seeds = [
            row["metrics"]["per_subject_mean_log_likelihood_nats"]
            for row in embedding_dimension["datasets"][dataset_name]["e8"]
        ]
        values = [
            math.exp(statistics.mean(float(seed[subject]) for seed in e8_seeds))
            - math.exp(float(q[subject]))
            for subject in subjects
        ]
        lines.append(
            f"| {LABELS[dataset_name]} | E8 | 614 | {statistics.median(values):+.5f} | "
            f"{statistics.mean(values):+.5f} | "
            f"{sum(value > 0 for value in values) / len(values):.0%} "
            f"({sum(value > 0 for value in values)}/{len(values)}) | "
            f"{_wilcoxon(values):.3g} |"
        )

    lines += [
        "",
        "### Scaling benefit from D=10 to D=614",
        "",
        "The subject-level value is D=614 GRU normalized likelihood minus D=10 GRU "
        "normalized likelihood, after averaging source seeds in log-likelihood space.",
        "",
        "| cohort | trial-pooled Δ | subject median Δ | subject mean Δ | subjects improved | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_ORDER:
        dataset = matched["datasets"][dataset_name]
        d10 = _gru_for_d(dataset, 10)
        d614 = _gru_for_d(dataset, 614)
        pooled = statistics.mean(_metric(row) for row in d614) - statistics.mean(
            _metric(row) for row in d10
        )
        ten = [row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in d10]
        six = [row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in d614]
        subjects = sorted(ten[0])
        if any(set(row) != set(subjects) for row in [*ten, *six]):
            raise AssertionError("D=10/D=614 subject keys do not align")
        values = [
            math.exp(statistics.mean(float(row[key]) for row in six))
            - math.exp(statistics.mean(float(row[key]) for row in ten))
            for key in subjects
        ]
        lines.append(
            f"| {LABELS[dataset_name]} | {pooled:+.5f} | "
            f"{statistics.median(values):+.5f} | {statistics.mean(values):+.5f} | "
            f"{sum(value > 0 for value in values) / len(values):.0%} | "
            f"{_wilcoxon(values):.3g} |"
        )

    lines += [
        "",
        *_stage_a_read(matched),
        "",
        "### Author-aligned baselines",
        "",
        "| cohort | published model | role | common Q | published model refit | E4 GRU D=614 | E8 GRU D=614 |",
        "|---|---|:---:|---:|---:|---:|---:|",
        *author_rows,
        "",
        *_primary_author_read(author_data, matched),
        "",
        "### Subject-level differences from the author-selected model",
        "",
        "The reference is zero. Positive values favor the displayed comparison over the "
        "author-selected model. P-values are unadjusted two-sided paired Wilcoxon tests.",
        "",
        "| cohort | author reference | comparison | median Δ likelihood | mean Δ likelihood | Wilcoxon p |",
        "|---|---|---|---:|---:|---:|",
        *author_subject_rows,
        "",
        "The correlations below relate each subject's author-model normalized likelihood "
        "to that subject's D=614 GRU-minus-author improvement. Negative values mean GRU "
        "benefit is concentrated among subjects fit poorly by the author model.",
        "",
        "| cohort | author reference | subjects | E4 Pearson r | E8 Pearson r |",
        "|---|---|---:|---:|---:|",
        *correlations,
        "",
        "### Why common Q can beat an author-selected model",
        "",
        "This report tests held-out generalization after fitting the same adaptation half; "
        "it does not reproduce each paper's original model-selection objective. Grossman (mouse) "
        "did compare against Q-learning, but our common Q includes forgetting, a one-step "
        "choice kernel, and side bias, while the Grossman (mouse) refit omits the paper's hierarchical "
        "Stan fit and parameter-ordering constraint. Zid (human) selected its model using all 300 "
        "trials and AIC on a smaller analysis cohort, whereas this benchmark fits trials "
        "0–149 and scores 150–299 for all 258 released participants. A ranking reversal here "
        "therefore means that common Q generalizes better under this matched protocol; it is "
        "not evidence that the papers failed to test Q or selected the wrong model for their "
        "own analysis.",
        "",
        "### Representative held-out sessions",
        "",
        "Sessions are selected deterministically at neighboring ranks around the lower "
        "(10th percentile), median, and upper (90th percentile) session-level D=614 "
        "GRU-minus-Q likelihood distribution. For v2, the complete real session is shown "
        "with the adaptation/test boundary; for v1, only a real held-out session is shown. "
        "No pseudo-sessions are constructed.",
        "",
    ]
    for dataset_name in DATASET_ORDER:
        lines += [f"#### {LABELS[dataset_name]}", ""]
        for category, category_label in EXAMPLE_CATEGORIES:
            path = f"../fig_example_sessions_{dataset_name}_{category}.png"
            selected = [
                row
                for row in examples["datasets"][dataset_name]["examples"]
                if row["selection_slug"] == category
            ]
            if selected:
                lines += [
                    f"##### {category_label}",
                    "",
                    f"![{category_label} {LABELS[dataset_name]} held-out sessions]({path})",
                    "",
                ]
        if dataset_name == "tang":
            lines += [
                "Tang (macaque) has only four held-out real sessions in the complete release. All four "
                "are shown once across the three rank regions; sessions are not duplicated to "
                "manufacture nine examples.",
                "",
            ]
    lines += [
        "Black and gray marks denote rewarded and unrewarded choices; the black trace is "
        "the nine-trial smoothed right-choice fraction. The reward-probability strip is "
        "shown only when that schedule is available in the public release. Plots use the "
        f"pinned [`plot_foraging_session`]({examples['plotting']['source_url']}) implementation.",
        "",
        "### Skipped datasets",
        "",
        _survey_section(
            "## Skipped cohorts", "## Stage-B author-model feasibility gate"
        ),
        "",
        "### Author-model feasibility — Stage-B stop gate",
        "",
        _survey_section("## Stage-B author-model feasibility gate"),
        "",
        "### Validation",
        "",
        "- Every admitted release passed pinned-source checksum and exact-count audits.",
        "- Every canonical choice and reward is binary.",
        "- Every manifest is deterministic and uses only schema v1 or v2.",
        "- Every GRU cell and common-Q baseline has identical ordered held-out "
        "`(subject_id, ses_idx, trial, choice)` keys.",
        "- V2 uses the complete first-half prefix with no K condition, then scores the "
        "second-half suffix after state replay.",
        "- Nominal source D is plotted. Realized source-subject counts were "
        + ", ".join(f"D={d}: {actual_ds[d]}" for d in DS)
        + ".",
        "- Every primary-set author model uses the same immutable adaptation and held-out "
        "trials as GRU and common Q; model-specific fitting deviations are disclosed in "
        "the feasibility table.",
    ]
    return "\n".join(lines)


def main() -> None:
    author_data = json.loads(AUTHOR_DATA.read_text())
    matched = json.loads(MATCHED_DATA.read_text())
    examples = json.loads(EXAMPLE_DATA.read_text())
    validation = _validation_map(json.loads(VALIDATION_DATA.read_text()))
    embedding_dimension = json.loads(EMBEDDING_DIMENSION_DATA.read_text())
    embedding_expansion = json.loads(EMBEDDING_DIMENSION_EXPANSION_DATA.read_text())
    task_design = json.loads(TASK_DESIGN_DATA.read_text())
    if tuple(matched["datasets"]) != ALL_DATASET_ORDER:
        raise AssertionError("Frozen matched-result dataset membership drifted")
    if tuple(examples["datasets"]) != ALL_DATASET_ORDER:
        raise AssertionError("Frozen example dataset membership drifted")
    overlap = set(embedding_dimension["datasets"]) & set(
        embedding_expansion["datasets"]
    )
    if overlap:
        raise AssertionError(f"Duplicate E8 datasets across frozen inputs: {sorted(overlap)}")
    embedding_dimension["datasets"].update(embedding_expansion["datasets"])
    if set(embedding_dimension["datasets"]) != set(DATASET_ORDER):
        raise AssertionError("E8 frozen inputs do not cover every displayed dataset")
    _plot_summary(
        author_data,
        matched,
        validation,
        embedding_dimension,
        task_design["analysis_tiers"],
    )
    _plot_author_subjects(author_data, matched, embedding_dimension)
    _plot_gru_q_subjects(matched, validation, embedding_dimension)
    example_paths = _plot_examples(examples)
    block = _result_block(
        author_data,
        matched,
        examples,
        validation,
        embedding_dimension,
    )
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + block + "\n" + text[end_start:])
    print(
        f"Wrote {FIGURE}, {SUBJECT_FIGURE}, {GRU_Q_SUBJECT_FIGURE}, "
        f"{len(example_paths)} example figures, "
        f"and {REPORT}"
    )


if __name__ == "__main__":
    main()
