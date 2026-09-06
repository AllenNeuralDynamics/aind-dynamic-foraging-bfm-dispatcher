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
SURVEY = STUDY / "DATASET_SURVEY.md"
FIGURE = STUDY / "analysis" / "fig_author_baseline_likelihood.png"
SUBJECT_FIGURE = STUDY / "analysis" / "fig_subject_baseline_likelihood.png"
REPORT = STUDY / "analysis" / "reports" / "r2-author-aligned-baselines.md"
START = "<!-- BEGIN result-2 -->"
END = "<!-- END result-2 -->"
DS = (10, 30, 100, 300, 614)
DATASET_ORDER = (
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
LABELS = {
    "grossman": "Grossman",
    "chen": "Chen",
    "zid": "Zid",
    "lebedeva": "Lebedeva",
    "beron": "Beron",
    "kwak": "Kwak",
    "miller": "Miller",
    "findling": "Findling",
    "tang": "Tang",
    "alsio": "Alsiö",
    "eckstein": "Eckstein",
    "costa": "Costa",
    "lopez_mouse": "López-Yépez mouse",
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
}
EXAMPLE_CATEGORIES = (
    ("lower", "Lower tail"),
    ("median", "Median"),
    ("upper", "Upper tail"),
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
    if tuple(rows) != DATASET_ORDER:
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


def _plot_summary(author_data: dict, matched: dict, validation: dict[str, dict]) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(4, 4, figsize=(18, 15), constrained_layout=True)
    records = author_data["records"]
    for axis, dataset_name in zip(axes.flat, DATASET_ORDER):
        dataset = matched["datasets"][dataset_name]
        means = []
        sds = []
        for d in DS:
            values = [_metric(row) for row in _gru_for_d(dataset, d)]
            means.append(statistics.mean(values))
            sds.append(statistics.stdev(values))
            axis.scatter([d] * 3, values, s=18, color="#4C72B0", alpha=0.35)
        axis.errorbar(
            DS,
            means,
            yerr=sds,
            marker="o",
            color="#4C72B0",
            capsize=2,
            linewidth=1.6,
            label="GRU mean ± SD",
        )
        axis.axhline(_metric(dataset["q"]), color="#222222", linestyle="--", label="common Q")
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
            f"{audit['species']} · v{audit['schema_version']} · "
            f"n={audit['num_subjects']}, sessions={audit['num_sessions']}, "
            f"test trials={audit['num_test_trials']:,}"
        )
        axis.set_title(
            f"{LABELS[dataset_name]}\n{textwrap.fill(TASKS[dataset_name], 34)}\n{details}",
            fontsize=9,
        )
        axis.set_xscale("log")
        axis.set_xticks(DS, [str(value) for value in DS])
        axis.set_xlabel("Source subjects D")
        axis.set_ylabel("Held-out normalized likelihood")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=6.5, loc="best")
    for axis in axes.flat[len(DATASET_ORDER) :]:
        axis.set_visible(False)
    fig.suptitle(
        "Frozen-core GRU transfer versus matched common Q\n"
        "Every panel uses the complete admitted cohort and identical held-out trials",
        fontsize=17,
    )
    fig.savefig(FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_subjects(matched: dict, validation: dict[str, dict]) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), constrained_layout=True)
    rng = np.random.default_rng(29)
    for axis, dataset_name in zip(axes.flat, DATASET_ORDER):
        dataset = matched["datasets"][dataset_name]
        values = [_subject_differences(dataset, d) for d in DS]
        positions = np.arange(len(DS))
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
        for body in violins["bodies"]:
            body.set_facecolor("#4C72B0")
            body.set_edgecolor("#4C72B0")
            body.set_alpha(0.22)
        violins["cmedians"].set_color("#17365D")
        violins["cmedians"].set_linewidth(1.5)
        for position, condition in zip(positions, values):
            jitter = rng.uniform(-0.15, 0.15, len(condition))
            axis.scatter(
                position + jitter,
                condition,
                s=6,
                color="#4C72B0",
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
                edgecolor="#17365D",
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
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.5},
            )
        audit = validation[dataset_name]
        axis.axhline(0, color="#222222", linewidth=1)
        axis.set_xticks(positions, [str(value) for value in DS])
        axis.set_title(
            f"{LABELS[dataset_name]} · {audit['species']} · "
            f"v{audit['schema_version']} · n={n_subjects}",
            fontsize=9,
        )
        axis.set_xlabel("Source subjects D")
        axis.set_ylabel("Subject GRU − Q likelihood")
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(DATASET_ORDER) :]:
        axis.set_visible(False)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#17365D", linewidth=1.5, label="median"),
            Line2D(
                [0],
                [0],
                marker="D",
                markerfacecolor="white",
                markeredgecolor="#17365D",
                color="none",
                label="mean",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Paired subject-level GRU improvement over common Q\n"
        "Dots are subjects; thin lines connect the same subject across D",
        fontsize=17,
    )
    fig.savefig(SUBJECT_FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_examples(example_data: dict) -> list[Path]:
    module = importlib.import_module(
        "aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session"
    )
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != example_data[
        "plotting"
    ]["source_sha256"]:
        raise AssertionError("Installed plot_foraging_session.py differs from frozen source")
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
                        axis.axvline(boundary + 0.5, color="#7B3294", linestyle="--", lw=1)
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
            path = STUDY / "analysis" / f"fig_example_sessions_{dataset_name}_{category}.png"
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


def _author_rows(author_data: dict, matched: dict) -> tuple[list[str], list[str]]:
    rows = []
    correlations = []
    for baseline, record in author_data["records"].items():
        dataset_name = record["dataset"]
        dataset = matched["datasets"][dataset_name]
        d614 = _gru_for_d(dataset, 614)
        rows.append(
            f"| {LABELS[dataset_name]} | {AUTHOR_LABELS[baseline]} | "
            f"{'yes' if record['author_selected'] else 'paper comparator'} | "
            f"{_metric(dataset['q']):.5f} | {_metric(record):.5f} | "
            f"{_mean_sd([_metric(row) for row in d614])} |"
        )
        if record["author_selected"]:
            author = record["metrics"]["per_subject_mean_log_likelihood_nats"]
            seeds = [
                row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in d614
            ]
            if any(set(seed) != set(author) for seed in seeds):
                raise AssertionError("Author/GRU subject keys do not align")
            reference = [math.exp(float(author[key])) for key in sorted(author)]
            improvement = [
                math.exp(statistics.mean(float(seed[key]) for seed in seeds))
                - math.exp(float(author[key]))
                for key in sorted(author)
            ]
            correlation = float(np.corrcoef(reference, improvement)[0, 1])
            correlations.append(
                f"| {LABELS[dataset_name]} | {AUTHOR_LABELS[baseline]} | "
                f"{len(reference)} | {correlation:+.2f} |"
            )
    return rows, correlations


def _result_block(
    author_data: dict,
    matched: dict,
    examples: dict,
    validation: dict[str, dict],
) -> str:
    author_rows, correlations = _author_rows(author_data, matched)
    lines = [
        "[regenerated by `analysis/report_author_baselines.py` — do not edit by hand]",
        "",
        "## Stage-A decision result",
        "",
        "![GRU, common Q, and available author baselines](../fig_author_baseline_likelihood.png)",
        "",
        "Every model uses the same immutable adaptation and held-out observations. "
        "GRU points are the three source-training seeds; the curve is their mean ± SD. "
        "Common Q is fitted independently per target subject on the identical adaptation half. "
        "Existing author-model lines are retained for Grossman, Chen, and Zid, but no new "
        "author-selected model was implemented in Stage A.",
        "",
        "![Paired subject-level GRU minus common-Q likelihood](../fig_subject_baseline_likelihood.png)",
        "",
        "Each dot is a subject's normalized likelihood under the three-seed mean GRU minus "
        "that subject's common-Q likelihood. Thin lines connect the same subject across D; "
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
        gru = [
            _mean_sd([_metric(row) for row in _gru_for_d(dataset, d)]) for d in DS
        ]
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
        "| cohort | D | median Δ likelihood | mean Δ likelihood | subjects GRU better | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name in DATASET_ORDER:
        dataset = matched["datasets"][dataset_name]
        for d in DS:
            values = _subject_differences(dataset, d)
            lines.append(
                f"| {LABELS[dataset_name]} | {d} | {statistics.median(values):+.5f} | "
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
        ten = [
            row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in d10
        ]
        six = [
            row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in d614
        ]
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
        "### Existing author-aligned baselines",
        "",
        "| cohort | published model | author-selected? | common Q | published model refit | GRU D=614 |",
        "|---|---|:---:|---:|---:|---:|",
        *author_rows,
        "",
        "The correlations below relate each subject's author-model normalized likelihood "
        "to that subject's D=614 GRU-minus-author improvement. Negative values mean GRU "
        "benefit is concentrated among subjects fit poorly by the author model.",
        "",
        "| cohort | author reference | subjects | Pearson r |",
        "|---|---|---:|---:|",
        *correlations,
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
                "Tang has only four held-out real sessions in the complete release. All four "
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
        _survey_section("## Skipped cohorts", "## Stage-B author-model feasibility gate"),
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
        "- New author-selected models remain outside Stage A.",
    ]
    return "\n".join(lines)


def main() -> None:
    author_data = json.loads(AUTHOR_DATA.read_text())
    matched = json.loads(MATCHED_DATA.read_text())
    examples = json.loads(EXAMPLE_DATA.read_text())
    validation = _validation_map(json.loads(VALIDATION_DATA.read_text()))
    if tuple(matched["datasets"]) != DATASET_ORDER:
        raise AssertionError("Frozen matched-result dataset membership drifted")
    if tuple(examples["datasets"]) != DATASET_ORDER:
        raise AssertionError("Frozen example dataset membership drifted")
    _plot_summary(author_data, matched, validation)
    _plot_subjects(matched, validation)
    example_paths = _plot_examples(examples)
    block = _result_block(author_data, matched, examples, validation)
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + block + "\n" + text[end_start:])
    print(
        f"Wrote {FIGURE}, {SUBJECT_FIGURE}, {len(example_paths)} example figures, "
        f"and {REPORT}"
    )


if __name__ == "__main__":
    main()
