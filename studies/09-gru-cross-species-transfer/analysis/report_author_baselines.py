"""Render the consolidated offline external-transfer comparison."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import apply_presentation_style, t975  # noqa: E402


AUTHOR_DATA = STUDY / "analysis" / "author_baseline_results.json"
MATCHED_DATA = STUDY / "analysis" / "matched_half_results.json"
EXAMPLE_DATA = STUDY / "analysis" / "example_behavior_sessions.json"
FIGURE = STUDY / "analysis" / "fig_author_baseline_likelihood.png"
SUBJECT_FIGURE = STUDY / "analysis" / "fig_subject_baseline_likelihood.png"
EXAMPLE_CATEGORIES = (
    ("lower", "Lower tail"),
    ("median", "Median"),
    ("upper", "Upper tail"),
)
EXAMPLE_FIGURES = {
    (dataset_name, category): STUDY
    / "analysis"
    / f"fig_example_sessions_{dataset_name}_{category}.png"
    for dataset_name in ("grossman", "chen", "zid")
    for category, _ in EXAMPLE_CATEGORIES
}
REPORT = STUDY / "analysis" / "reports" / "r2-author-aligned-baselines.md"
START = "<!-- BEGIN result-2 -->"
END = "<!-- END result-2 -->"
DS = (10, 30, 100, 300, 614)
LABELS = {
    "grossman": "Grossman mouse",
    "chen": "Chen mouse",
    "zid": "Zid human",
}
TASK_DETAILS = {
    "grossman": "Blockwise reversal bandit",
    "chen": "Restless random-walk bandit",
    "zid": "Restless random-walk bandit",
}
BASELINE_LABELS = {
    "grossman-meta-learning": "meta-learning RL",
    "chen-rlck": "4-parameter RLCK",
    "zid-traditional-rlck": "traditional RLCK",
    "zid-history-kernel-foraging": "HK2 foraging RL",
}
PARAM_COUNTS = {
    "grossman-meta-learning": 7,
    "chen-rlck": 4,
    "zid-traditional-rlck": 4,
    "zid-history-kernel-foraging": 5,
}


def _metric(record: dict) -> float:
    return float(record["metrics"]["normalized_likelihood"])


def _gru_for_d(dataset: dict, d: int) -> list[dict]:
    rows = [row for row in dataset["gru"] if int(row["nominal_D"]) == d]
    if len(rows) != 3:
        raise AssertionError(f"Expected exactly three D={d} GRU source seeds")
    return rows


def _gru_d614(dataset: dict) -> list[dict]:
    return _gru_for_d(dataset, 614)


def _paired(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "low": mean - t975(len(values)) * sem,
        "high": mean + t975(len(values)) * sem,
        "fraction_positive": sum(value > 0 for value in values) / len(values),
        "n": len(values),
    }


def _paired_comparisons(record: dict, dataset: dict) -> tuple[dict, dict]:
    author = record["metrics"]["per_subject_mean_log_likelihood_nats"]
    q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
    gru = [
        row["metrics"]["per_subject_mean_log_likelihood_nats"]
        for row in _gru_d614(dataset)
    ]
    if set(author) != set(q) or any(set(seed) != set(author) for seed in gru):
        raise AssertionError("Author, Q, and GRU per-subject metric sets do not align")
    author_minus_q = [float(author[key]) - float(q[key]) for key in author]
    gru_minus_author = [
        statistics.mean(float(seed[key]) for seed in gru) - float(author[key])
        for key in author
    ]
    return _paired(author_minus_q), _paired(gru_minus_author)


def _gru_q_comparison(dataset: dict, d: int) -> dict:
    q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
    gru = [
        row["metrics"]["per_subject_mean_log_likelihood_nats"]
        for row in _gru_for_d(dataset, d)
    ]
    if any(set(seed) != set(q) for seed in gru):
        raise AssertionError("Q and GRU per-subject metric sets do not align")
    differences = [
        statistics.mean(float(seed[key]) for seed in gru) - float(q[key]) for key in q
    ]
    return _paired(differences)


def _subject_conditions(
    dataset_name: str, author_data: dict, dataset: dict
) -> tuple[str, list[str], list[list[float]], list[str], list[float], float]:
    records = [
        (key, record)
        for key, record in author_data["records"].items()
        if record["dataset"] == dataset_name
    ]
    selected = [(key, record) for key, record in records if record["author_selected"]]
    if len(selected) != 1:
        raise AssertionError("Expected exactly one author-selected model per dataset")
    selected_baseline, selected_record = selected[0]
    records = [
        (key, record) for key, record in records if not record["author_selected"]
    ]
    records.sort(key=lambda item: item[0])
    q = dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
    subjects = sorted(q)
    reference = selected_record["metrics"]["per_subject_mean_log_likelihood_nats"]
    if set(reference) != set(q):
        raise AssertionError("Author and Q per-subject metric sets do not align")
    labels = ["Common Q"]
    log_values = [[float(q[subject]) for subject in subjects]]
    colors = ["#666666"]
    for baseline, record in records:
        values = record["metrics"]["per_subject_mean_log_likelihood_nats"]
        if set(values) != set(q):
            raise AssertionError("Author and Q per-subject metric sets do not align")
        labels.append(BASELINE_LABELS[baseline])
        log_values.append([float(values[subject]) for subject in subjects])
        colors.append("#DD8452")
    for d in DS:
        seeds = [
            row["metrics"]["per_subject_mean_log_likelihood_nats"]
            for row in _gru_for_d(dataset, d)
        ]
        labels.append(f"GRU D={d}")
        log_values.append(
            [
                statistics.mean(float(seed[subject]) for seed in seeds)
                for subject in subjects
            ]
        )
        colors.append("#4C72B0")
    reference_likelihood = [math.exp(float(reference[subject])) for subject in subjects]
    differences = [
        [
            math.exp(value) - reference_value
            for value, reference_value in zip(values, reference_likelihood)
        ]
        for values in log_values
    ]
    p_values = [
        float(wilcoxon(values, alternative="two-sided").pvalue)
        for values in differences
    ]
    if labels[-1] != "GRU D=614":
        raise AssertionError("Expected D=614 GRU to be the final subject condition")
    correlation = float(np.corrcoef(reference_likelihood, differences[-1])[0, 1])
    return (
        BASELINE_LABELS[selected_baseline],
        labels,
        differences,
        colors,
        p_values,
        correlation,
    )


def _plot(author_data: dict, matched: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    records = author_data["records"]
    for axis, dataset_name in zip(axes, ("grossman", "chen", "zid")):
        dataset = matched["datasets"][dataset_name]
        baselines = [
            (key, record)
            for key, record in records.items()
            if record["dataset"] == dataset_name
        ]
        means, sds = [], []
        for d in DS:
            values = [
                _metric(row) for row in dataset["gru"] if int(row["nominal_D"]) == d
            ]
            if len(values) != 3:
                raise AssertionError(f"Expected three GRU seeds for D={d}")
            means.append(statistics.mean(values))
            sds.append(statistics.stdev(values))
            axis.scatter([d] * len(values), values, s=28, color="#4C72B0", alpha=0.4)
        axis.errorbar(
            DS,
            means,
            yerr=sds,
            marker="o",
            color="#4C72B0",
            capsize=3,
            label="GRU mean ± SD",
            zorder=4,
        )
        axis.axhline(
            _metric(dataset["q"]),
            color="#333333",
            linestyle="--",
            label="common Q",
        )
        for baseline, record in baselines:
            selected = bool(record["author_selected"])
            suffix = "author-selected" if selected else "paper comparator"
            axis.axhline(
                _metric(record),
                color="#C44E52" if selected else "#DD8452",
                linestyle="-" if selected else ":",
                linewidth=2.2 if selected else 1.7,
                label=f"{BASELINE_LABELS[baseline]} ({suffix})",
            )
        axis.set_xscale("log")
        axis.set_xticks(DS, [str(d) for d in DS])
        n_subjects = len(
            dataset["q"]["metrics"]["per_subject_mean_log_likelihood_nats"]
        )
        axis.set_title(
            f"{LABELS[dataset_name]}\n{TASK_DETAILS[dataset_name]} · n_subject={n_subjects}"
        )
        axis.set_xlabel("Source subjects D")
        axis.set_ylabel("Held-out normalized likelihood")
        axis.grid(axis="y", alpha=0.2)
        legend_location = "upper left" if dataset_name == "zid" else "best"
        axis.legend(frameon=False, fontsize=8.5, loc=legend_location)
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def _plot_subjects(author_data: dict, matched: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), constrained_layout=True)
    for axis, dataset_name in zip(axes, ("grossman", "chen", "zid")):
        dataset = matched["datasets"][dataset_name]
        reference_label, labels, values, colors, p_values, correlation = (
            _subject_conditions(dataset_name, author_data, dataset)
        )
        positions = list(range(len(labels)))
        n_subjects = len(values[0])
        jitter = [((index % 17) - 8) / 80 for index in range(n_subjects)]
        for subject_index in range(n_subjects):
            xs = [position + jitter[subject_index] for position in positions]
            ys = [condition[subject_index] for condition in values]
            axis.plot(xs, ys, color="#777777", alpha=0.08, linewidth=0.45, zorder=1)
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
        violins["cmedians"].set_linewidth(1.8)
        for position, condition, color in zip(positions, values, colors):
            axis.scatter(
                [position + offset for offset in jitter],
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
        tick_labels = [
            label.replace("4-parameter ", "4-param\n")
            .replace("traditional ", "traditional\n")
            .replace("GRU ", "GRU\n")
            for label in labels
        ]
        axis.set_xticks(positions, tick_labels)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_title(
            f"{LABELS[dataset_name]} (n_subject={n_subjects})\n"
            f"{TASK_DETAILS[dataset_name]}\nReference: {reference_label}"
            f"\nD=614 corr(author likelihood, GRU Δ): r={correlation:+.2f}"
        )
        axis.axhline(0, color="#C44E52", linewidth=1.6, alpha=0.8, zorder=0)
        axis.set_yscale("symlog", linthresh=0.01)
        axis.set_ylabel("Δ subject held-out\nnormalized likelihood")
        axis.grid(axis="y", alpha=0.2)
        for position, p_value in zip(positions, p_values):
            axis.text(
                position,
                0.985,
                _p_axis_label(p_value),
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
    fig.savefig(SUBJECT_FIGURE, bbox_inches="tight")
    plt.close(fig)


def _plot_examples(example_data: dict) -> None:
    module = importlib.import_module(
        "aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session"
    )
    source_digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if source_digest != example_data["plotting"]["source_sha256"]:
        raise AssertionError(
            "Installed plot_foraging_session.py does not match the frozen source digest"
        )
    plot_foraging_session = module.plot_foraging_session
    apply_presentation_style()
    for dataset_name in ("grossman", "chen", "zid"):
        all_examples = example_data["datasets"][dataset_name]["examples"]
        for category, category_label in EXAMPLE_CATEGORIES:
            examples = [
                example
                for example in all_examples
                if example["selection_slug"] == category
            ]
            if len(examples) != 3:
                raise AssertionError(
                    f"Expected three {dataset_name} / {category} examples"
                )
            plt.close("all")
            fig, holders = plt.subplots(3, 1, figsize=(15, 8.6), dpi=160)
            legend_handles = None
            legend_labels = None
            for holder, example in zip(holders, examples):
                _, axes = plot_foraging_session(
                    choice_history=np.asarray(example["choice"], dtype=float),
                    reward_history=np.asarray(example["reward"], dtype=bool),
                    p_reward=np.asarray(
                        [
                            example["reward_probability_arm_0"],
                            example["reward_probability_arm_1"],
                        ],
                        dtype=float,
                    ),
                    smooth_factor=9,
                    ax=holder,
                    vertical=False,
                    plot_list=["choice", "reward_prob"],
                )
                if legend_handles is None:
                    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
                axes[0].get_legend().remove()
                delta = example["gru_d614_minus_author_normalized_likelihood"]
                axes[0].set_title(
                    f"{example['subject_id']} · {example['session_id']} · "
                    f"Δ={delta:+.3f}",
                    fontsize=10,
                    loc="left",
                )
                boundary = example["adapt_prefix_trials"]
                if boundary is not None:
                    for axis in axes:
                        axis.axvline(
                            boundary + 0.5,
                            color="#7B3294",
                            linestyle="--",
                            lw=1.2,
                        )
                    axes[0].text(
                        boundary + 2,
                        1.19,
                        "held-out suffix",
                        color="#7B3294",
                        fontsize=8,
                        va="top",
                    )
            fig.legend(
                legend_handles,
                legend_labels,
                fontsize=7,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=5,
                frameon=True,
            )
            fig.suptitle(
                f"{LABELS[dataset_name]}: {category_label} examples",
                fontsize=15,
                y=0.995,
            )
            fig.subplots_adjust(
                left=0.1,
                right=0.98,
                bottom=0.06,
                top=0.88,
                hspace=0.55,
            )
            fig.savefig(EXAMPLE_FIGURES[(dataset_name, category)])
            plt.close(fig)


def _interval(stats: dict) -> str:
    return f"{stats['mean']:+.5f} [{stats['low']:+.5f}, {stats['high']:+.5f}]"


def _p_value(value: float) -> str:
    return f"{value:.2g}"


def _p_axis_label(value: float) -> str:
    return "p<.001" if value < 0.001 else f"p={value:.3f}".replace("0.", ".")


def _result_block(author_data: dict, matched: dict, example_data: dict) -> str:
    actual_ds = {
        d: sorted(
            {
                row["actual_D"]
                for dataset in matched["datasets"].values()
                for row in dataset["gru"]
                if int(row["nominal_D"]) == d
            }
        )
        for d in DS
    }
    lines = [
        "[regenerated by `analysis/report_author_baselines.py` — do not edit by hand]",
        "",
        "![Author-aligned baselines versus common Q and transferred GRU](../fig_author_baseline_likelihood.png)",
        "",
        "All models use the same subject-level adaptation/test split and score the exact same held-out trials. "
        "The GRU curve and table report mean ± SD across three source-training seeds at every D.",
        "",
        "![Subject-level held-out likelihood distributions with paired trajectories](../fig_subject_baseline_likelihood.png)",
        "",
        "Each value is that model's subject-level normalized likelihood minus the same subject's "
        "author-selected-model likelihood. Thus the red zero line is the author-model reference; "
        "positive values favor the displayed model. Dots are subjects, thin lines connect each subject "
        "across models, violins show the distributions, the short horizontal bar is the median, and "
        "the hollow diamond is the arithmetic mean. GRU subject log likelihood is averaged "
        "across the three source seeds before conversion to normalized likelihood. Panel annotations report "
        "unadjusted two-sided paired Wilcoxon signed-rank p-values versus the author model. The "
        "panel title also reports the D=614 Pearson correlation between author-model likelihood and "
        "GRU-minus-author improvement. The "
        "symmetric-log y-axis is linear within ±0.01 and retains the large Zid outliers while resolving "
        "the central distribution.",
        "The common Q fits five parameters: one reward learning rate, unchosen-value forgetting, "
        "one-step choice-kernel weight, side bias, and softmax inverse temperature. The "
        "`ForagerQLearning` parameter generator always adds `biasL`; the one-step kernel's step size "
        "is fixed at 1 and is not counted as a fitted parameter.",
        "",
        "### Representative behavior sessions",
        "",
        "These examples were selected deterministically, not by visual inspection: each category "
        "contains the three subjects at neighboring ranks around the 10th, 50th, or 90th "
        "percentile of the subject-level D=614 GRU minus author-selected-model "
        "normalized-likelihood difference. "
        "Negative values favor the author model; positive values favor the GRU.",
        "",
    ]
    for dataset_name in ("grossman", "chen", "zid"):
        lines += [f"#### {LABELS[dataset_name]}", ""]
        for category, category_label in EXAMPLE_CATEGORIES:
            lines += [
                f"##### {category_label}",
                "",
                f"![{category_label} {LABELS[dataset_name]} behavior sessions]"
                f"(../{EXAMPLE_FIGURES[(dataset_name, category)].name})",
                "",
            ]
        if dataset_name == "zid":
            lines.append(
                "The full 300-trial session is shown; the purple dashed line separates the "
                "150-trial adaptation prefix from the held-out suffix."
            )
        else:
            lines.append(
                "For each selected subject, the plot shows the first chronologically held-out "
                "session from the frozen odd/even session split."
            )
        lines += [
            "Black and gray choice marks denote rewarded and unrewarded trials, the black line "
            "is the nine-trial smoothed right-choice fraction, and the lower strip shows the "
            "left/right reward probabilities.",
            "",
        ]
    lines += [
        "Plots use the pinned [`plot_foraging_session`]("
        + example_data["plotting"]["source_url"]
        + ") implementation from `aind-dynamic-foraging-basic-analysis`.",
        "",
        "### Subject-level likelihood differences from the author model",
        "",
        "The median uses normalized-likelihood differences shown in the figure. P-values are "
        "unadjusted two-sided paired Wilcoxon signed-rank tests against zero.",
        "",
        "| target | author reference | comparison | median Δ likelihood | Wilcoxon p |",
        "|---|---|---|---:|---:|",
    ]
    correlations = []
    for dataset_name in ("grossman", "chen", "zid"):
        dataset = matched["datasets"][dataset_name]
        reference_label, labels, values, _, p_values, correlation = (
            _subject_conditions(dataset_name, author_data, dataset)
        )
        correlations.append((dataset_name, reference_label, correlation, len(values[0])))
        for label, differences, p_value in zip(labels, values, p_values):
            lines.append(
                f"| {LABELS[dataset_name]} | {reference_label} | {label} | "
                f"{statistics.median(differences):+.5f} | {_p_value(p_value)} |"
            )
    lines += [
        "",
        "### Does GRU improvement depend on author-model fit?",
        "",
        "Pearson r relates each subject's author-model normalized likelihood to that subject's "
        "D=614 GRU-minus-author normalized-likelihood difference. A negative value means the GRU "
        "tends to help subjects that the author-selected model fits poorly. This association is "
        "descriptive and is not an independent model-comparison test.",
        "",
        "| target | author reference | n subjects | Pearson r |",
        "|---|---|---:|---:|",
    ]
    for dataset_name, reference_label, correlation, n_subjects in correlations:
        lines.append(
            f"| {LABELS[dataset_name]} | {reference_label} | {n_subjects} | "
            f"{correlation:+.2f} |"
        )
    lines += [
        "",
        "### Trial-pooled held-out likelihood",
        "",
        "| target | published baseline | selected? | params | common Q | baseline | GRU D=10 | D=30 | D=100 | D=300 | D=614 |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    comparisons = []
    for baseline, record in author_data["records"].items():
        dataset = matched["datasets"][record["dataset"]]
        gru_cells = []
        for d in DS:
            gru_values = [_metric(row) for row in _gru_for_d(dataset, d)]
            gru_cells.append(
                f"{statistics.mean(gru_values):.5f} ± {statistics.stdev(gru_values):.5f}"
            )
        lines.append(
            f"| {LABELS[record['dataset']]} | {BASELINE_LABELS[baseline]} | "
            f"{'yes' if record['author_selected'] else 'no — paper comparator'} | "
            f"{PARAM_COUNTS[baseline]} | {_metric(dataset['q']):.5f} | "
            f"**{_metric(record):.5f}** | " + " | ".join(gru_cells) + " |"
        )
        comparisons.append((baseline, record, *_paired_comparisons(record, dataset)))

    lines += [
        "",
        "### Subject-balanced paired differences",
        "",
        "Values are mean log-likelihood differences in nats/trial with 95% confidence intervals. "
        "Positive favors the model named first.",
        "",
        "| target | published baseline | author − common Q | subjects author better | GRU D=614 − author | subjects GRU better |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for baseline, record, author_q, gru_author in comparisons:
        lines.append(
            f"| {LABELS[record['dataset']]} | {BASELINE_LABELS[baseline]} | "
            f"{_interval(author_q)} | {author_q['fraction_positive']:.0%} ({author_q['n']}) | "
            f"{_interval(gru_author)} | {gru_author['fraction_positive']:.0%} ({gru_author['n']}) |"
        )

    lines += [
        "",
        "### GRU versus common Q across source D",
        "",
        "Each value is the subject-balanced GRU minus common-Q mean log likelihood in nats/trial, "
        "after averaging the GRU value across source seeds. Positive favors GRU.",
        "",
        "| target | source D | GRU − common Q (95% CI) | subjects GRU better |",
        "|---|---:|---:|---:|",
    ]
    for dataset_name in ("grossman", "chen", "zid"):
        dataset = matched["datasets"][dataset_name]
        for d in DS:
            stats = _gru_q_comparison(dataset, d)
            lines.append(
                f"| {LABELS[dataset_name]} | {d} | {_interval(stats)} | "
                f"{stats['fraction_positive']:.0%} ({stats['n']}) |"
            )

    selected = {
        record["dataset"]: (baseline, record)
        for baseline, record in author_data["records"].items()
        if record["author_selected"]
    }
    lines += ["", "### Bottom line", ""]
    for dataset_name in ("grossman", "chen", "zid"):
        baseline, record = selected[dataset_name]
        dataset = matched["datasets"][dataset_name]
        gru_mean = statistics.mean(_metric(row) for row in _gru_d614(dataset))
        lines.append(
            f"- **{LABELS[dataset_name]}:** {BASELINE_LABELS[baseline]} is "
            f"{_metric(record) - _metric(dataset['q']):+.5f} versus common Q; "
            f"D=614 GRU is {gru_mean - _metric(record):+.5f} versus that published baseline."
        )
    lines += [
        "",
        "### Reproduction confidence",
        "",
        "These ratings concern whether our implementation reproduces the authors' model dynamics, "
        "not uncertainty in the measured likelihood. They do not claim reproduction of the papers' "
        "reported fit values because our adaptation/test split is intentionally different.",
        "",
        "| baseline | confidence | evidence | remaining difference from the paper |",
        "|---|:---:|---|---|",
        "| [Grossman meta-learning RL](https://pmc.ncbi.nlm.nih.gov/articles/PMC8825708/) | **Moderate** | Published value, expected/unexpected-uncertainty, asymmetric learning-rate, forgetting, bias, and softmax equations are implemented and covered by a hand-calculated trajectory test. | The paper did not release model code and used hierarchical session-level Stan fits with mouse-level hyperparameters and the constraint `negative-rate integration > expected-uncertainty rate`. We instead fit one parameter vector per subject by differential evolution on the adaptation sessions, without that ordering constraint. This is an equation-faithful model-family benchmark, not a reproduction of the paper's Bayesian fitting pipeline. |",
        "| [Chen 4-parameter RLCK](https://elifesciences.org/articles/69748) | **High** | The published `alpha`, value inverse temperature, choice-kernel learning rate, and independent kernel inverse temperature map directly to our implementation; zero initialization, chosen-value update, full choice-kernel update, and policy are covered by a hand-calculated trajectory test. | We changed the fitting data and optimizer to the common matched-half protocol and have not reproduced the paper's fitted parameters or model-agreement figure from author outputs. |",
        "| [Zid traditional RLCK](https://www.nature.com/articles/s41467-026-75773-4) (Eq. 19) | **Very high** | Initialization, value and choice-kernel updates, two inverse temperatures, policy, and bounds were checked directly against the authors' released [`model_RLchoice.m`](https://github.com/Mariemzd/HumansForageFoRwd_paper/blob/v1.0.0/modelling_matlab/model_RLchoice.m), in addition to the equation-level test. | The authors fit all 300 main trials with 20 random-start `fminsearch` fits; we fit trials 0-149 with differential evolution and score trials 150-299. |",
        "| [Zid HK2 foraging RL](https://www.nature.com/articles/s41467-026-75773-4) (Eq. 22) | **Very high** | Exploitation value initialization at 1, reset-to-threshold only after a switch, state-history kernel, and stay policy were checked directly against the authors' released [`model_ForagingFlex.m`](https://github.com/Mariemzd/HumansForageFoRwd_paper/blob/v1.0.0/modelling_matlab/model_ForagingFlex.m) and covered by a hand-calculated trajectory test. | The same intentional matched-half and optimizer differences apply. |",
        "",
        "Overall, confidence is high for the Chen and Zid model equations and state transitions. "
        "Confidence is only moderate for Grossman because the published Bayesian hierarchy and "
        "parameter-ordering constraint are not part of this matched subject-level baseline. "
        "Accordingly, the Grossman result should be described as a reimplementation of the selected "
        "model family, not an exact reproduction of the authors' full analysis.",
        "",
        "### Why common Q can beat an author-selected model here",
        "",
        "This benchmark asks which fitted model generalizes from the adaptation half to held-out data. "
        "It does **not** reproduce each paper's original model-selection objective, so a ranking reversal "
        "is not evidence that the paper's conclusion was wrong.",
        "",
        "- **[Grossman did test Q-learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC8825708/).** "
        "The paper compared meta-learning with a static-learning "
        "Q model and favored meta-learning under hierarchical session-level Stan fits and symmetric "
        "two-fold cross-validation. Our common Q is a different sticky, forgetful, side-biased model; "
        "our meta-learning fit is one vector per subject, omits the paper's hierarchy and ordering "
        "constraint, and uses only the fixed odd-session-to-even-session direction. The small common-Q "
        "trial-pooled advantage here is only 0.00201. The paired author-minus-Q difference is "
        "-0.00245 nats/trial with a 95% CI of [-0.00442, -0.00048], so the direction is fairly "
        "consistent but small. It is more plausibly a pipeline/model-specification difference than "
        "evidence that the authors never tested Q.",
        "- **[Zid tested many traditional RL extensions](https://www.nature.com/articles/s41467-026-75773-4), "
        "but not necessarily our exact combination.** "
        "The paper selected HK2 foraging RL using fits to all 300 trials and AIC, after excluding four "
        "non-switching participants from many analyses. We combine unchosen-value decay, a one-step "
        "choice kernel, and side bias, fit only trials 0–149, score 150–299, and retain all 258 subjects. "
        "The released HK2 state dynamics match our implementation closely. Although common Q is "
        "0.02121 higher in the trial-pooled score, the paired author-minus-Q interval "
        "[-0.07337, +0.01222] crosses zero and only 43% of subjects favor HK2. This is a "
        "heterogeneous result, more consistent with a different generalization target and fit budget "
        "than with an obvious equation bug.",
        "",
        "The conservative claim is therefore: **common Q generalizes better than these author-model "
        "refits under our matched held-out protocol in Grossman and Zid**. It is not a paper-level "
        "reproduction claim. Stronger attribution would require reproducing Grossman's hierarchy and "
        "both fold directions, plus Zid's full-data AIC ranking and 254-subject analysis subset.",
        "",
        "### Verification",
        "",
        "- Each published baseline was fitted independently per subject on the adaptation half; no paper-reported likelihood was copied.",
        "- Published-baseline, common-Q, and GRU prediction files have identical ordered `(subject, session, trial, choice)` keys within each target dataset.",
        "- Grossman and Chen use odd-positioned sessions for adaptation and even-positioned sessions for test. Zid adapts on trials 0–149 and scores trials 150–299 after state-only prefix replay.",
        "- The Grossman, Chen, and selected Zid implementations follow the authors' published update equations; Zid traditional RLCK is retained as the paper's simpler comparator.",
        "- Nominal source D is plotted. Realized source-subject counts were "
        + ", ".join(f"D={d}: {actual_ds[d]}" for d in DS)
        + ".",
        "- Common Q and each author model have one fitted baseline per target dataset and therefore "
        "no source-training-seed error bar.",
    ]
    return "\n".join(lines)


def main() -> None:
    author_data = json.loads(AUTHOR_DATA.read_text())
    matched = json.loads(MATCHED_DATA.read_text())
    example_data = json.loads(EXAMPLE_DATA.read_text())
    _plot(author_data, matched)
    _plot_subjects(author_data, matched)
    _plot_examples(example_data)
    block = _result_block(author_data, matched, example_data)
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + block + "\n" + text[end_start:])
    figure_paths = ", ".join(str(path) for path in EXAMPLE_FIGURES.values())
    print(f"Wrote {FIGURE}, {SUBJECT_FIGURE}, {figure_paths}, and {REPORT}")


if __name__ == "__main__":
    main()
