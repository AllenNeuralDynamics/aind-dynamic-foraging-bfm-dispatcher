"""Render Result 4 from the frozen paired E=4/E=8 transfer artifact."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import apply_presentation_style  # noqa: E402


DATA = STUDY / "analysis" / "embedding_dimension_results.json"
SUMMARY_FIGURE = STUDY / "analysis" / "fig_embedding_dimension_transfer.png"
SUBJECT_FIGURE = STUDY / "analysis" / "fig_subject_embedding_dimension_delta.png"
REPORT = STUDY / "analysis" / "reports" / "r4-embedding-dimension-transfer.md"
START = "<!-- BEGIN result-4 -->"
END = "<!-- END result-4 -->"
DATASET_ORDER = ("grossman", "lebedeva", "miller", "findling", "eckstein")
LABELS = {
    "grossman": "Grossman (mouse)",
    "lebedeva": "Lebedeva (mouse)",
    "miller": "Miller (rat)",
    "findling": "Findling (human)",
    "eckstein": "Eckstein (human)",
}


def _metric(row: dict) -> float:
    return float(row["metrics"]["normalized_likelihood"])


def _dimension_rows(dataset: dict, dimension: str) -> list[dict]:
    rows = sorted(dataset[dimension], key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != [0, 1, 2]:
        raise AssertionError(f"{dimension} must contain source seeds 0, 1, and 2")
    return rows


def _subject_deltas(dataset: dict) -> list[float]:
    e4 = [
        row["metrics"]["per_subject_mean_log_likelihood_nats"]
        for row in _dimension_rows(dataset, "e4")
    ]
    e8 = [
        row["metrics"]["per_subject_mean_log_likelihood_nats"]
        for row in _dimension_rows(dataset, "e8")
    ]
    keys = set(e4[0])
    if any(set(row) != keys for row in e4 + e8):
        raise AssertionError("E=4 and E=8 subject keys do not align")
    return [
        math.exp(statistics.mean(float(row[key]) for row in e8))
        - math.exp(statistics.mean(float(row[key]) for row in e4))
        for key in sorted(keys)
    ]


def _p(values: list[float]) -> float:
    if not any(value != 0 for value in values):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def _p_text(value: float) -> str:
    return "<.001" if value < 0.001 else f"={value:.3f}".replace("0.", ".")


def _plot_summary(data: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=False, constrained_layout=True)
    for axis, name in zip(axes, DATASET_ORDER):
        dataset = data["datasets"][name]
        e4 = _dimension_rows(dataset, "e4")
        e8 = _dimension_rows(dataset, "e8")
        for seed in range(3):
            values = [_metric(e4[seed]), _metric(e8[seed])]
            axis.plot([4, 8], values, color="#777777", alpha=0.45, linewidth=1)
            axis.scatter([4, 8], values, color=["#4C72B0", "#DD8452"], s=28, zorder=3)
        means = [statistics.mean(map(_metric, e4)), statistics.mean(map(_metric, e8))]
        axis.plot([4, 8], means, color="#222222", linewidth=2.2, marker="D", markersize=5)
        axis.set_xticks([4, 8], ["E=4", "E=8"])
        axis.set_title(LABELS[name])
        axis.set_ylabel("Held-out normalized likelihood")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Paired current-code source embeddings: external transfer")
    fig.savefig(SUMMARY_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_subjects(data: dict) -> None:
    apply_presentation_style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=False, constrained_layout=True)
    for axis, name in zip(axes, DATASET_ORDER):
        values = np.asarray(_subject_deltas(data["datasets"][name]))
        violin = axis.violinplot(values, positions=[0], widths=0.7, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#8172B3")
            body.set_alpha(0.25)
        axis.scatter(rng.normal(0, 0.055, len(values)), values, s=13, alpha=0.45, color="#8172B3")
        axis.plot([-0.18, 0.18], [np.median(values)] * 2, color="#222222", linewidth=2)
        axis.scatter([0], [np.mean(values)], marker="D", facecolor="white", edgecolor="#222222", zorder=4)
        axis.axhline(0, color="#C44E52", linewidth=1.4)
        axis.set_xticks([])
        axis.set_title(f"{LABELS[name]}\nWilcoxon p{_p_text(_p(values.tolist()))}")
        axis.set_ylabel("E=8 minus E=4 likelihood")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Subject-paired benefit of an eight-dimensional embedding")
    fig.savefig(SUBJECT_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _result_markdown(data: dict) -> str:
    lines = [
        "![Paired seed-level E=4 and E=8 transfer](../fig_embedding_dimension_transfer.png)",
        "",
        "Each thin line joins the same source seed. The black diamond-line is the three-seed mean.",
        "",
        "![Subject-paired E=8 minus E=4 likelihood](../fig_subject_embedding_dimension_delta.png)",
        "",
        "Dots are subjects; the short bar is the median and the hollow diamond is the mean. P-values are two-sided paired Wilcoxon signed-rank tests against zero.",
        "",
        "| cohort | subjects | held-out trials | E=4 mean ± SD | E=8 mean ± SD | subject median Δ | subject mean Δ | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DATASET_ORDER:
        dataset = data["datasets"][name]
        e4 = [_metric(row) for row in _dimension_rows(dataset, "e4")]
        e8 = [_metric(row) for row in _dimension_rows(dataset, "e8")]
        deltas = _subject_deltas(dataset)
        lines.append(
            f"| {LABELS[name]} | {dataset['n_subjects']} | {dataset['n_heldout_trials']:,} | "
            f"{statistics.mean(e4):.5f} ± {statistics.stdev(e4):.5f} | "
            f"{statistics.mean(e8):.5f} ± {statistics.stdev(e8):.5f} | "
            f"{statistics.median(deltas):+.5f} | {statistics.mean(deltas):+.5f} | "
            f"{_p(deltas):.3g} |"
        )
    return "\n".join(lines)


def main() -> None:
    data = json.loads(DATA.read_text())
    if data.get("status") != "complete":
        raise RuntimeError("embedding_dimension_results.json is not complete")
    if tuple(data.get("datasets", {})) != DATASET_ORDER:
        raise AssertionError("Dataset order or membership drifted")
    _plot_summary(data)
    _plot_subjects(data)
    text = REPORT.read_text()
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    REPORT.write_text(f"{before}{START}\n{_result_markdown(data)}\n{END}{after}")


if __name__ == "__main__":
    main()
