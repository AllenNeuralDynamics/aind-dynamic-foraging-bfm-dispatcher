"""Render the paired Study 01 E=4/E=8 source-capacity report."""

from __future__ import annotations

import json
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
PERFORMANCE_FIGURE = STUDY / "analysis" / "fig_embedding_dimension_source.png"
SPECTRUM_FIGURE = STUDY / "analysis" / "fig_embedding_dimension_spectrum.png"
REPORT = STUDY / "analysis" / "reports" / "r11-embedding-dimension-capacity.md"
START = "<!-- BEGIN result-11 -->"
END = "<!-- END result-11 -->"
COLORS = {"e4": "#4C72B0", "e8": "#DD8452"}


def _rows(data: dict, dimension: str) -> list[dict]:
    rows = sorted(data["dimensions"][dimension], key=lambda row: int(row["seed"]))
    if [int(row["seed"]) for row in rows] != [0, 1, 2]:
        raise AssertionError(f"{dimension} must contain source seeds 0, 1, and 2")
    return rows


def _subject_deltas(data: dict) -> list[float]:
    e4 = [row["heldout"]["per_subject_test_likelihood"] for row in _rows(data, "e4")]
    e8 = [row["heldout"]["per_subject_test_likelihood"] for row in _rows(data, "e8")]
    subjects = set(e4[0])
    if any(set(row) != subjects for row in e4 + e8):
        raise AssertionError("E=4 and E=8 held-out subject keys do not align")
    return [
        statistics.mean(
            float(e8[seed][subject]["eval_likelihood"])
            - float(e4[seed][subject]["eval_likelihood"])
            for seed in range(3)
        )
        for subject in sorted(subjects)
    ]


def _wilcoxon(values: list[float]) -> float:
    if not any(value != 0 for value in values):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def _p_text(value: float) -> str:
    return "<.001" if value < 0.001 else f"={value:.3f}".replace("0.", ".")


def _plot_performance(data: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)
    e4 = _rows(data, "e4")
    e8 = _rows(data, "e8")

    for seed in range(3):
        values = [
            e4[seed]["heldout"]["test_likelihood"],
            e8[seed]["heldout"]["test_likelihood"],
        ]
        axes[0].plot([4, 8], values, color="#777777", alpha=0.55, linewidth=1)
        axes[0].scatter([4, 8], values, color=[COLORS["e4"], COLORS["e8"]], s=32)
    means = [
        statistics.mean(row["heldout"]["test_likelihood"] for row in e4),
        statistics.mean(row["heldout"]["test_likelihood"] for row in e8),
    ]
    axes[0].plot([4, 8], means, color="#222222", linewidth=2.2, marker="D")
    axes[0].set_xticks([4, 8], ["E=4", "E=8"])
    axes[0].set_ylabel("Held-out normalized likelihood")
    axes[0].set_title("AIND held-out mice\npaired source seeds")

    for position, (name, rows) in enumerate((("e4", e4), ("e8", e8))):
        values = [row["heldout"]["test_minus_adaptation"] for row in rows]
        axes[1].scatter(
            np.full(3, position), values, color=COLORS[name], s=32, zorder=3
        )
        axes[1].plot(
            [position - 0.16, position + 0.16],
            [statistics.mean(values)] * 2,
            color="#222222",
            linewidth=2,
        )
    axes[1].axhline(0, color="#C44E52", linewidth=1.2)
    axes[1].set_xticks([0, 1], ["E=4", "E=8"])
    axes[1].set_ylabel("Test minus adaptation likelihood")
    axes[1].set_title("Generalization gap\npositive favors test half")

    deltas = np.asarray(_subject_deltas(data))
    violin = axes[2].violinplot(deltas, positions=[0], widths=0.7, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#8172B3")
        body.set_alpha(0.25)
    rng = np.random.default_rng(0)
    axes[2].scatter(
        rng.normal(0, 0.055, len(deltas)), deltas, s=13, alpha=0.45, color="#8172B3"
    )
    axes[2].plot(
        [-0.18, 0.18], [np.median(deltas)] * 2, color="#222222", linewidth=2
    )
    axes[2].scatter(
        [0], [np.mean(deltas)], marker="D", facecolor="white", edgecolor="#222222"
    )
    axes[2].axhline(0, color="#C44E52", linewidth=1.2)
    axes[2].set_xticks([])
    axes[2].set_ylabel("E=8 minus E=4 likelihood")
    axes[2].set_title(
        f"149 held-out mice\nWilcoxon p{_p_text(_wilcoxon(deltas.tolist()))}"
    )

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Does a wider source embedding improve AIND generalization?")
    fig.savefig(PERFORMANCE_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _spectrum_matrix(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            row["source_embedding_spectrum"]["explained_variance_ratio"]
            for row in rows
        ]
    )


def _plot_spectrum(data: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for name in ("e4", "e8"):
        values = _spectrum_matrix(_rows(data, name))
        x = np.arange(1, values.shape[1] + 1)
        axes[0].plot(x, values.mean(axis=0), marker="o", color=COLORS[name], label=name.upper())
        axes[0].fill_between(
            x, values.min(axis=0), values.max(axis=0), color=COLORS[name], alpha=0.16
        )
        cumulative = np.cumsum(values, axis=1)
        axes[1].plot(
            x, cumulative.mean(axis=0), marker="o", color=COLORS[name], label=name.upper()
        )
        axes[1].fill_between(
            x,
            cumulative.min(axis=0),
            cumulative.max(axis=0),
            color=COLORS[name],
            alpha=0.16,
        )
    for axis in axes:
        axis.axvline(4.5, color="#777777", linestyle="--", linewidth=1)
        axis.set_xticks(range(1, 9))
        axis.set_xlabel("Principal component")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Explained covariance fraction")
    axes[0].set_title("Source-embedding covariance spectrum")
    axes[1].set_ylabel("Cumulative covariance fraction")
    axes[1].set_title("Cumulative spectrum")
    fig.suptitle("Do E8 dimensions 5–8 carry source-subject variation?")
    fig.savefig(SPECTRUM_FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _result_markdown(data: dict) -> str:
    e4 = _rows(data, "e4")
    e8 = _rows(data, "e8")
    deltas = _subject_deltas(data)
    lines = [
        "![Paired source performance](../fig_embedding_dimension_source.png)",
        "",
        "Thin lines pair the same source seed; black diamonds show seed means. "
        "In the subject panel, dots are held-out mice, the short bar is the median, "
        "and the hollow diamond is the mean.",
        "",
        "![Source embedding covariance spectrum](../fig_embedding_dimension_spectrum.png)",
        "",
        "Lines are the three-seed mean and ribbons span the seed range. The dashed "
        "boundary separates PCs 1–4 from the added E8 PCs 5–8.",
        "",
        "| quantity | E=4 | E=8 | E8 minus E4 |",
        "|---|---:|---:|---:|",
    ]
    e4_test = [row["heldout"]["test_likelihood"] for row in e4]
    e8_test = [row["heldout"]["test_likelihood"] for row in e8]
    e4_gap = [row["heldout"]["test_minus_adaptation"] for row in e4]
    e8_gap = [row["heldout"]["test_minus_adaptation"] for row in e8]
    e4_rank = [row["source_embedding_spectrum"]["effective_rank"] for row in e4]
    e8_rank = [row["source_embedding_spectrum"]["effective_rank"] for row in e8]
    e8_tail = [
        row["source_embedding_spectrum"]["pc5_to_pc8_variance_fraction"]
        for row in e8
    ]
    lines.extend(
        [
            "| held-out likelihood, seed mean ± SD | "
            f"{statistics.mean(e4_test):.5f} ± {statistics.stdev(e4_test):.5f} | "
            f"{statistics.mean(e8_test):.5f} ± {statistics.stdev(e8_test):.5f} | "
            f"{statistics.mean(b - a for a, b in zip(e4_test, e8_test)):+.5f} |",
            "| test minus adaptation likelihood | "
            f"{statistics.mean(e4_gap):+.5f} | {statistics.mean(e8_gap):+.5f} | "
            f"{statistics.mean(b - a for a, b in zip(e4_gap, e8_gap)):+.5f} |",
            "| source-embedding effective rank | "
            f"{statistics.mean(e4_rank):.2f} | {statistics.mean(e8_rank):.2f} | "
            f"{statistics.mean(b - a for a, b in zip(e4_rank, e8_rank)):+.2f} |",
            f"| covariance in PCs 5–8 | — | {statistics.mean(e8_tail):.1%} | — |",
            "",
            "Across 149 held-out mice, the seed-paired E8−E4 likelihood difference "
            f"has median **{statistics.median(deltas):+.5f}**, mean "
            f"**{statistics.mean(deltas):+.5f}**, and two-sided Wilcoxon "
            f"**p={_wilcoxon(deltas):.3g}**.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    data = json.loads(DATA.read_text())
    if data.get("status") != "complete":
        raise RuntimeError("embedding_dimension_results.json is not complete")
    _plot_performance(data)
    _plot_spectrum(data)
    text = REPORT.read_text()
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    REPORT.write_text(f"{before}{START}\n{_result_markdown(data)}\n{END}{after}")


if __name__ == "__main__":
    main()
