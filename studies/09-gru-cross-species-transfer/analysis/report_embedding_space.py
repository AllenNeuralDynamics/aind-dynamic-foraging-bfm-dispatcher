"""Render source-fitted PCA and Mahalanobis analyses for all Stage-A cohorts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from plot_style import apply_presentation_style  # noqa: E402


DATA = STUDY / "analysis" / "embedding_space_results.json"
PCA_FIGURE = STUDY / "analysis" / "fig_embedding_space_pca.png"
DISTANCE_FIGURE = STUDY / "analysis" / "fig_embedding_space_distance.png"
REPORT = STUDY / "analysis" / "reports" / "r3-embedding-space.md"
START = "<!-- BEGIN result-3 -->"
END = "<!-- END result-3 -->"
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
CHI2_95_DF2 = 5.991464547107979
COLORS = {
    "aind_source": "#B7B7B7",
    "aind_heldout": "#111111",
    "grossman": "#4C72B0",
    "chen": "#55A868",
    "zid": "#8172B3",
    "lebedeva": "#C44E52",
    "beron": "#64B5CD",
    "kwak": "#937860",
    "miller": "#CCB974",
    "findling": "#DA8BC3",
    "tang": "#8C8C8C",
    "alsio": "#1F77B4",
    "eckstein": "#2CA02C",
    "costa": "#9467BD",
    "lopez_mouse": "#D62728",
}
MARKERS = (".", "o", "^", "s", "D", "v", "P", "X", "<", ">", "h", "p", "*", "8", "d")


def _group_order(data: dict) -> tuple[str, ...]:
    order = tuple(data["groups"])
    if order[:2] != ("aind_source", "aind_heldout"):
        raise AssertionError("Embedding reference group order drifted")
    return order


def _arrays(seed: dict, order: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [subject["embedding"] for subject in seed["groups"][name]["subjects"]],
            dtype=float,
        )
        for name in order
    }


def _pca(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = source.mean(axis=0)
    centered = source - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    components = right_vectors.copy()
    for index, component in enumerate(components):
        anchor = int(np.argmax(np.abs(component)))
        if component[anchor] < 0:
            components[index] *= -1
    explained = singular_values**2 / np.sum(singular_values**2)
    return mean, components, explained


def _project(values: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (values - mean) @ components.T


def _distances(
    values: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray
) -> np.ndarray:
    centered = values - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered))


def _statistics(seed: dict, order: tuple[str, ...]) -> dict:
    arrays = _arrays(seed, order)
    source = arrays["aind_source"]
    mean = source.mean(axis=0)
    inverse_covariance = np.linalg.pinv(np.cov(source, rowvar=False))
    distances = {
        name: _distances(values, mean, inverse_covariance)
        for name, values in arrays.items()
    }
    threshold = float(np.quantile(distances["aind_source"], 0.95))
    return {
        "threshold": threshold,
        "groups": {
            name: {
                "distances": value,
                "median": float(np.median(value)),
                "outside": float(np.mean(value > threshold)),
                "centroid": float(
                    _distances(
                        arrays[name].mean(axis=0)[None, :],
                        mean,
                        inverse_covariance,
                    )[0]
                ),
            }
            for name, value in distances.items()
        },
    }


def _plot_pca(data: dict, order: tuple[str, ...]) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(3, 3, figsize=(14.8, 13.2), constrained_layout=True)
    for row, seed in enumerate(data["seeds"]):
        arrays = _arrays(seed, order)
        mean, components, explained = _pca(arrays["aind_source"])
        projected = {
            name: _project(values, mean, components) for name, values in arrays.items()
        }
        for column, (x_index, y_index) in enumerate(PAIR_INDICES):
            axis = axes[row, column]
            source_scores = projected["aind_source"]
            axis.add_patch(
                Ellipse(
                    (0, 0),
                    width=2
                    * np.sqrt(CHI2_95_DF2)
                    * float(np.std(source_scores[:, x_index], ddof=1)),
                    height=2
                    * np.sqrt(CHI2_95_DF2)
                    * float(np.std(source_scores[:, y_index], ddof=1)),
                    facecolor="none",
                    edgecolor="#777777",
                    linestyle="--",
                    linewidth=1,
                    zorder=1,
                )
            )
            for group_index, name in enumerate(order):
                values = projected[name]
                axis.scatter(
                    values[:, x_index],
                    values[:, y_index],
                    s=8 if name == "aind_source" else 17,
                    alpha=0.16 if name == "aind_source" else 0.48,
                    color=COLORS[name],
                    marker=MARKERS[group_index],
                    edgecolors="none",
                    label=data["groups"][name]["label"],
                    zorder=2 if name == "aind_source" else 3,
                )
            axis.scatter(
                0,
                0,
                marker="*",
                s=115,
                color="#FFD54F",
                edgecolor="#333333",
                linewidth=0.5,
                zorder=5,
            )
            axis.axhline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
            axis.axvline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
            axis.set_xlabel(f"PC{x_index + 1} ({explained[x_index] * 100:.1f}%)")
            axis.set_ylabel(f"PC{y_index + 1} ({explained[y_index] * 100:.1f}%)")
            if row == 0:
                axis.set_title(f"PC{x_index + 1} vs PC{y_index + 1}")
            if column == 0:
                axis.text(
                    -0.23,
                    0.5,
                    f"Seed {seed['seed']}",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(
        "Held-out AIND and external subjects in source-fitted embedding PCA\n"
        "PCA is fit separately on the 614 source-training mice for each seed",
        fontsize=16,
    )
    fig.savefig(PCA_FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_distances(
    data: dict, order: tuple[str, ...], statistics: list[dict]
) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(3, 1, figsize=(19, 13), constrained_layout=True)
    rng = np.random.default_rng(17)
    for axis, seed, seed_statistics in zip(axes, data["seeds"], statistics):
        values = [seed_statistics["groups"][name]["distances"] for name in order]
        positions = np.arange(len(values))
        violins = axis.violinplot(
            values, positions=positions, showextrema=False, widths=0.76
        )
        for body, name in zip(violins["bodies"], order):
            body.set_facecolor(COLORS[name])
            body.set_edgecolor("none")
            body.set_alpha(0.36)
        for position, (name, distances) in enumerate(zip(order, values)):
            jitter = rng.uniform(-0.14, 0.14, len(distances))
            axis.scatter(
                position + jitter,
                distances,
                s=5,
                alpha=min(0.3, 15 / len(distances)),
                color=COLORS[name],
                edgecolors="none",
            )
            axis.scatter(
                position,
                np.median(distances),
                s=30,
                marker="_",
                linewidth=2,
                color="#111111",
                zorder=5,
            )
        axis.axhline(
            seed_statistics["threshold"],
            color="#777777",
            linestyle="--",
            linewidth=1,
            label="source empirical 95th percentile",
        )
        axis.set_xticks(
            positions,
            [
                data["groups"][name]["label"].replace(" ", "\n", 1)
                for name in order
            ],
            fontsize=7.5,
        )
        axis.set_title(f"Seed {seed['seed']}")
        axis.set_ylabel("4D Mahalanobis distance")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Distance from the source-training AIND embedding distribution",
        fontsize=16,
    )
    fig.savefig(DISTANCE_FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _report_body(
    data: dict, order: tuple[str, ...], statistics: list[dict]
) -> str:
    external = order[2:]
    explained = []
    for seed in data["seeds"]:
        _, _, variance = _pca(_arrays(seed, order)["aind_source"])
        explained.append(float(variance[:3].sum()))
    rankings = sorted(
        (
            np.mean([item["groups"][name]["median"] for item in statistics]),
            name,
        )
        for name in external
    )
    rows = []
    for seed, seed_statistics in zip(data["seeds"], statistics):
        for name in order[1:]:
            group = seed_statistics["groups"][name]
            rows.append(
                f"| {seed['seed']} | {data['groups'][name]['label']} | "
                f"{data['groups'][name]['species']} | {data['groups'][name]['n_subjects']} | "
                f"{group['median']:.2f} | {group['centroid']:.2f} | "
                f"{group['outside'] * 100:.1f}% |"
            )
    direction = [
        (
            name,
            sum(
                item["groups"][name]["median"]
                > item["groups"]["aind_heldout"]["median"]
                for item in statistics
            ),
        )
        for name in external
    ]
    return f"""## Result

![All transferred subjects in source-fitted PCA space](../fig_embedding_space_pca.png)

The primary comparison is **held-out AIND mice versus external subjects**. All of
these subjects were unseen during GRU-core training, initialized at the source
embedding mean, and adapted for the same 500 steps at learning rate 0.001 while
the core remained frozen. The 614 source-training mice define the coordinate
system but are not treated as the transfer control.

PCA is fit independently to each seed's source-training mice; raw coordinates
are never pooled across seeds. The first three PCs explain
{min(explained) * 100:.1f}%–{max(explained) * 100:.1f}% of source variance. The
star is the common initialization point and the dashed ellipse is the Gaussian
95% source contour in each displayed projection.

![Full-dimensional distance from the source distribution](../fig_embedding_space_distance.png)

The Mahalanobis analysis uses all four embedding dimensions and each seed's
source covariance. External median distance exceeds the held-out-AIND median in
{'; '.join(f"{data['groups'][name]['label']} ({count}/3 seeds)" for name, count in direction)}.

| seed | population | species | n | median distance | centroid distance | outside source 95% |
|---:|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Cross-cohort read

Average median distance across the three independently trained spaces, nearest
to farthest, is:

{chr(10).join(f"{index}. **{data['groups'][name]['label']}** — {value:.2f}" for index, (value, name) in enumerate(rankings, start=1))}

This ordering is descriptive. Species, task schedule, reward contingencies,
recording duration, and adaptation-data volume change together across these
datasets, so distance cannot be interpreted as a pure species effect. Proximity
means the frozen core can express a target near the coordinates used by new
in-distribution mice; distance does not by itself imply poor prediction.

## Reproduce

The committed JSON contains every 4D embedding plus SHA-256 digests of the
downloaded embedding tables, subject maps, and adaptation summaries. Regenerate
both figures and this report offline with:

```bash
make r3
```
"""


def main() -> None:
    data = json.loads(DATA.read_text())
    order = _group_order(data)
    if set(order) != set(COLORS):
        raise AssertionError("Embedding plot color map does not match frozen groups")
    statistics = [_statistics(seed, order) for seed in data["seeds"]]
    _plot_pca(data, order)
    _plot_distances(data, order, statistics)
    body = _report_body(data, order, statistics)
    text = REPORT.read_text()
    start_end = text.index(START) + len(START)
    end_start = text.index(END, start_end)
    REPORT.write_text(text[:start_end] + "\n" + body + text[end_start:])
    print(f"Wrote {PCA_FIGURE}, {DISTANCE_FIGURE}, and {REPORT}")


if __name__ == "__main__":
    main()
