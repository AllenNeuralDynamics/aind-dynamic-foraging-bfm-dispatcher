"""Render the offline D=614 subject-embedding analysis."""

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
GROUP_ORDER = ("aind_source", "aind_heldout", "grossman", "chen", "zid")
TARGET_ORDER = GROUP_ORDER[1:]
COLORS = {
    "aind_source": "#A7A7A7",
    "aind_heldout": "#111111",
    "grossman": "#DD8452",
    "chen": "#55A868",
    "zid": "#8172B3",
}
MARKERS = {
    "aind_source": ".",
    "aind_heldout": "o",
    "grossman": "^",
    "chen": "s",
    "zid": "D",
}
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
CHI2_95_DF2 = 5.991464547107979


def _arrays(seed: dict) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            [subject["embedding"] for subject in seed["groups"][name]["subjects"]],
            dtype=float,
        )
        for name in GROUP_ORDER
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


def _distances(values: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray) -> np.ndarray:
    centered = values - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered))


def _statistics(seed: dict) -> dict:
    arrays = _arrays(seed)
    source = arrays["aind_source"]
    mean = source.mean(axis=0)
    inverse_covariance = np.linalg.pinv(np.cov(source, rowvar=False))
    all_distances = {
        name: _distances(values, mean, inverse_covariance)
        for name, values in arrays.items()
    }
    threshold = float(np.quantile(all_distances["aind_source"], 0.95))
    return {
        "threshold": threshold,
        "groups": {
            name: {
                "distances": distances,
                "median": float(np.median(distances)),
                "outside": float(np.mean(distances > threshold)),
                "centroid": float(
                    _distances(arrays[name].mean(axis=0)[None, :], mean, inverse_covariance)[0]
                ),
            }
            for name, distances in all_distances.items()
        },
    }


def _plot_pca(data: dict) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(3, 3, figsize=(12.8, 12.2), constrained_layout=True)
    for row, seed in enumerate(data["seeds"]):
        arrays = _arrays(seed)
        mean, components, explained = _pca(arrays["aind_source"])
        projected = {
            name: _project(values, mean, components) for name, values in arrays.items()
        }
        for column, (x_index, y_index) in enumerate(PAIR_INDICES):
            axis = axes[row, column]
            source_scores = projected["aind_source"]
            std_x = float(np.std(source_scores[:, x_index], ddof=1))
            std_y = float(np.std(source_scores[:, y_index], ddof=1))
            axis.add_patch(
                Ellipse(
                    (0, 0),
                    width=2 * np.sqrt(CHI2_95_DF2) * std_x,
                    height=2 * np.sqrt(CHI2_95_DF2) * std_y,
                    facecolor="none",
                    edgecolor="#777777",
                    linestyle="--",
                    linewidth=1.2,
                    zorder=1,
                )
            )
            for name in GROUP_ORDER:
                values = projected[name]
                axis.scatter(
                    values[:, x_index],
                    values[:, y_index],
                    s=10 if name == "aind_source" else 20,
                    alpha=0.22 if name == "aind_source" else 0.58,
                    color=COLORS[name],
                    marker=MARKERS[name],
                    edgecolors="none",
                    label=data["groups"][name]["label"],
                    zorder=2 if name == "aind_source" else 3,
                )
            axis.scatter(0, 0, marker="*", s=135, color="#CCB974", edgecolor="#333333", linewidth=0.5, zorder=5)
            axis.axhline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
            axis.axvline(0, color="#DDDDDD", linewidth=0.7, zorder=0)
            axis.set_xlabel(f"PC{x_index + 1} ({explained[x_index] * 100:.1f}%)")
            axis.set_ylabel(f"PC{y_index + 1} ({explained[y_index] * 100:.1f}%)")
            axis.set_aspect("equal", adjustable="datalim")
            if row == 0:
                axis.set_title(f"PC{x_index + 1} vs PC{y_index + 1}")
            if column == 0:
                axis.text(-0.26, 0.5, f"Seed {seed['seed']}", transform=axis.transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Unseen subjects in the D=614 source-trained embedding space\nPCA is fit on source-training AIND mice separately for each seed", fontsize=16)
    fig.savefig(PCA_FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _plot_distances(data: dict, statistics: list[dict]) -> None:
    apply_presentation_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), constrained_layout=True, sharey=True)
    rng = np.random.default_rng(17)
    for axis, seed, seed_statistics in zip(axes, data["seeds"], statistics):
        values = [seed_statistics["groups"][name]["distances"] for name in GROUP_ORDER]
        violins = axis.violinplot(values, positions=range(len(values)), showextrema=False, widths=0.78)
        for body, name in zip(violins["bodies"], GROUP_ORDER):
            body.set_facecolor(COLORS[name])
            body.set_edgecolor("none")
            body.set_alpha(0.42)
        for position, (name, distances) in enumerate(zip(GROUP_ORDER, values)):
            jitter = rng.uniform(-0.16, 0.16, len(distances))
            axis.scatter(position + jitter, distances, s=7, alpha=0.28, color=COLORS[name], edgecolors="none")
            axis.scatter(position, np.median(distances), s=34, marker="_", linewidth=2.2, color="#111111", zorder=5)
        axis.axhline(seed_statistics["threshold"], color="#777777", linestyle="--", linewidth=1.2, label="source empirical 95th percentile")
        axis.set_xticks(range(len(GROUP_ORDER)), ["Source", "AIND\nheld-out", "Grossman", "Chen", "Zid"])
        axis.set_title(f"Seed {seed['seed']}")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("Full 4D Mahalanobis distance\nfrom source center")
    fig.suptitle("Distance from the source-training AIND embedding distribution", fontsize=15)
    fig.savefig(DISTANCE_FIGURE, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _report_body(data: dict, statistics: list[dict]) -> str:
    labels = data["groups"]
    rows = []
    for seed, seed_statistics in zip(data["seeds"], statistics):
        for name in TARGET_ORDER:
            group = seed_statistics["groups"][name]
            rows.append(
                f"| {seed['seed']} | {labels[name]['label']} | {labels[name]['n_subjects']} | "
                f"{group['median']:.2f} | {group['centroid']:.2f} | {group['outside'] * 100:.1f}% |"
            )
    direction = []
    for name in TARGET_ORDER[1:]:
        count = sum(
            item["groups"][name]["median"] > item["groups"]["aind_heldout"]["median"]
            for item in statistics
        )
        direction.append(f"{labels[name]['label']} ({count}/3 seeds)")
    explained = []
    for seed in data["seeds"]:
        source = _arrays(seed)["aind_source"]
        _, _, variance = _pca(source)
        explained.append(float(variance[:3].sum()))
    return f"""{START}
## Result

![Subject embeddings in source-fitted PCA space](../fig_embedding_space_pca.png)

The comparison is deliberately anchored on **held-out AIND mice**, not on the source-training mice. The 149 held-out AIND mice and all external subjects were unseen while the GRU core was trained; each entered at the source-embedding mean and received the same 500-step, learning-rate-0.001 embedding-only adaptation. The 614 source-training mice define the coordinate system and reference distribution.

Each seed has its own independently learned embedding coordinates, so PCA was fit on that seed's source-training mice and no raw coordinates were pooled across seeds. The first three PCs contain {min(explained) * 100:.1f}%--{max(explained) * 100:.1f}% of source variance across seeds. The star is the source mean and therefore the initialization point for every adapted subject; the dashed ellipse is the Gaussian 95% contour of the source distribution in each displayed 2D projection.

![Full-dimensional distance from the source distribution](../fig_embedding_space_distance.png)

The second figure checks the same question in the full four-dimensional space rather than relying on a 2D projection. Distances use each seed's source mean and covariance; the dashed line is that seed's empirical source 95th percentile. External median distance exceeds the held-out-AIND median for {', '.join(direction)}. This is descriptive evidence of how far each transferred cohort must move in the learned subject space, not a test of a pure species effect.

| seed | population | n | median distance | centroid distance | outside source 95% |
|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Scientific interpretation

- **Matched internal control:** held-out AIND mice are the clean reference for transfer because, like external subjects, they were absent from source-core training and only their embeddings were adapted.
- **Main result:** held-out AIND is calibrated to the source distribution (4.0%--5.4% outside the source 95th percentile), Grossman is moderately shifted (14.6%--22.9%), and Chen (100%) and Zid (97.3%--98.1%) are strongly displaced in every seed.
- **Task is a better first explanation than species:** Chen mice and Zid humans share the restless random-walk task and both move far from the AIND manifold, while Grossman mice perform a blockwise task closer to AIND dynamic foraging and remain much nearer. This repeated cross-seed geometry argues against reading the Zid separation as simply mouse versus human.
- **What proximity means:** overlap with held-out AIND says the frozen core can represent the target behavior using subject coordinates similar to those used for new in-distribution mice. Larger distance says adaptation found a more out-of-distribution coordinate; it does not by itself mean worse prediction.
- **What this cannot identify:** dataset, task schedule, species, recording duration, and adaptation-data volume change together. Consequently, external separation cannot be assigned to species alone. Grossman and Chen are especially useful mouse controls for judging whether task structure, rather than species, drives the displacement.
- **Seed discipline:** agreement of the qualitative ordering across independently trained spaces is stronger evidence than any absolute PC direction. PC axes and embedding coordinates have no cross-seed identity.

## Reproduce

The committed JSON contains every 4D embedding plus SHA-256 digests of the downloaded tables, subject maps, and adaptation summaries. Regenerate both figures and this report offline with:

```bash
make r3
```
{END}"""


def main() -> None:
    data = json.loads(DATA.read_text())
    if data["contract"]["cross_seed_rule"] != "Never pool raw coordinates; fit and interpret each seed separately":
        raise AssertionError("Unexpected cross-seed analysis contract")
    statistics = [_statistics(seed) for seed in data["seeds"]]
    _plot_pca(data)
    _plot_distances(data, statistics)
    report = REPORT.read_text()
    before, remainder = report.split(START, 1)
    _, after = remainder.split(END, 1)
    REPORT.write_text(before + _report_body(data, statistics) + after)
    print(f"Wrote {PCA_FIGURE}, {DISTANCE_FIGURE}, and {REPORT}")


if __name__ == "__main__":
    main()
