"""Freeze auditable categorical and empirical task-design features for Study 09."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402
from external_bandit_datasets.adapters import build_dataset  # noqa: E402
from external_bandit_datasets.sources import SOURCES, file_digest  # noqa: E402


GENERALIZATION = {
    4: STUDY / "analysis" / "generalization_drivers.json",
    8: STUDY / "analysis" / "generalization_drivers_e8.json",
}
ANNOTATIONS = STUDY / "analysis" / "task_design_annotations.json"
OUTPUT = {
    4: STUDY / "analysis" / "task_design_features.json",
    8: STUDY / "analysis" / "task_design_features_e8.json",
}
AIND_CONFIGS = (
    REPO / "code" / "config" / "data" / "mice_snapshot.yaml",
    REPO / "code" / "config" / "data" / "task" / "uncoupled_block.yaml",
    REPO / "code" / "config" / "data" / "task" / "coupled_block.yaml",
)
PROBABILITY_COLUMNS = (
    "reward_probability_arm_0",
    "reward_probability_arm_1",
)
SCHEDULE_FEATURES = (
    "mean_arm_lag1_autocorrelation",
    "reward_gap_lag1_autocorrelation",
    "cross_arm_correlation",
    "probability_change_rate",
    "mean_absolute_arm_step_on_change",
    "mean_absolute_reward_gap",
    "mean_reward_probability",
    "equal_probability_fraction",
)
DISTANCE_FEATURES = (
    "mean_arm_lag1_autocorrelation",
    "cross_arm_correlation",
    "probability_change_rate",
    "mean_absolute_arm_step_on_change",
    "mean_absolute_reward_gap",
    "mean_reward_probability",
    "equal_probability_fraction",
)
N_PERMUTATIONS = 100_000
N_BOOTSTRAPS = 20_000
RNG_SEED = 20260906


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=STUDY / "data-cache" / "raw",
        help="Directory containing checksum-verified source files by dataset name.",
    )
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sets_overlap(left: object, right: object) -> bool:
    left_values = set(left if isinstance(left, list) else [left])
    right_values = set(right if isinstance(right, list) else [right])
    return bool(left_values & right_values)


def _categorical_distance(
    annotation: dict, prototype: dict, axes: list[str]
) -> float:
    return float(
        np.mean(
            [
                0.0 if _sets_overlap(annotation[axis], prototype[axis]) else 1.0
                for axis in axes
            ]
        )
    )


def _safe_correlation(x: list[float], y: list[float]) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) < 3 or np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        return math.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _subject_schedule_features(subject) -> dict[str, float]:
    arm_previous = [[], []]
    arm_next = [[], []]
    gap_previous: list[float] = []
    gap_next: list[float] = []
    arm_0: list[float] = []
    arm_1: list[float] = []
    transition_steps: list[np.ndarray] = []

    for _, session in subject.groupby("ses_idx", sort=False):
        values = session.sort_values("trial")[list(PROBABILITY_COLUMNS)].to_numpy(
            dtype=float
        )
        arm_0.extend(values[:, 0])
        arm_1.extend(values[:, 1])
        if len(values) < 2:
            continue
        for arm in (0, 1):
            arm_previous[arm].extend(values[:-1, arm])
            arm_next[arm].extend(values[1:, arm])
        gaps = values[:, 1] - values[:, 0]
        gap_previous.extend(gaps[:-1])
        gap_next.extend(gaps[1:])
        transition_steps.extend(np.abs(np.diff(values, axis=0)))

    values = np.column_stack([arm_0, arm_1])
    steps = np.asarray(transition_steps, dtype=float)
    changed_transitions = np.any(steps > 1e-12, axis=1)
    nonzero_arm_steps = steps[steps > 1e-12]
    arm_correlations = [
        _safe_correlation(arm_previous[arm], arm_next[arm]) for arm in (0, 1)
    ]
    finite_arm_correlations = [
        value for value in arm_correlations if math.isfinite(value)
    ]
    if not finite_arm_correlations:
        raise AssertionError("No estimable within-session arm autocorrelation")
    return {
        "mean_arm_lag1_autocorrelation": float(
            np.mean(finite_arm_correlations)
        ),
        "reward_gap_lag1_autocorrelation": _safe_correlation(
            gap_previous, gap_next
        ),
        "cross_arm_correlation": _safe_correlation(arm_0, arm_1),
        "probability_change_rate": float(np.mean(changed_transitions)),
        "mean_absolute_arm_step_on_change": float(np.mean(nonzero_arm_steps)),
        "mean_absolute_reward_gap": float(np.mean(np.abs(values[:, 1] - values[:, 0]))),
        "mean_reward_probability": float(np.mean(values)),
        "equal_probability_fraction": float(
            np.mean(np.isclose(values[:, 0], values[:, 1]))
        ),
    }


def _cohort_schedule_features(df) -> tuple[dict, list[dict]]:
    subject_rows = []
    for subject_id, subject in df.groupby("subject_id", sort=True):
        subject_rows.append(
            {
                "subject_id": str(subject_id),
                **_subject_schedule_features(subject),
            }
        )
    summary = {}
    for feature in SCHEDULE_FEATURES:
        values = np.asarray(
            [row[feature] for row in subject_rows if math.isfinite(row[feature])],
            dtype=float,
        )
        if not len(values):
            raise AssertionError(f"No finite values for {feature}")
        summary[feature] = {
            "subject_balanced_mean": float(values.mean()),
            "subject_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_subjects_contributing": int(len(values)),
        }
    return summary, subject_rows


def _spearman_summary(
    x: list[float], y: list[float], rng: np.random.Generator
) -> dict:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    x_rank = rankdata(x_values)
    y_rank = rankdata(y_values)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    observed = float((x_centered @ y_centered) / denominator)
    if len(x_values) <= 8:
        indices = np.asarray(
            list(itertools.permutations(range(len(y_rank)))), dtype=int
        )
        permuted_correlations = (
            (y_rank[indices] - y_rank.mean()) @ x_centered
        ) / denominator
        permutation_p = float(
            np.mean(np.abs(permuted_correlations) >= abs(observed))
        )
        n_permutations = math.factorial(len(x_values))
        permutation_method = "exact"
    else:
        extreme = 0
        remaining = N_PERMUTATIONS
        while remaining:
            batch_size = min(10_000, remaining)
            indices = np.argsort(rng.random((batch_size, len(y_rank))), axis=1)
            correlations = ((y_rank[indices] - y_rank.mean()) @ x_centered) / denominator
            extreme += int(np.sum(np.abs(correlations) >= abs(observed)))
            remaining -= batch_size
        permutation_p = (extreme + 1) / (N_PERMUTATIONS + 1)
        n_permutations = N_PERMUTATIONS
        permutation_method = "monte_carlo"

    bootstrap_values = []
    for _ in range(N_BOOTSTRAPS):
        indices = rng.integers(0, len(x_values), size=len(x_values))
        bootstrap_x = rankdata(x_values[indices])
        bootstrap_y = rankdata(y_values[indices])
        bootstrap_x -= bootstrap_x.mean()
        bootstrap_y -= bootstrap_y.mean()
        bootstrap_denominator = np.linalg.norm(bootstrap_x) * np.linalg.norm(
            bootstrap_y
        )
        if bootstrap_denominator:
            bootstrap_values.append(
                float((bootstrap_x @ bootstrap_y) / bootstrap_denominator)
            )
    leave_one_out = [
        _rank_correlation(
            np.delete(x_values, index), np.delete(y_values, index)
        )
        for index in range(len(x_values))
    ]
    return {
        "n_cohorts": len(x_values),
        "spearman_rho": observed,
        "permutation_p_two_sided": float(permutation_p),
        "permutation_method": permutation_method,
        "permutations": n_permutations,
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap_values, 0.025)),
            float(np.quantile(bootstrap_values, 0.975)),
        ],
        "bootstraps": N_BOOTSTRAPS,
        "leave_one_out_range": [min(leave_one_out), max(leave_one_out)],
    }


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.linalg.norm(x_rank) * np.linalg.norm(y_rank)
    return float((x_rank @ y_rank) / denominator) if denominator else math.nan


def _bh_q_values(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype=float)[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return [float(value) for value in output]


def main() -> None:
    args = _parser().parse_args()
    generalization_path = GENERALIZATION[args.dimension]
    output_path = OUTPUT[args.dimension]
    generalization = json.loads(generalization_path.read_text())
    if int(generalization["contract"]["subject_embedding_size"]) != args.dimension:
        raise AssertionError("Generalization embedding-dimension contract drifted")
    annotations = json.loads(ANNOTATIONS.read_text())
    cohort_order = list(generalization["contract"]["cohort_order"])
    if [name for name in annotations["cohorts"] if name in cohort_order] != cohort_order:
        raise AssertionError("Task annotations must follow the frozen cohort order")

    task_axes = annotations["contract"]["task_structure_axes"]
    all_axes = task_axes + annotations["contract"]["apparatus_axes"]
    prototypes = annotations["aind_source_prototypes"]
    tiers = {
        tier: [name for name in names if name in cohort_order]
        for tier, names in annotations["analysis_tiers"].items()
    }
    tier_by_cohort = {
        name: tier for tier, names in tiers.items() for name in names
    }
    if (
        len(tier_by_cohort) != sum(len(names) for names in tiers.values())
        or set(tier_by_cohort) != set(cohort_order)
    ):
        raise AssertionError("Analysis tiers must partition every cohort exactly once")
    primary_names = tuple(tiers["primary"])
    sensitivity_names = tuple(
        tiers["primary"] + tiers["stress_test"] + tiers["descriptive_only"]
    )
    cohorts = {}
    schedule_available_names = []
    for name in cohort_order:
        annotation = annotations["cohorts"][name]
        task_distances = {
            key: _categorical_distance(annotation, prototype, task_axes)
            for key, prototype in prototypes.items()
        }
        full_distances = {
            key: _categorical_distance(annotation, prototype, all_axes)
            for key, prototype in prototypes.items()
        }
        source = SOURCES[name]
        source_path = args.raw_root / name / source.filename
        verified_digest = file_digest(source_path, source.digest_algorithm)
        if verified_digest != source.digest:
            raise AssertionError(f"Source checksum mismatch for {name}")
        source_record = {
            **asdict(source),
            "verified_digest": verified_digest,
        }
        schedule_summary = None
        subject_features = None
        schedule_status = (
            "excluded: canonical adapter does not expose complete trial-wise arm probabilities"
        )
        df, _, audit = build_dataset(name, source_path)
        probabilities_available = all(column in df for column in PROBABILITY_COLUMNS)
        if probabilities_available:
            probabilities_available = not df[list(PROBABILITY_COLUMNS)].isna().any().any()
        if probabilities_available:
            schedule_summary, subject_features = _cohort_schedule_features(df)
            schedule_status = "included: complete trial-wise arm probabilities"
            schedule_available_names.append(name)
        if name in tiers["quarantined"] and probabilities_available:
            schedule_status = (
                "available but quarantined: outcome split crosses treatment; excluded "
                "from schedule-distance inference until the within-DMSO rerun"
            )
        cohorts[name] = {
            "label": generalization["cohorts"][name]["label"],
            "species": generalization["cohorts"][name]["species"],
            "analysis_tier": tier_by_cohort[name],
            "n_subjects": int(df["subject_id"].nunique()),
            "n_sessions": int(df[["subject_id", "ses_idx"]].drop_duplicates().shape[0]),
            "n_trials": int(len(df)),
            "annotation": annotation,
            "categorical_distance": {
                "task_structure": min(task_distances.values()),
                "task_structure_nearest_prototypes": [
                    key for key, value in task_distances.items() if value == min(task_distances.values())
                ],
                "full_design": min(full_distances.values()),
                "full_design_nearest_prototypes": [
                    key for key, value in full_distances.items() if value == min(full_distances.values())
                ],
            },
            "schedule_feature_status": schedule_status,
            "schedule_features": schedule_summary,
            "subject_schedule_features": subject_features,
            "source": source_record,
            "adapter_audit": audit,
            "outcomes": {
                "gru_d614_minus_q_bits_per_trial": generalization["cohorts"][name]["summary"]["gru_d614_minus_q_bits_per_trial"]["mean"],
                "embedding_centroid_mahalanobis": generalization["cohorts"][name]["summary"]["embedding_centroid_mahalanobis"]["mean"],
                **(
                    {
                        "gru_d614_minus_d10_bits_per_trial": generalization[
                            "cohorts"
                        ][name]["summary"]["gru_d614_minus_d10_bits_per_trial"][
                            "mean"
                        ]
                    }
                    if args.dimension == 4
                    else {}
                ),
            },
        }

    schedule_names = tuple(
        name for name in primary_names if name in schedule_available_names
    )
    schedule_matrix = np.asarray(
        [
            [
                cohorts[name]["schedule_features"][feature]["subject_balanced_mean"]
                for feature in DISTANCE_FEATURES
            ]
            for name in schedule_names
        ]
    )
    feature_means = schedule_matrix.mean(axis=0)
    feature_scales = schedule_matrix.std(axis=0, ddof=0)
    if np.any(feature_scales == 0):
        raise AssertionError("A schedule-distance feature is constant across cohorts")
    standardized = (schedule_matrix - feature_means) / feature_scales
    reference_index = schedule_names.index("grossman")
    distances = np.linalg.norm(
        standardized - standardized[reference_index], axis=1
    ) / math.sqrt(len(DISTANCE_FEATURES))
    for index, name in enumerate(schedule_names):
        cohorts[name]["empirical_schedule_distance_to_grossman"] = float(
            distances[index]
        )

    rng = np.random.default_rng(RNG_SEED)
    def categorical_relationships(names: tuple[str, ...]) -> dict:
        delta = [
            cohorts[name]["outcomes"]["gru_d614_minus_q_bits_per_trial"]
            for name in names
        ]
        embedding = [
            cohorts[name]["outcomes"]["embedding_centroid_mahalanobis"]
            for name in names
        ]
        task_distance = [
            cohorts[name]["categorical_distance"]["task_structure"]
            for name in names
        ]
        full_distance = [
            cohorts[name]["categorical_distance"]["full_design"]
            for name in names
        ]
        return {
            "gru_d614_minus_q_vs_task_structure_distance": _spearman_summary(task_distance, delta, rng),
            "embedding_centroid_vs_task_structure_distance": _spearman_summary(task_distance, embedding, rng),
            "gru_d614_minus_q_vs_full_design_distance": _spearman_summary(full_distance, delta, rng),
            "embedding_centroid_vs_full_design_distance": _spearman_summary(full_distance, embedding, rng),
        }

    schedule_distance = [
        cohorts[name]["empirical_schedule_distance_to_grossman"]
        for name in schedule_names
    ]
    schedule_delta = [
        cohorts[name]["outcomes"]["gru_d614_minus_q_bits_per_trial"]
        for name in schedule_names
    ]
    schedule_embedding = [
        cohorts[name]["outcomes"]["embedding_centroid_mahalanobis"]
        for name in schedule_names
    ]
    relationships = {
        **categorical_relationships(primary_names),
        "gru_d614_minus_q_vs_empirical_schedule_distance": _spearman_summary(schedule_distance, schedule_delta, rng),
        "embedding_centroid_vs_empirical_schedule_distance": _spearman_summary(schedule_distance, schedule_embedding, rng),
    }
    sensitivity_relationships = categorical_relationships(sensitivity_names)

    feature_screen = []
    for feature in SCHEDULE_FEATURES:
        values = [cohorts[name]["schedule_features"][feature]["subject_balanced_mean"] for name in schedule_names]
        result = _spearman_summary(values, schedule_delta, rng)
        feature_screen.append({"feature": feature, **result})
    q_values = _bh_q_values([row["permutation_p_two_sided"] for row in feature_screen])
    for row, q_value in zip(feature_screen, q_values, strict=True):
        row["bh_fdr_q"] = q_value

    output = {
        "_meta": build_meta(
            "analysis/freeze_task_design_features.py",
            generalization["_meta"]["wandb_groups"],
            study_root=STUDY,
        ),
        "inputs": {
            str(generalization_path.relative_to(STUDY)): _sha256(
                generalization_path
            ),
            str(ANNOTATIONS.relative_to(STUDY)): _sha256(ANNOTATIONS),
            **{str(path.relative_to(REPO)): _sha256(path) for path in AIND_CONFIGS},
        },
        "contract": {
            "cohort_order": cohort_order,
            "subject_embedding_size": args.dimension,
            "analysis_tiers": tiers,
            "tier_reasons": annotations["tier_reasons"],
            "required_reruns": annotations["required_reruns"],
            "primary_inference_cohorts": list(primary_names),
            "all_valid_sensitivity_cohorts": list(sensitivity_names),
            "schedule_available_cohorts": schedule_available_names,
            "schedule_cohort_order": list(schedule_names),
            "categorical_distance": annotations["contract"],
            "aind_source_prototypes": prototypes,
            "schedule_summary": "Compute within-session metrics per subject, then average subjects equally within each cohort.",
            "empirical_schedule_distance": "Root-mean-square standardized Euclidean distance to Grossman across the declared distance features; standardization uses only the six primary, non-quarantined complete-probability cohorts.",
            "schedule_distance_features": list(DISTANCE_FEATURES),
            "inference": "Primary Spearman inference uses eight primary cohorts for categorical distance and six primary complete-probability cohorts for schedule distance. The all-valid categorical sensitivity uses 12 non-quarantined cohorts. Exact two-sided permutation is used for n<=8; otherwise deterministic 100,000-draw permutation; all analyses use 20,000 cohort bootstraps and a leave-one-cohort-out range.",
            "rng_seed": RNG_SEED,
        },
        "schedule_standardization": {
            feature: {"mean": float(mean), "population_sd": float(scale)}
            for feature, mean, scale in zip(DISTANCE_FEATURES, feature_means, feature_scales, strict=True)
        },
        "cohorts": cohorts,
        "relationships": relationships,
        "sensitivity_relationships": sensitivity_relationships,
        "schedule_feature_screen_vs_gru_d614_minus_q": feature_screen,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
