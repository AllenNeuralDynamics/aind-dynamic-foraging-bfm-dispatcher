"""Freeze Study 09 cross-cohort generalization-driver estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402


MATCHED = STUDY / "analysis" / "matched_half_results.json"
EMBEDDINGS = {
    4: STUDY / "analysis" / "embedding_space_results.json",
    8: STUDY / "analysis" / "embedding_space_results_e8.json",
}
E8_PERFORMANCE = (
    STUDY / "analysis" / "embedding_dimension_results.json",
    STUDY / "analysis" / "embedding_dimension_expansion_results.json",
)
ANNOTATIONS = STUDY / "analysis" / "task_design_annotations.json"
OUTPUTS = {
    4: STUDY / "analysis" / "generalization_drivers.json",
    8: STUDY / "analysis" / "generalization_drivers_e8.json",
}
DATASET_ORDER = (
    "grossman",
    "chen",
    "zid",
    "lebedeva",
    "beron",
    "miller",
    "findling",
    "tang",
    "alsio",
    "eckstein",
    "costa",
    "lopez_mouse",
    "hattori",
)
BASE_LABELS = {
    "grossman": "Grossman",
    "chen": "Chen",
    "zid": "Zid",
    "lebedeva": "Lebedeva",
    "beron": "Beron",
    "miller": "Miller",
    "findling": "Findling",
    "tang": "Tang",
    "alsio": "Alsiö",
    "eckstein": "Eckstein",
    "costa": "Costa",
    "lopez_mouse": "López-Yépez",
    "hattori": "Hattori",
}
N_PERMUTATIONS = 100_000
N_BOOTSTRAPS = 20_000
RNG_SEED = 20260906


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(4, 8), default=4)
    return parser


def _performance_by_dataset(
    dimension: int, matched: dict
) -> tuple[dict[str, dict], list[Path]]:
    if dimension == 4:
        return {name: matched["datasets"][name] for name in DATASET_ORDER}, [MATCHED]
    documents = [json.loads(path.read_text()) for path in E8_PERFORMANCE]
    datasets = {
        name: dataset
        for document in documents
        for name, dataset in document["datasets"].items()
    }
    if set(datasets) != set(DATASET_ORDER):
        raise AssertionError("E8 performance cohort membership drifted")
    return datasets, list(E8_PERFORMANCE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gru_by_seed(dataset: dict, nominal_d: int) -> dict[int, dict]:
    rows = {
        int(row["seed"]): row
        for row in dataset["gru"]
        if int(row["nominal_D"]) == nominal_d
    }
    if tuple(sorted(rows)) != (0, 1, 2):
        raise AssertionError(f"Expected three seeds for D={nominal_d}")
    return rows


def _subject_ll(record: dict) -> dict[str, float]:
    return {
        str(subject): float(value)
        for subject, value in record["metrics"][
            "per_subject_mean_log_likelihood_nats"
        ].items()
    }


def _embedding_map(group: dict) -> dict[str, np.ndarray]:
    values = {
        str(row["subject_id"]): np.asarray(row["embedding"], dtype=float)
        for row in group["subjects"]
    }
    if len(values) != len(group["subjects"]):
        raise AssertionError("Duplicate subject ID in embedding group")
    return values


def _mahalanobis(
    values: np.ndarray, mean: np.ndarray, inverse_covariance: np.ndarray
) -> np.ndarray:
    centered = values - mean
    return np.sqrt(
        np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered)
    )


def _correlation_summary(
    x: list[float], y: list[float], rng: np.random.Generator
) -> dict[str, float | int | list[float]]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) < 3 or len(x_values) != len(y_values):
        raise AssertionError("Cross-cohort statistic needs paired values for >=3 cohorts")
    observed = float(spearmanr(x_values, y_values).statistic)

    x_rank = rankdata(x_values)
    y_rank = rankdata(y_values)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    extreme = 0
    remaining = N_PERMUTATIONS
    while remaining:
        batch_size = min(10_000, remaining)
        permutations = np.argsort(
            rng.random((batch_size, len(y_rank))), axis=1
        )
        permuted = y_rank[permutations] - y_rank.mean()
        correlations = (permuted @ x_centered) / denominator
        extreme += int(np.sum(np.abs(correlations) >= abs(observed)))
        remaining -= batch_size
    permutation_p = (extreme + 1) / (N_PERMUTATIONS + 1)

    bootstraps = []
    for _ in range(N_BOOTSTRAPS):
        indices = rng.integers(0, len(x_values), size=len(x_values))
        value = float(spearmanr(x_values[indices], y_values[indices]).statistic)
        if math.isfinite(value):
            bootstraps.append(value)
    if len(bootstraps) < N_BOOTSTRAPS * 0.99:
        raise AssertionError("Too many degenerate bootstrap samples")

    leave_one_out = [
        float(spearmanr(np.delete(x_values, index), np.delete(y_values, index)).statistic)
        for index in range(len(x_values))
    ]
    return {
        "n_cohorts": len(x_values),
        "spearman_rho": observed,
        "permutation_p_two_sided": float(permutation_p),
        "permutations": N_PERMUTATIONS,
        "bootstrap_95_ci": [
            float(np.quantile(bootstraps, 0.025)),
            float(np.quantile(bootstraps, 0.975)),
        ],
        "bootstraps": N_BOOTSTRAPS,
        "leave_one_out_range": [min(leave_one_out), max(leave_one_out)],
    }


def _seed_record(
    dataset_name: str,
    seed: int,
    matched_dataset: dict,
    performance_dataset: dict,
    embedding_seed: dict,
    dimension: int,
) -> dict:
    q_record = matched_dataset["q"]
    d614 = (
        _gru_by_seed(performance_dataset, 614)[seed]
        if dimension == 4
        else {int(row["seed"]): row for row in performance_dataset["e8"]}[seed]
    )
    q = _subject_ll(q_record)
    gru614 = _subject_ll(d614)

    source = _embedding_map(embedding_seed["groups"]["aind_source"])
    target = _embedding_map(embedding_seed["groups"][dataset_name])
    subject_ids = sorted(q)
    if set(subject_ids) != set(gru614):
        raise AssertionError(f"GRU/Q subject mismatch for {dataset_name}, seed {seed}")
    if set(subject_ids) != set(target):
        raise AssertionError(
            f"Performance/embedding subject mismatch for {dataset_name}, seed {seed}"
        )

    source_values = np.stack(list(source.values()))
    target_values = np.stack([target[subject] for subject in subject_ids])
    source_mean = source_values.mean(axis=0)
    inverse_covariance = np.linalg.pinv(np.cov(source_values, rowvar=False))
    source_distances = _mahalanobis(
        source_values, source_mean, inverse_covariance
    )
    target_distances = _mahalanobis(
        target_values, source_mean, inverse_covariance
    )
    centroid_distance = float(
        _mahalanobis(
            target_values.mean(axis=0)[None, :],
            source_mean,
            inverse_covariance,
        )[0]
    )

    q_values = np.asarray([q[subject] for subject in subject_ids])
    gru614_values = np.asarray([gru614[subject] for subject in subject_ids])
    q_mean = float(q_values.mean())
    gru614_mean = float(gru614_values.mean())
    log_two = math.log(2)
    embedding_provenance = embedding_seed["groups"][dataset_name]["provenance"]

    record = {
        "seed": seed,
        "n_subjects": len(subject_ids),
        "q_subject_mean_log_likelihood_nats": q_mean,
        "q_subject_balanced_normalized_likelihood": math.exp(q_mean),
        "q_bits_above_chance": (q_mean + log_two) / log_two,
        "gru_d614_subject_mean_log_likelihood_nats": gru614_mean,
        "gru_d614_subject_balanced_normalized_likelihood": math.exp(gru614_mean),
        "gru_d614_minus_q_bits_per_trial": (gru614_mean - q_mean) / log_two,
        "gru_d614_minus_q_mean_subject_normalized_likelihood": float(
            np.mean(np.exp(gru614_values) - np.exp(q_values))
        ),
        "embedding_centroid_mahalanobis": centroid_distance,
        "embedding_median_subject_mahalanobis": float(
            np.median(target_distances)
        ),
        "embedding_fraction_outside_source_95pct": float(
            np.mean(target_distances > np.quantile(source_distances, 0.95))
        ),
        "gru_d614_wandb_run_id": d614["wandb_run_id"],
        "gru_d614_artifact_digest": d614["evaluation_artifact"]["digest"],
        "q_wandb_run_id": q_record["wandb_run_id"],
        "q_artifact_digest": q_record["training_artifact"]["digest"],
        "embedding_wandb_run_id": embedding_provenance["wandb_run_id"],
        "embedding_artifact_digest": embedding_provenance["artifact_digest"],
        "source_embedding_run_id": embedding_seed["groups"]["aind_source"][
            "provenance"
        ]["run_id"],
        "source_embedding_artifact_digest": embedding_seed["groups"][
            "aind_source"
        ]["provenance"]["artifact_digest"],
    }
    if dimension == 4:
        d10 = _gru_by_seed(performance_dataset, 10)[seed]
        gru10 = _subject_ll(d10)
        if set(subject_ids) != set(gru10):
            raise AssertionError(f"D10/D614 subject mismatch for {dataset_name}, seed {seed}")
        gru10_mean = float(
            np.mean([gru10[subject] for subject in subject_ids])
        )
        record.update(
            {
                "gru_d614_minus_d10_bits_per_trial": (
                    gru614_mean - gru10_mean
                )
                / log_two,
                "gru_d10_wandb_run_id": d10["wandb_run_id"],
                "gru_d10_artifact_digest": d10["evaluation_artifact"]["digest"],
            }
        )
    return record


def _mean_sd(records: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(record[key]) for record in records])
    return {
        "mean": float(values.mean()),
        "sd_across_source_seeds": float(values.std(ddof=1)),
    }


def main() -> None:
    dimension = _parser().parse_args().dimension
    output_path = OUTPUTS[dimension]
    matched = json.loads(MATCHED.read_text())
    embedding_path = EMBEDDINGS[dimension]
    embeddings = json.loads(embedding_path.read_text())
    performance, performance_paths = _performance_by_dataset(dimension, matched)
    annotations = json.loads(ANNOTATIONS.read_text())
    if not set(DATASET_ORDER).issubset(matched["datasets"]):
        raise AssertionError("Matched-result cohort membership drifted")
    if tuple(embeddings["groups"])[2:] != DATASET_ORDER:
        raise AssertionError("Embedding cohort order drifted")
    embedding_seeds = {int(row["seed"]): row for row in embeddings["seeds"]}
    if tuple(sorted(embedding_seeds)) != (0, 1, 2):
        raise AssertionError("Embedding seeds drifted")

    tiers = annotations["analysis_tiers"]
    tier_by_cohort = {
        name: tier for tier, names in tiers.items() for name in names
    }
    if (
        len(tier_by_cohort) != sum(len(names) for names in tiers.values())
        or set(DATASET_ORDER) != set(tier_by_cohort) - set(tiers["quarantined"])
    ):
        raise AssertionError("Analysis tiers must partition every cohort exactly once")
    primary_names = tuple(tiers["primary"])
    sensitivity_names = tuple(
        tiers["primary"] + tiers["stress_test"] + tiers["descriptive_only"]
    )

    cohorts = {}
    for dataset_name in DATASET_ORDER:
        records = [
            _seed_record(
                dataset_name,
                seed,
                matched["datasets"][dataset_name],
                performance[dataset_name],
                embedding_seeds[seed],
                dimension,
            )
            for seed in (0, 1, 2)
        ]
        species = embeddings["groups"][dataset_name]["species"]
        cohorts[dataset_name] = {
            "label": f"{BASE_LABELS[dataset_name]} ({species})",
            "species": species,
            "analysis_tier": tier_by_cohort[dataset_name],
            "n_subjects": records[0]["n_subjects"],
            "seeds": records,
            "summary": {
                key: _mean_sd(records, key)
                for key in (
                    "q_subject_balanced_normalized_likelihood",
                    "q_bits_above_chance",
                    "gru_d614_subject_balanced_normalized_likelihood",
                    "gru_d614_minus_q_bits_per_trial",
                    "gru_d614_minus_q_mean_subject_normalized_likelihood",
                    "embedding_centroid_mahalanobis",
                    "embedding_median_subject_mahalanobis",
                    "embedding_fraction_outside_source_95pct",
                    *(
                        ("gru_d614_minus_d10_bits_per_trial",)
                        if dimension == 4
                        else ()
                    ),
                )
            },
        }

    rng = np.random.default_rng(RNG_SEED)

    def relationship_bundle(names: tuple[str, ...]) -> dict:
        def values(key: str) -> list[float]:
            return [cohorts[name]["summary"][key]["mean"] for name in names]

        delta = values("gru_d614_minus_q_bits_per_trial")
        centroid = values("embedding_centroid_mahalanobis")
        relationships = {
            "gru_d614_minus_q_vs_embedding_centroid": _correlation_summary(
                centroid, delta, rng
            ),
            "gru_d614_minus_q_vs_embedding_median_subject_distance": (
                _correlation_summary(
                    values("embedding_median_subject_mahalanobis"), delta, rng
                )
            ),
            "gru_d614_minus_q_vs_common_q_predictability": _correlation_summary(
                values("q_bits_above_chance"), delta, rng
            ),
        }
        if dimension == 4:
            relationships["gru_d614_minus_d10_vs_embedding_centroid"] = (
                _correlation_summary(
                    centroid, values("gru_d614_minus_d10_bits_per_trial"), rng
                )
            )
        return relationships

    groups = list(
        dict.fromkeys(
            [
                *matched["_meta"]["wandb_groups"],
                *embeddings["_meta"]["wandb_groups"],
            ]
        )
    )
    output = {
        "_meta": build_meta(
            "analysis/freeze_generalization_drivers.py",
            groups,
            study_root=STUDY,
        ),
        "inputs": {
            str(MATCHED.relative_to(STUDY)): _sha256(MATCHED),
            str(embedding_path.relative_to(STUDY)): _sha256(embedding_path),
            str(ANNOTATIONS.relative_to(STUDY)): _sha256(ANNOTATIONS),
            **{
                str(path.relative_to(STUDY)): _sha256(path)
                for path in performance_paths
                if path != MATCHED
            },
        },
        "contract": {
            "cohort_order": list(DATASET_ORDER),
            "subject_embedding_size": dimension,
            "analysis_tiers": tiers,
            "tier_reasons": annotations["tier_reasons"],
            "required_reruns": annotations["required_reruns"],
            "primary_inference_cohorts": list(primary_names),
            "all_valid_sensitivity_cohorts": list(sensitivity_names),
            "cross_cohort_unit": "one equal-weight cohort",
            "performance_unit": (
                "mean held-out log likelihood across subjects, equal subject weight"
            ),
            "seed_rule": (
                "pair each source seed's D=614 prediction and adapted embedding; "
                "average only after seed-specific estimates"
            ),
            "embedding_distance": (
                f"full-{dimension}D Mahalanobis distance from external cohort centroid to "
                "the 614-source-mouse centroid, using source covariance per seed"
            ),
            "inference": (
                f"Primary Spearman inference across {len(primary_names)} declared primary cohorts; "
                f"all-valid sensitivity across {len(sensitivity_names)} non-quarantined cohorts; deterministic "
                "two-sided permutation p, cohort bootstrap CI, and leave-one-cohort-out range"
            ),
            "rng_seed": RNG_SEED,
        },
        "cohorts": cohorts,
        "relationships": relationship_bundle(primary_names),
        "sensitivity_relationships": relationship_bundle(sensitivity_names),
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
