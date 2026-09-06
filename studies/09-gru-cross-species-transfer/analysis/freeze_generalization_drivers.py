"""Freeze Study 09 cross-cohort generalization-driver estimates."""

from __future__ import annotations

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
EMBEDDINGS = STUDY / "analysis" / "embedding_space_results.json"
OUTPUT = STUDY / "analysis" / "generalization_drivers.json"
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
    "lopez_mouse": "López-Yépez",
}
N_PERMUTATIONS = 100_000
N_BOOTSTRAPS = 20_000
RNG_SEED = 20260906


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
    if len(x_values) != len(DATASET_ORDER):
        raise AssertionError("Cross-cohort statistic must contain every cohort")
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
    dataset: dict,
    embedding_seed: dict,
) -> dict:
    q_record = dataset["q"]
    d10 = _gru_by_seed(dataset, 10)[seed]
    d614 = _gru_by_seed(dataset, 614)[seed]
    q = _subject_ll(q_record)
    gru10 = _subject_ll(d10)
    gru614 = _subject_ll(d614)

    source = _embedding_map(embedding_seed["groups"]["aind_source"])
    target = _embedding_map(embedding_seed["groups"][dataset_name])
    subject_ids = sorted(q)
    if set(subject_ids) != set(gru10) or set(subject_ids) != set(gru614):
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
    gru10_values = np.asarray([gru10[subject] for subject in subject_ids])
    gru614_values = np.asarray([gru614[subject] for subject in subject_ids])
    q_mean = float(q_values.mean())
    gru10_mean = float(gru10_values.mean())
    gru614_mean = float(gru614_values.mean())
    log_two = math.log(2)
    embedding_provenance = embedding_seed["groups"][dataset_name]["provenance"]

    return {
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
        "gru_d614_minus_d10_bits_per_trial": (
            gru614_mean - gru10_mean
        ) / log_two,
        "embedding_centroid_mahalanobis": centroid_distance,
        "embedding_median_subject_mahalanobis": float(
            np.median(target_distances)
        ),
        "embedding_fraction_outside_source_95pct": float(
            np.mean(target_distances > np.quantile(source_distances, 0.95))
        ),
        "gru_d614_wandb_run_id": d614["wandb_run_id"],
        "gru_d614_artifact_digest": d614["evaluation_artifact"]["digest"],
        "gru_d10_wandb_run_id": d10["wandb_run_id"],
        "gru_d10_artifact_digest": d10["evaluation_artifact"]["digest"],
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


def _mean_sd(records: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(record[key]) for record in records])
    return {
        "mean": float(values.mean()),
        "sd_across_source_seeds": float(values.std(ddof=1)),
    }


def main() -> None:
    matched = json.loads(MATCHED.read_text())
    embeddings = json.loads(EMBEDDINGS.read_text())
    if tuple(matched["datasets"]) != DATASET_ORDER:
        raise AssertionError("Matched-result cohort order drifted")
    if tuple(embeddings["groups"])[2:] != DATASET_ORDER:
        raise AssertionError("Embedding cohort order drifted")
    embedding_seeds = {int(row["seed"]): row for row in embeddings["seeds"]}
    if tuple(sorted(embedding_seeds)) != (0, 1, 2):
        raise AssertionError("Embedding seeds drifted")

    cohorts = {}
    for dataset_name in DATASET_ORDER:
        records = [
            _seed_record(
                dataset_name,
                seed,
                matched["datasets"][dataset_name],
                embedding_seeds[seed],
            )
            for seed in (0, 1, 2)
        ]
        cohorts[dataset_name] = {
            "label": LABELS[dataset_name],
            "species": embeddings["groups"][dataset_name]["species"],
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
                    "gru_d614_minus_d10_bits_per_trial",
                    "embedding_centroid_mahalanobis",
                    "embedding_median_subject_mahalanobis",
                    "embedding_fraction_outside_source_95pct",
                )
            },
        }

    rng = np.random.default_rng(RNG_SEED)
    centroid = [
        cohorts[name]["summary"]["embedding_centroid_mahalanobis"]["mean"]
        for name in DATASET_ORDER
    ]
    median_distance = [
        cohorts[name]["summary"]["embedding_median_subject_mahalanobis"]["mean"]
        for name in DATASET_ORDER
    ]
    q_predictability = [
        cohorts[name]["summary"]["q_bits_above_chance"]["mean"]
        for name in DATASET_ORDER
    ]
    delta = [
        cohorts[name]["summary"]["gru_d614_minus_q_bits_per_trial"]["mean"]
        for name in DATASET_ORDER
    ]
    scaling = [
        cohorts[name]["summary"]["gru_d614_minus_d10_bits_per_trial"]["mean"]
        for name in DATASET_ORDER
    ]

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
            str(EMBEDDINGS.relative_to(STUDY)): _sha256(EMBEDDINGS),
        },
        "contract": {
            "cohort_order": list(DATASET_ORDER),
            "cross_cohort_unit": "one equal-weight cohort",
            "performance_unit": (
                "mean held-out log likelihood across subjects, equal subject weight"
            ),
            "seed_rule": (
                "pair each source seed's D=614 prediction and adapted embedding; "
                "average only after seed-specific estimates"
            ),
            "embedding_distance": (
                "full-4D Mahalanobis distance from external cohort centroid to "
                "the 614-source-mouse centroid, using source covariance per seed"
            ),
            "inference": (
                "Spearman across 13 cohort means; deterministic two-sided "
                "permutation p, cohort bootstrap CI, leave-one-cohort-out range"
            ),
            "rng_seed": RNG_SEED,
        },
        "cohorts": cohorts,
        "relationships": {
            "gru_d614_minus_q_vs_embedding_centroid": _correlation_summary(
                centroid, delta, rng
            ),
            "gru_d614_minus_q_vs_embedding_median_subject_distance": (
                _correlation_summary(median_distance, delta, rng)
            ),
            "gru_d614_minus_q_vs_common_q_predictability": _correlation_summary(
                q_predictability, delta, rng
            ),
            "gru_d614_minus_d10_vs_embedding_centroid": _correlation_summary(
                centroid, scaling, rng
            ),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
