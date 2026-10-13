"""Canonical trial schema and reproducible split-manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_REQUIRED_COLUMNS = (
    "subject_id",
    "ses_idx",
    "trial",
    "animal_response",
    "rewarded",
    "earned_reward",
)


def _deterministically_ordered_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical rows in stable identifier/trial order."""
    return df.sort_values(
        ["subject_id", "ses_idx", "trial"],
        kind="mergesort",
    )


def validate_canonical_table(df: pd.DataFrame) -> None:
    missing = [column for column in CANONICAL_REQUIRED_COLUMNS if column not in df]
    if missing:
        raise ValueError(f"Canonical table is missing columns: {missing}.")
    if df.empty:
        raise ValueError("Canonical table contains no trials.")
    if df[list(CANONICAL_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Canonical columns cannot contain missing values.")
    for column in ("animal_response", "rewarded", "earned_reward"):
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{column!r} must contain only binary 0/1 values.")
    if not np.array_equal(df["rewarded"].to_numpy(), df["earned_reward"].to_numpy()):
        raise ValueError("rewarded and earned_reward must agree row-wise.")
    if df.duplicated(["subject_id", "ses_idx", "trial"]).any():
        raise ValueError("Rows must be unique by (subject_id, ses_idx, trial).")
    for (_, _), trials in df.groupby(["subject_id", "ses_idx"], sort=False)["trial"]:
        if not np.array_equal(np.sort(trials.to_numpy()), np.arange(len(trials))):
            raise ValueError("Each canonical session must have contiguous zero-based trials.")


def interleaved_session_manifest(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    species: str,
) -> dict[str, object]:
    """Use earliest odd-positioned sessions for adaptation and even for test.

    This matches the existing GRU convention: eval_every_n=2 reserves the
    second, fourth, and later even-positioned sessions, and a K-shot run takes
    the first K sessions from the remaining adaptation sequence.
    """
    # Derive ordering from canonical identifiers rather than incoming row order,
    # so a semantically identical table always produces the same frozen split.
    ordered_df = _deterministically_ordered_rows(df)
    subjects: list[dict[str, object]] = []
    for subject_id, subject_df in ordered_df.groupby("subject_id", sort=False):
        sessions = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
        if len(sessions) < 2:
            raise ValueError(f"Subject {subject_id!r} has fewer than two sessions.")
        subjects.append(
            {
                "subject_id": subject_id,
                "adapt_session_ids": sessions[::2],
                "test_session_ids": sessions[1::2],
            }
        )
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "species": species,
        "split_strategy": "interleaved_sessions_odd_adapt_even_test",
        "subjects": subjects,
    }


def prefix_trial_manifest(
    df: pd.DataFrame,
    *,
    dataset_id: str,
    species: str,
    adapt_fraction: float = 0.5,
) -> dict[str, object]:
    """Create a within-session prefix/suffix split without resetting model state."""
    if not 0.0 < adapt_fraction < 1.0:
        raise ValueError("adapt_fraction must be strictly between zero and one.")
    ordered_df = _deterministically_ordered_rows(df)
    subjects: list[dict[str, object]] = []
    for subject_id, subject_df in ordered_df.groupby("subject_id", sort=False):
        sessions = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
        if len(sessions) != 1:
            raise ValueError(
                f"Prefix split expects exactly one session for {subject_id!r}; got {sessions}."
            )
        n_trials = len(subject_df)
        adapt_trials = int(np.floor(n_trials * adapt_fraction))
        if adapt_trials <= 0 or adapt_trials >= n_trials:
            raise ValueError(f"Cannot split {n_trials} trials for {subject_id!r}.")
        subjects.append(
            {
                "subject_id": subject_id,
                "session_id": sessions[0],
                "adapt_prefix_trials": adapt_trials,
                "total_trials": n_trials,
            }
        )
    return {
        "schema_version": 2,
        "dataset_id": dataset_id,
        "species": species,
        "split_strategy": "within_session_prefix_suffix",
        "adapt_fraction": adapt_fraction,
        "subjects": subjects,
    }


def write_dataset(
    df: pd.DataFrame,
    manifest: dict[str, object],
    *,
    output_root: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    validate_canonical_table(df)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{stem}.parquet"
    manifest_path = output_dir / f"{stem}.split.json"
    df.to_parquet(table_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return table_path, manifest_path
