"""Validate frozen Study 09 tables and summarize adaptation/test membership."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR.parents[1] / "code"))

from external_bandit_datasets.schema import (  # noqa: E402
    interleaved_session_manifest,
    prefix_trial_manifest,
    validate_canonical_table,
)
from external_bandit_datasets.sources import SOURCES  # noqa: E402


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_one(root: Path, name: str) -> dict[str, object]:
    table_path = root / f"{name}.parquet"
    manifest_path = root / f"{name}.split.json"
    audit_path = root / f"{name}.audit.json"
    table = pd.read_parquet(table_path)
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())

    validate_canonical_table(table)
    if set(table["animal_response"].unique()) - {0, 1}:
        raise ValueError(f"{name}: choices are not binary")
    if set(table["rewarded"].unique()) - {0, 1}:
        raise ValueError(f"{name}: rewards are not binary")
    if audit["dataset_id"] != SOURCES[name].dataset_id:
        raise ValueError(f"{name}: audit dataset id does not match the source pin")

    schema_version = int(manifest["schema_version"])
    if schema_version == 1:
        regenerated = interleaved_session_manifest(
            table,
            dataset_id=audit["dataset_id"],
            species=audit["species"],
        )
        session_to_split = {}
        for subject in manifest["subjects"]:
            subject_id = subject["subject_id"]
            for session_id in subject["adapt_session_ids"]:
                session_to_split[(subject_id, session_id)] = "adapt"
            for session_id in subject["test_session_ids"]:
                session_to_split[(subject_id, session_id)] = "test"
        split = table[["subject_id", "ses_idx"]].apply(
            lambda row: session_to_split[(row.iloc[0], row.iloc[1])], axis=1
        )
    elif schema_version == 2:
        regenerated = prefix_trial_manifest(
            table,
            dataset_id=audit["dataset_id"],
            species=audit["species"],
        )
        prefix_by_subject = {
            subject["subject_id"]: int(subject["adapt_prefix_trials"])
            for subject in manifest["subjects"]
        }
        split = table.apply(
            lambda row: (
                "adapt"
                if int(row["trial"]) < prefix_by_subject[row["subject_id"]]
                else "test"
            ),
            axis=1,
        )
    else:
        raise ValueError(f"{name}: unsupported split schema {schema_version}")

    if regenerated != manifest:
        raise ValueError(f"{name}: split manifest is not deterministic")
    if (split == "adapt").sum() == 0 or (split == "test").sum() == 0:
        raise ValueError(f"{name}: empty adaptation or test partition")

    return {
        "dataset": name,
        "dataset_id": audit["dataset_id"],
        "species": audit["species"],
        "schema_version": schema_version,
        "split_strategy": manifest["split_strategy"],
        "num_subjects": audit["num_subjects"],
        "num_sessions": audit["num_sessions"],
        "num_trials": audit["num_trials"],
        "num_adapt_trials": int((split == "adapt").sum()),
        "num_test_trials": int((split == "test").sum()),
        "excluded_trials": audit["excluded_trials"],
        "table_sha256": _digest(table_path),
        "manifest_sha256": _digest(manifest_path),
        "audit_sha256": _digest(audit_path),
        "source_digest": audit["verified_source_digest"],
        "source_digest_algorithm": audit["source"]["digest_algorithm"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=STUDY_DIR / "data-cache" / "canonical",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "_meta": {
            "schema_version": 1,
            "producer": "analysis/validate_dataset_suite.py",
            "validation": [
                "exact source-release counts",
                "binary choices and rewards",
                "schema v1/v2 only",
                "deterministic manifest regeneration",
                "non-empty adaptation and test partitions",
            ],
        },
        "datasets": [_validate_one(args.canonical_root, name) for name in SOURCES],
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
