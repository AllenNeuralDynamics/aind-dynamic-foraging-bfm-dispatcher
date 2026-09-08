#!/usr/bin/env python
"""Freeze paired trained-GRU and reservoir results from pinned W&B runs."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import wandb


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402


RESERVOIR_PROJECT = "AIND-disRNN/gru_reservoir_ablation"
SOURCE_PROJECT = "AIND-disRNN/mice_data_scaling"
# Hard allowlist. Add the seeds-1/2 group after that launch; never discover by project scan.
WANDB_GROUPS = ("frozen-random-core-d614@20260907-175533",)
REFERENCE = STUDY / "reference" / "study01-trained-gru.json"
OUTPUT = STUDY / "analysis" / "reservoir_results.json"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_SUBJECTS = 149


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"value"}:
        return _unwrap(value["value"])
    return value


def _seed(run: Any) -> int:
    value = _unwrap(run.config.get("seed"))
    if value is None:
        raise ValueError(f"run {run.id} has no seed")
    return int(value)


def _config_value(run: Any, *path: str) -> Any:
    value: Any = run.config
    for key in path:
        value = _unwrap(value)
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"run {run.id} config has no {'.'.join(path)}")
        value = value[key]
    return _unwrap(value)


def _validate_reservoir_config(run: Any) -> None:
    expected = {
        ("model", "architecture", "hidden_size"): 128,
        ("model", "architecture", "subject_embedding_size"): 4,
        ("model", "training", "freeze_gru_core"): True,
        ("model", "training", "auto_heldout_finetune", "n_steps"): 500,
        ("model", "training", "auto_heldout_finetune", "lr"): 0.001,
        ("data", "snapshot"): "20260603",
    }
    for path, expected_value in expected.items():
        actual = _config_value(run, *path)
        if path == ("data", "snapshot"):
            actual = str(actual)
        if actual != expected_value:
            raise ValueError(
                f"run {run.id} config {'.'.join(path)}={actual!r}, expected {expected_value!r}"
            )
    subject_ids = _unwrap(run.config.get("resolved_subject_ids")) or []
    if len(subject_ids) != 614:
        raise ValueError(f"run {run.id} resolved D={len(subject_ids)}, expected 614")


def _per_subject_table(run: Any) -> Any:
    for artifact in run.logged_artifacts():
        if artifact.type != "run_table":
            continue
        for entry_name in artifact.manifest.entries:
            if "per_subject_likelihood" in str(entry_name):
                return artifact.get(entry_name).get_dataframe()
    raise ValueError(f"run {run.id} has no held-out per-subject likelihood table")


def _subject_rows(run: Any) -> list[dict[str, Any]]:
    frame = _per_subject_table(run)
    required = {"heldout_subject_id", "n_trials", "eval_likelihood"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"run {run.id} table missing columns: {sorted(missing)}")
    rows = [
        {
            "subject_id": str(row.heldout_subject_id),
            "n_trials": int(row.n_trials),
            "likelihood": float(row.eval_likelihood),
        }
        for row in frame.itertuples(index=False)
    ]
    rows.sort(key=lambda row: row["subject_id"])
    if len(rows) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"run {run.id} has {len(rows)} held-out subjects, expected {EXPECTED_SUBJECTS}"
        )
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError(f"run {run.id} has duplicate held-out subject ids")
    return rows


def _pooled_likelihood(rows: list[dict[str, Any]]) -> float:
    total_trials = sum(row["n_trials"] for row in rows)
    return math.exp(
        sum(row["n_trials"] * math.log(row["likelihood"]) for row in rows)
        / total_trials
    )


def _training_artifact(run: Any) -> Any:
    candidates = [
        artifact
        for artifact in run.logged_artifacts()
        if artifact.type == "training-output"
    ]
    if len(candidates) != 1:
        raise ValueError(f"run {run.id} has {len(candidates)} training-output artifacts")
    return candidates[0]


def _download_entry(artifact: Any, suffix: str, root: Path) -> Path:
    names = [str(name) for name in artifact.manifest.entries]
    exact = [name for name in names if name == suffix]
    matches = exact or [name for name in names if name.endswith("/" + suffix)]
    if not exact and matches:
        shallowest = min(name.count("/") for name in matches)
        matches = [name for name in matches if name.count("/") == shallowest]
    if len(matches) != 1:
        raise ValueError(
            f"artifact {artifact.name} has {len(matches)} entries ending in {suffix!r}"
        )
    return Path(artifact.get_path(matches[0]).download(root=str(root)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tree_sha256(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _initial_tree_sha256(run: Any) -> str:
    artifact = _training_artifact(run)
    with tempfile.TemporaryDirectory(prefix=f"study10-source-{run.id}-") as temp_dir:
        path = _download_entry(
            artifact,
            "initialization/before_training/params.json",
            Path(temp_dir),
        )
        return _canonical_tree_sha256(json.loads(path.read_text()))


def _frozen_parameter_audit(run: Any) -> dict[str, Any]:
    artifact = _training_artifact(run)
    with tempfile.TemporaryDirectory(prefix=f"study10-{run.id}-") as temp_dir:
        root = Path(temp_dir)
        initial_path = _download_entry(
            artifact, "initialization/before_training/params.json", root
        )
        final_path = _download_entry(artifact, "params.json", root)
        initial = json.loads(initial_path.read_text())
        final = json.loads(final_path.read_text())
        frozen_modules = sorted(
            module
            for module in initial
            if module != "multisubject_gru" and not module.endswith("/~/readout")
        )
        if not any(module.endswith("/~/gru") for module in frozen_modules):
            raise ValueError(f"run {run.id} has no GRU module in its parameter tree")
        changed = [module for module in frozen_modules if initial[module] != final[module]]
        if changed:
            raise ValueError(f"run {run.id} changed frozen modules: {changed}")
        if initial["multisubject_gru"]["subject_embeddings"] == final[
            "multisubject_gru"
        ]["subject_embeddings"]:
            raise ValueError(f"run {run.id} did not update subject embeddings")
        readout = next(module for module in initial if module.endswith("/~/readout"))
        if initial[readout] == final[readout]:
            raise ValueError(f"run {run.id} did not update the readout")
        return {
            "passed": True,
            "frozen_modules": frozen_modules,
            "initial_params_sha256": _sha256(initial_path),
            "final_params_sha256": _sha256(final_path),
            "initial_tree_sha256": _canonical_tree_sha256(initial),
        }


def _freeze_run(run: Any, *, audit_frozen: bool) -> dict[str, Any]:
    rows = _subject_rows(run)
    pooled = _pooled_likelihood(rows)
    summary_value = float(run.summary["heldout/final/eval_likelihood"])
    if not math.isclose(pooled, summary_value, rel_tol=0, abs_tol=2e-6):
        raise ValueError(
            f"run {run.id} pooled likelihood {pooled} != summary {summary_value}"
        )
    artifact = _training_artifact(run)
    result = {
        "seed": _seed(run),
        "wandb_run_id": run.id,
        "wandb_url": run.url,
        "artifact_name": artifact.name,
        "artifact_digest": artifact.digest,
        "heldout_likelihood": summary_value,
        "pooled_likelihood_from_subjects": pooled,
        "training_steps_completed": int(run.summary["training_steps_completed"]),
        "subjects": rows,
    }
    if audit_frozen:
        result["frozen_parameter_audit"] = _frozen_parameter_audit(run)
    return result


def main() -> None:
    if len(WANDB_GROUPS) < 2:
        raise RuntimeError(
            "Add the pinned seeds-1/2 W&B group to WANDB_GROUPS after its launch."
        )
    api = wandb.Api()
    reservoir_runs = []
    for group in WANDB_GROUPS:
        reservoir_runs.extend(
            api.runs(RESERVOIR_PROJECT, filters={"group": group, "state": "finished"})
        )
    reservoir_by_seed = {_seed(run): run for run in reservoir_runs}
    if tuple(sorted(reservoir_by_seed)) != EXPECTED_SEEDS:
        raise ValueError(f"reservoir seed coverage is {sorted(reservoir_by_seed)}")
    for run in reservoir_by_seed.values():
        _validate_reservoir_config(run)

    reference = json.loads(REFERENCE.read_text())
    source_cells = {cell["seed"]: cell for cell in reference["cells"] if cell["D"] == 614}
    source_runs = {
        seed: api.run(f"{SOURCE_PROJECT}/{source_cells[seed]['wandb_run_id']}")
        for seed in EXPECTED_SEEDS
    }

    reservoir = [
        _freeze_run(reservoir_by_seed[seed], audit_frozen=True)
        for seed in EXPECTED_SEEDS
    ]
    trained = [_freeze_run(source_runs[seed], audit_frozen=False) for seed in EXPECTED_SEEDS]
    for seed, reservoir_result, trained_result in zip(EXPECTED_SEEDS, reservoir, trained):
        if trained_result["artifact_digest"] != source_cells[seed]["artifact_digest"]:
            raise ValueError(f"seed {seed} trained-GRU artifact digest changed")
        trained_initial_tree_sha256 = _initial_tree_sha256(source_runs[seed])
        trained_result["initial_tree_sha256"] = trained_initial_tree_sha256
        if (
            reservoir_result["frozen_parameter_audit"]["initial_tree_sha256"]
            != trained_initial_tree_sha256
        ):
            raise ValueError(f"seed {seed} native initial parameter trees do not match")
        reservoir_ids = [row["subject_id"] for row in reservoir_result["subjects"]]
        trained_ids = [row["subject_id"] for row in trained_result["subjects"]]
        if reservoir_ids != trained_ids:
            raise ValueError(f"seed {seed} held-out subject keys do not match")
        reservoir_trials = [row["n_trials"] for row in reservoir_result["subjects"]]
        trained_trials = [row["n_trials"] for row in trained_result["subjects"]]
        if reservoir_trials != trained_trials:
            raise ValueError(f"seed {seed} held-out subject trial counts do not match")

    output = {
        "_meta": build_meta(
            "analysis/freeze_results.py", list(WANDB_GROUPS), study_root=STUDY
        ),
        "reservoir": reservoir,
        "trained_gru": trained,
        "parity": {
            "seeds": list(EXPECTED_SEEDS),
            "subjects_per_seed": EXPECTED_SUBJECTS,
            "subject_keys_equal": True,
            "subject_trial_counts_equal": True,
            "native_initial_parameter_trees_equal": True,
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
