"""Freeze E=8 transfer results for the seven expansion cohorts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402

sys.path.insert(0, str(STUDY / "analysis"))
from freeze_matched_half import (  # noqa: E402
    ENTITY,
    PROJECT,
    _artifact,
    _beaker_tasks,
    _cached_wandb_report_files,
    _job_env,
    _prediction_summary,
    _report_metrics,
    _unwrapped,
    _wandb_runs,
)


DATASET_ORDER = (
    "zid",
    "chen",
    "beron",
    "lopez_mouse",
    "alsio",
    "costa",
    "tang",
)
VARIANT = STUDY / "variants" / "gru-e8-d614-expansion"
LAUNCHES = (
    (VARIANT / "launch_record_primary", 9, {"zid", "chen", "beron"}),
    (
        VARIANT / "launch_record_boundary",
        12,
        {"lopez_mouse", "alsio", "costa", "tang"},
    ),
)
VALIDATION = STUDY / "analysis" / "dataset_suite_validation.json"
HISTORICAL = STUDY / "analysis" / "matched_half_results.json"
MANIFEST = STUDY / "source_runs_e8.json"
CACHE = STUDY / "analysis" / "_cache_embedding_dimension" / "expansion"
OUTPUT = STUDY / "analysis" / "embedding_dimension_expansion_results.json"


def _source_manifest() -> tuple[dict, str]:
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("status") != "complete":
        raise AssertionError("source_runs_e8.json is not complete")
    if manifest["architecture"]["subject_embedding_size"] != 8:
        raise AssertionError("source_runs_e8.json does not declare E=8")
    expected = {f"e8-d614-s{seed}" for seed in range(3)}
    if set(manifest["runs"]) != expected:
        raise AssertionError("source_runs_e8.json does not contain exactly three seeds")
    return manifest, hashlib.sha256(raw).hexdigest()


def _launch(
    root: Path,
    expected_tasks: int,
    expected_datasets: set[str],
    manifest: dict,
    manifest_digest: str,
) -> tuple[str, str, dict[str, list[dict]]]:
    record = json.loads((root / "beaker_resumable.json").read_text())
    spec = yaml.safe_load((root / "experiment_resumable_submitted.yaml").read_text())
    groups = {
        _job_env({"execution": {"spec": task}})["WANDB_RUN_GROUP"]
        for task in spec["tasks"]
    }
    if len(spec["tasks"]) != expected_tasks or len(groups) != 1:
        raise AssertionError(f"{root.name} has the wrong task count or group count")
    group = groups.pop()
    if group != record["wandb_group"]:
        raise AssertionError(f"{root.name} group disagrees with its launch record")

    runs = _wandb_runs(group)
    tasks = _beaker_tasks(record["experiment_id"])
    if len(tasks) != expected_tasks or len(runs) != expected_tasks:
        raise AssertionError(f"{root.name} is not a complete {expected_tasks}-cell launch")
    records = {name: [] for name in expected_datasets}
    for task in tasks:
        job = task["jobs"][-1]
        status = job["status"]
        if "finalized" not in status or status.get("exitCode") != 0:
            raise AssertionError(f"Beaker job {job['id']} is not a successful final result")
        env = _job_env(job)
        run_id = env["WANDB_RUN_ID"]
        node = runs.pop(run_id)
        if node["state"] != "finished":
            raise AssertionError(f"W&B run {run_id} is {node['state']}, not finished")
        config = json.loads(node["config"] or "{}")
        source = _unwrapped(config, "source")
        target = _unwrapped(config, "target")
        dataset = target.get("dataset")
        if dataset not in records:
            raise AssertionError(f"Unexpected target dataset {dataset!r}")
        if source.get("manifest") != MANIFEST.name:
            raise AssertionError(f"{run_id} used {source.get('manifest')!r}")
        if source.get("manifest_sha256") != manifest_digest:
            raise AssertionError(f"{run_id} source-manifest digest mismatch")
        source_key = source["key"]
        declared = manifest["runs"][source_key]
        if source["run_id"] != declared["run_id"]:
            raise AssertionError(f"{run_id} source-run mismatch")
        if source["artifact_digest"] != declared["artifact_digest"]:
            raise AssertionError(f"{run_id} source-artifact mismatch")

        artifact = _artifact(node, "external-gru-output-")
        files = _cached_wandb_report_files(
            artifact, CACHE / dataset / source_key
        )
        metrics_bytes = files["test_metrics.json"]
        predictions_bytes = files["test_trial_predictions.csv"]
        trial_digest, n_rows, per_session = _prediction_summary(predictions_bytes)
        metrics = json.loads(metrics_bytes)
        summary = json.loads(node["summaryMetrics"] or "{}")
        if not abs(
            metrics["normalized_likelihood"]
            - summary["target/test/normalized_likelihood"]
        ) < 1e-10:
            raise AssertionError(f"W&B/file likelihood mismatch for {run_id}")
        records[dataset].append(
            {
                "source_key": source_key,
                "seed": int(source["seed"]),
                "wandb_run_id": run_id,
                "wandb_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}",
                "evaluation_artifact": artifact,
                "beaker_job_id": job["id"],
                "beaker_result_dataset_id": job["result"]["beaker"],
                "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
                "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
                "ordered_trial_key_sha256": trial_digest,
                "n_prediction_rows": n_rows,
                "metrics": _report_metrics(metrics, per_session),
            }
        )
    if runs:
        raise AssertionError(f"W&B group has runs absent from Beaker: {sorted(runs)}")
    for dataset, rows in records.items():
        rows.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != [0, 1, 2]:
            raise AssertionError(f"E=8 {dataset} lacks three source seeds")
    return group, record["experiment_id"], records


def main() -> None:
    manifest, manifest_digest = _source_manifest()
    groups = []
    experiments = []
    e8 = {}
    for root, expected_tasks, expected_datasets in LAUNCHES:
        group, experiment, records = _launch(
            root,
            expected_tasks,
            expected_datasets,
            manifest,
            manifest_digest,
        )
        groups.append(group)
        experiments.append(experiment)
        e8.update(records)
    if set(e8) != set(DATASET_ORDER):
        raise AssertionError("Expansion launches do not cover the expected datasets")

    validation = {
        row["dataset"]: row
        for row in json.loads(VALIDATION.read_text())["datasets"]
        if row["dataset"] in DATASET_ORDER
    }
    historical = json.loads(HISTORICAL.read_text())["datasets"]
    datasets = {}
    for name in DATASET_ORDER:
        old_d614 = [
            row
            for row in historical[name]["gru"]
            if int(row["nominal_D"]) == 614
        ]
        signatures = {
            (row["ordered_trial_key_sha256"], row["n_prediction_rows"])
            for row in e8[name] + old_d614
        }
        if len(signatures) != 1:
            raise AssertionError(f"E=8/historical trial keys differ for {name}")
        audit = validation[name]
        datasets[name] = {
            "species": audit["species"],
            "schema_version": audit["schema_version"],
            "split_strategy": audit["split_strategy"],
            "n_subjects": audit["num_subjects"],
            "n_sessions": audit["num_sessions"],
            "n_heldout_trials": audit["num_test_trials"],
            "ordered_trial_key_sha256": next(iter(signatures))[0],
            "e8": e8[name],
        }
    output = {
        "_meta": build_meta(
            "analysis/freeze_embedding_dimension_expansion.py",
            groups,
            study_root=STUDY,
        ),
        "status": "complete",
        "contract": {
            "source_D": 614,
            "source_hidden_size": 128,
            "subject_embedding_size": 8,
            "source_seeds": [0, 1, 2],
            "target_embedding_steps": 500,
            "target_embedding_lr": 0.001,
            "selection_policy": "fixed_final",
            "historical_e4_trial_key_parity": True,
        },
        "wandb_project": f"https://wandb.ai/{ENTITY}/{PROJECT}",
        "beaker_experiments": experiments,
        "datasets": datasets,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
