"""Freeze paired Study 01 E=4/E=8 source metrics and embedding spectra."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402

sys.path.insert(
    0,
    str(REPO / "studies" / "09-gru-cross-species-transfer" / "analysis"),
)
from freeze_matched_half import (  # noqa: E402
    ARTIFACT_FILES_QUERY,
    ENTITY,
    RUNS_QUERY,
    _artifact,
    _unwrapped,
    _wandb_graphql,
)


PROJECT = "mice_data_scaling"
GROUP = "e4-e8-d614-source@20260906-195409"
EXPERIMENT_ID = "01M1WWK9HP269XQS440Y29631F"
EXPECTED_REFS = {
    "DISPATCHER_REF": "ab399a50a736add59fcc189d0ac14e2dc6687279",
    "WRAPPER_REF": "9595dd371ab87de49c281d8ca4bb6ae8af7c32e4",
    "FORAGING_MODELS_REF": "faa0f5ad063e375765aa9c31c7d3fee5eca78ecf",
}
OUTPUT = STUDY / "analysis" / "embedding_dimension_results.json"


def _wandb_runs() -> dict[str, dict]:
    data = _wandb_graphql(
        RUNS_QUERY,
        {
            "entity": ENTITY,
            "project": PROJECT,
            "filters": json.dumps({"group": GROUP}),
        },
    )
    nodes = [edge["node"] for edge in data["project"]["runs"]["edges"]]
    return {node["name"]: node for node in nodes}


def _beaker_tasks() -> list[dict]:
    result = subprocess.run(
        ["beaker", "experiment", "tasks", EXPERIMENT_ID, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout)


def _job_env(job: dict) -> dict[str, str]:
    return {
        item["name"]: item["value"]
        for item in job["execution"]["spec"]["envVars"]
        if item.get("value") is not None
    }


def _artifact_file(artifact_id: str, name: str) -> bytes:
    data = _wandb_graphql(ARTIFACT_FILES_QUERY, {"id": artifact_id})
    matches = [
        edge["node"]
        for edge in data["artifact"]["files"]["edges"]
        if edge["node"]["name"] == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"Artifact {artifact_id} has {len(matches)} files named {name}")
    response = requests.get(matches[0]["directUrl"], timeout=60)
    response.raise_for_status()
    return response.content


def _per_subject(blob: bytes, pooled_likelihood: float) -> dict[str, dict]:
    table = json.loads(blob)
    if table.get("columns") != [
        "heldout_subject_id",
        "n_trials",
        "eval_likelihood",
    ]:
        raise AssertionError("Held-out per-subject table schema drifted")
    rows = {
        str(subject_id): {
            "n_trials": int(n_trials),
            "eval_likelihood": float(likelihood),
        }
        for subject_id, n_trials, likelihood in table["data"]
    }
    if len(rows) != 149 or any(row["n_trials"] <= 0 for row in rows.values()):
        raise AssertionError("Expected 149 unique held-out mice with positive trial counts")
    total_trials = sum(row["n_trials"] for row in rows.values())
    reconstructed = math.exp(
        sum(
            row["n_trials"] * math.log(row["eval_likelihood"])
            for row in rows.values()
        )
        / total_trials
    )
    if not math.isclose(reconstructed, pooled_likelihood, rel_tol=0, abs_tol=1e-7):
        raise AssertionError("Per-subject table does not reconstruct pooled likelihood")
    return dict(sorted(rows.items()))


def _embedding_spectrum(values: np.ndarray) -> dict:
    eigenvalues = np.linalg.eigvalsh(np.cov(values, rowvar=False))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    ratios = eigenvalues / eigenvalues.sum()
    effective_rank = math.exp(-sum(value * math.log(value) for value in ratios if value > 0))
    return {
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_explained_variance": np.cumsum(ratios).tolist(),
        "effective_rank": effective_rank,
        "pc5_to_pc8_variance_fraction": (
            float(ratios[4:].sum()) if values.shape[1] == 8 else None
        ),
    }


def _best_eval_embeddings(
    artifact_id: str, dimension: int, subject_ids: set[str]
) -> tuple[dict, dict]:
    index_blob = _artifact_file(artifact_id, "checkpoints/index.json")
    index = json.loads(index_blob)
    checkpoints = index.get("checkpoints", [])
    if not checkpoints:
        raise AssertionError("Source artifact has no checkpoints for best_eval selection")
    selected = max(
        checkpoints,
        key=lambda row: float(row.get("eval_likelihood", float("-inf"))),
    )
    params_name = "/".join(Path(selected["params_path"]).parts[-3:])
    params_blob = _artifact_file(artifact_id, params_name)
    params = json.loads(params_blob)
    values = np.asarray(
        params["multisubject_gru"]["subject_embeddings"], dtype=float
    )
    if values.shape != (614, dimension):
        raise AssertionError(
            f"Best-eval embedding matrix is {values.shape}, expected (614, {dimension})"
        )

    map_blob = _artifact_file(artifact_id, "subject_index_map.json")
    subject_map = json.loads(map_blob)["subject_id_to_index"]
    if set(subject_map) != subject_ids or set(subject_map.values()) != set(range(614)):
        raise AssertionError("Source subject-index map does not match the resolved cohort")
    return _embedding_spectrum(values), {
        "policy": "best_eval",
        "step": int(selected["step"]),
        "eval_likelihood": float(selected["eval_likelihood"]),
        "params_path": params_name,
        "checkpoint_index_sha256": hashlib.sha256(index_blob).hexdigest(),
        "params_sha256": hashlib.sha256(params_blob).hexdigest(),
        "subject_index_map_sha256": hashlib.sha256(map_blob).hexdigest(),
    }


def main() -> None:
    runs = _wandb_runs()
    tasks = _beaker_tasks()
    if len(runs) != 6 or len(tasks) != 6:
        raise AssertionError("Paired source launch must contain six W&B runs and Beaker tasks")

    dimensions = {"e4": [], "e8": []}
    subject_signature = None
    for task in tasks:
        job = task["jobs"][-1]
        status = job["status"]
        if "finalized" not in status or status.get("exitCode") != 0:
            raise AssertionError(f"Beaker job {job['id']} is not a successful final result")
        env = _job_env(job)
        for name, expected in EXPECTED_REFS.items():
            if env[name] != expected:
                raise AssertionError(f"{job['id']} {name} drifted")
        run_id = env["WANDB_RUN_ID"]
        node = runs.pop(run_id)
        if node["state"] != "finished":
            raise AssertionError(f"W&B run {run_id} is {node['state']}, not finished")

        config = json.loads(node["config"] or "{}")
        model = _unwrapped(config, "model")
        data = _unwrapped(config, "data")
        architecture = model["architecture"]
        dimension = int(architecture["subject_embedding_size"])
        seed = int(data["seed"])
        if dimension not in (4, 8) or seed not in (0, 1, 2):
            raise AssertionError(f"Unexpected source cell E={dimension}, seed={seed}")
        if architecture["hidden_size"] != 128 or data["snapshot"] != "20260603":
            raise AssertionError(f"{run_id} source architecture or snapshot drifted")
        if data["subject_ratio"] != 1:
            raise AssertionError(f"{run_id} did not use the full source cohort")
        resolved_subject_ids = _unwrapped(config, "resolved_subject_ids")
        source_subject_ids = {str(value) for value in resolved_subject_ids}
        if len(source_subject_ids) != 614:
            raise AssertionError(f"{run_id} did not resolve exactly 614 source mice")

        summary = json.loads(node["summaryMetrics"] or "{}")
        eval_likelihood = float(summary["heldout/final/eval_likelihood"])
        train_likelihood = float(summary["heldout/final/train_likelihood"])
        training_artifact = _artifact(node, "gru-output-")
        table_artifact = _artifact(
            node, f"run-{run_id}-heldoutper_subject_likelihood-"
        )
        table_blob = _artifact_file(
            table_artifact["id"], "heldout/per_subject_likelihood.table.json"
        )
        embedding_spectrum, embedding_checkpoint = _best_eval_embeddings(
            training_artifact["id"], dimension, source_subject_ids
        )
        per_subject = _per_subject(table_blob, eval_likelihood)
        signature = tuple(
            (subject_id, row["n_trials"]) for subject_id, row in per_subject.items()
        )
        if subject_signature is None:
            subject_signature = signature
        elif signature != subject_signature:
            raise AssertionError("Held-out subject IDs or trial counts differ across source cells")
        result_dataset_id = job.get("result", {}).get("beaker")
        if not result_dataset_id:
            raise AssertionError(f"Beaker job {job['id']} has no result dataset")

        dimensions[f"e{dimension}"].append(
            {
                "seed": seed,
                "run_id": run_id,
                "wandb_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}",
                "beaker_job_id": job["id"],
                "beaker_result_dataset_id": result_dataset_id,
                "source_artifact": training_artifact,
                "heldout_table_artifact": table_artifact,
                "heldout_table_sha256": hashlib.sha256(table_blob).hexdigest(),
                "heldout": {
                    "adaptation_likelihood": train_likelihood,
                    "test_likelihood": eval_likelihood,
                    "test_minus_adaptation": eval_likelihood - train_likelihood,
                    "per_subject_test_likelihood": per_subject,
                },
                "source_embedding_checkpoint": embedding_checkpoint,
                "source_embedding_spectrum": embedding_spectrum,
            }
        )

    if runs:
        raise AssertionError(f"W&B group has runs absent from Beaker: {sorted(runs)}")
    for name, rows in dimensions.items():
        rows.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in rows] != [0, 1, 2]:
            raise AssertionError(f"{name} does not contain seeds 0, 1, and 2")

    output = {
        "_meta": build_meta(
            "analysis/freeze_embedding_dimension.py", [GROUP], study_root=STUDY
        ),
        "status": "complete",
        "frozen_at_pt": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
            timespec="seconds"
        ),
        "wandb_project": f"https://wandb.ai/{ENTITY}/{PROJECT}",
        "wandb_group": GROUP,
        "beaker_experiment_id": EXPERIMENT_ID,
        "contract": {
            "source_D": 614,
            "source_hidden_size": 128,
            "embedding_dimensions": [4, 8],
            "source_seeds": [0, 1, 2],
            "source_snapshot": "20260603",
            "n_heldout_subjects": len(subject_signature or ()),
            "exact_heldout_subject_and_trial_count_parity": True,
        },
        "dimensions": dimensions,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
