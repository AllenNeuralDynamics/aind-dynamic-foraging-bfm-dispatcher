"""Freeze paired Study 01 E=4/E=8 source runs for Study 09 transfer."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "analysis"))
from freeze_matched_half import (  # noqa: E402
    ENTITY,
    RUNS_QUERY,
    _artifact,
    _beaker_tasks,
    _job_env,
    _unwrapped,
    _wandb_graphql,
)


MANIFESTS = {
    4: STUDY / "source_runs_e4_pair.json",
    8: STUDY / "source_runs_e8.json",
}
EXPECTED_DISPATCHER = "ab399a50a736add59fcc189d0ac14e2dc6687279"
EXPECTED_WRAPPER = "9595dd371ab87de49c281d8ca4bb6ae8af7c32e4"
EXPECTED_MODELS = "faa0f5ad063e375765aa9c31c7d3fee5eca78ecf"
SOURCE_PROJECT = "mice_data_scaling"


def _wandb_runs(group: str) -> dict[str, dict]:
    data = _wandb_graphql(
        RUNS_QUERY,
        {
            "entity": ENTITY,
            "project": SOURCE_PROJECT,
            "filters": json.dumps({"group": group}),
        },
    )
    nodes = [edge["node"] for edge in data["project"]["runs"]["edges"]]
    return {node["name"]: node for node in nodes}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimension",
        type=int,
        choices=sorted(MANIFESTS),
        action="append",
        help="Freeze only this dimension (repeatable); default: freeze both.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_dimensions = set(args.dimension or MANIFESTS)
    manifests = {dimension: json.loads(path.read_text()) for dimension, path in MANIFESTS.items()}
    groups = {manifest["group"] for manifest in manifests.values()}
    experiments = {
        manifest["source_beaker_experiment_id"] for manifest in manifests.values()
    }
    if len(groups) != 1 or len(experiments) != 1:
        raise AssertionError("E=4 and E=8 must come from one paired source launch")
    group = groups.pop()
    experiment_id = experiments.pop()
    runs = _wandb_runs(group)
    tasks = _beaker_tasks(experiment_id)
    if len(tasks) != 6 or len(runs) != 6:
        raise AssertionError("Paired source launch must contain six Beaker jobs and six W&B runs")

    records = {4: {}, 8: {}}
    for task in tasks:
        job = task["jobs"][-1]
        command = job["execution"]["spec"]["command"]
        dimension = int(
            next(
                item.rsplit("=", 1)[1]
                for item in command
                if item.startswith("model.architecture.subject_embedding_size=")
            )
        )
        if dimension not in selected_dimensions:
            continue
        status = job["status"]
        if "finalized" not in status or status.get("exitCode") != 0:
            raise AssertionError(f"Beaker job {job['id']} is not a successful final result")
        env = _job_env(job)
        if env["DISPATCHER_REF"] != EXPECTED_DISPATCHER:
            raise AssertionError(f"{job['id']} dispatcher ref drifted")
        if env["WRAPPER_REF"] != EXPECTED_WRAPPER:
            raise AssertionError(f"{job['id']} wrapper ref drifted")
        if env["FORAGING_MODELS_REF"] != EXPECTED_MODELS:
            raise AssertionError(f"{job['id']} foraging-models ref drifted")
        run_id = env["WANDB_RUN_ID"]
        node = runs.pop(run_id)
        if node["state"] != "finished":
            raise AssertionError(f"W&B run {run_id} is {node['state']}, not finished")
        config = json.loads(node["config"] or "{}")
        model = _unwrapped(config, "model")
        data = _unwrapped(config, "data")
        architecture = model["architecture"]
        configured_dimension = int(architecture["subject_embedding_size"])
        seed = int(data["seed"])
        if configured_dimension != dimension or seed not in (0, 1, 2):
            raise AssertionError(
                f"Unexpected source cell E={configured_dimension}, seed={seed}"
            )
        if architecture["hidden_size"] != 128 or data["snapshot"] != "20260603":
            raise AssertionError(f"{run_id} source architecture or snapshot drifted")
        if data["subject_ratio"] != 1:
            raise AssertionError(f"{run_id} did not use the full source cohort")
        summary = json.loads(node["summaryMetrics"] or "{}")
        heldout = summary.get("heldout/final/eval_likelihood")
        if heldout is None:
            raise AssertionError(f"{run_id} lacks heldout/final/eval_likelihood")
        artifact = _artifact(node, "gru-output-")
        resolved = _unwrapped(config, "resolved_subject_ids")
        actual_n = len(resolved) if isinstance(resolved, list) else 614
        key = f"e{dimension}-d614-s{seed}"
        if key in records[dimension]:
            raise AssertionError(f"Duplicate source cell {key}")
        records[dimension][key] = {
            "nominal_D": 614,
            "actual_n_source_subjects": actual_n,
            "seed": seed,
            "subject_embedding_size": dimension,
            "run_id": run_id,
            "wandb_url": f"https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/{run_id}",
            "artifact_id": artifact["id"],
            "artifact_digest": artifact["digest"],
            "beaker_job_id": job["id"],
            "beaker_result_dataset_id": job["result"]["beaker"],
            "heldout_final_eval_likelihood": float(heldout),
        }
    if selected_dimensions == set(MANIFESTS) and runs:
        raise AssertionError(f"W&B group has runs absent from Beaker: {sorted(runs)}")

    frozen_at = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
    for dimension in sorted(selected_dimensions):
        path = MANIFESTS[dimension]
        expected = {f"e{dimension}-d614-s{seed}" for seed in range(3)}
        if set(records[dimension]) != expected:
            raise AssertionError(f"E={dimension} does not contain seeds 0, 1, and 2")
        manifest = manifests[dimension]
        manifest["status"] = "complete"
        manifest["frozen_at_pt"] = frozen_at
        manifest["runs"] = dict(sorted(records[dimension].items()))
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
