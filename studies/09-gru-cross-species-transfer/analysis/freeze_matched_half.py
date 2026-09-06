"""Freeze matched-half GRU and Q results from pinned W&B/Beaker launches."""

from __future__ import annotations

import csv
import hashlib
import json
import netrc
import os
import sys
from pathlib import Path

import requests


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402


ENTITY = "AIND-disRNN"
PROJECT = "gru_cross_species_transfer"
WANDB_GROUPS = [
    "gru-grossman-matched-half@20260905-022602",
    "gru-chen-matched-half@20260905-024731",
    "gru-zid-matched-half@20260905-025752",
    "q-matched-half@20260905-024031",
]
GRU_LAUNCHES = {
    "grossman": (WANDB_GROUPS[0], "01M1RE7RE42MHTHFDDRYJWTWHV"),
    "chen": (WANDB_GROUPS[1], "01M1RFF0YVREC2924A9Z2Y13XF"),
    "zid": (WANDB_GROUPS[2], "01M1RG1X3W0VK8ZBYQ8ZB4V4BR"),
}
Q_GROUP = WANDB_GROUPS[3]
Q_SLURM_ARRAY_JOB_ID = "25580070"
CACHE = STUDY / "analysis" / "_cache_matched_half"
OUTPUT = STUDY / "analysis" / "matched_half_results.json"

RUNS_QUERY = """query Runs($entity:String!,$project:String!,$filters:JSONString){
  project(name:$project,entityName:$entity){
    runs(filters:$filters,first:100){edges{node{
      name displayName state group config summaryMetrics
      outputArtifacts(first:30){edges{node{
        id state digest fileCount size artifactSequence{name}
      }}}
    }}}
  }
}"""
ARTIFACT_FILES_QUERY = """query ArtifactFiles($id:ID!){
  artifact(id:$id){files(first:1000){edges{node{name directUrl}}}}
}"""


def _wandb_key() -> str:
    if key := os.environ.get("WANDB_API_KEY"):
        return key
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError) as error:
        raise RuntimeError(
            "WANDB_API_KEY is unset and ~/.netrc could not be read"
        ) from error
    if not credentials or not credentials[2]:
        raise RuntimeError("WANDB_API_KEY is unset and api.wandb.ai is absent from ~/.netrc")
    return credentials[2]


def _wandb_graphql(query: str, variables: dict) -> dict:
    response = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", _wandb_key()),
        json={"query": query, "variables": variables},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload["data"]


def _wandb_runs(group: str) -> dict[str, dict]:
    data = _wandb_graphql(
        RUNS_QUERY,
        {
            "entity": ENTITY,
            "project": PROJECT,
            "filters": json.dumps({"group": group}),
        },
    )
    nodes = [edge["node"] for edge in data["project"]["runs"]["edges"]]
    return {node["name"]: node for node in nodes}


def _unwrapped(config: dict, key: str) -> dict:
    value = config.get(key, {})
    return value.get("value", value) if isinstance(value, dict) else {}


def _artifact(node: dict, prefix: str) -> dict:
    matches = [
        edge["node"]
        for edge in node["outputArtifacts"]["edges"]
        if edge["node"]["artifactSequence"]["name"].startswith(prefix)
    ]
    if len(matches) != 1 or matches[0]["state"] != "COMMITTED":
        raise AssertionError(f"Expected one committed {prefix!r} artifact for {node['name']}")
    artifact = matches[0]
    return {
        "id": artifact["id"],
        "name": artifact["artifactSequence"]["name"],
        "digest": artifact["digest"],
        "file_count": artifact["fileCount"],
        "size_bytes": artifact["size"],
    }


def _cached_wandb_report_files(artifact: dict, destination: Path) -> dict[str, bytes]:
    data = _wandb_graphql(ARTIFACT_FILES_QUERY, {"id": artifact["id"]})
    files = [edge["node"] for edge in data["artifact"]["files"]["edges"]]
    selected = {
        suffix: [item for item in files if item["name"].endswith(suffix)]
        for suffix in ("test_metrics.json", "test_trial_predictions.csv")
    }
    if any(len(matches) != 1 for matches in selected.values()):
        raise AssertionError(f"W&B artifact {artifact['name']} has ambiguous report files")
    output = {}
    for suffix, matches in selected.items():
        path = destination / suffix
        if not path.exists():
            response = requests.get(matches[0]["directUrl"], timeout=300)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        output[suffix] = path.read_bytes()
    return output


def _trial_key_digest(data: bytes) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    text = data.decode("utf-8").splitlines()
    for row in csv.DictReader(text):
        digest.update(
            f"{row['subject_id']}\t{row['ses_idx']}\t{row['trial']}\t{row['choice']}\n".encode()
        )
        rows += 1
    return digest.hexdigest(), rows


def _report_metrics(metrics: dict) -> dict:
    """Keep only the likelihood values consumed by the committed report."""
    return {
        "n_trials": metrics["n_trials"],
        "mean_log_likelihood_nats": metrics["mean_log_likelihood_nats"],
        "normalized_likelihood": metrics["normalized_likelihood"],
        "per_subject_mean_log_likelihood_nats": {
            item["subject_id"]: item["mean_log_likelihood_nats"]
            for item in metrics["per_subject"]
        },
    }


def _cached_beaker_file(beaker, dataset, file_info, destination: Path) -> bytes:
    if destination.exists():
        data = destination.read_bytes()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = beaker.dataset.get_file(dataset, file_info, quiet=True)
        destination.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if digest != file_info.digest.value:
        raise AssertionError(f"Cached file digest mismatch for {file_info.path}")
    return data


def _job_env(job) -> dict[str, str]:
    return {
        item.name: item.value
        for item in job.execution.spec.env_vars
        if item.value is not None
    }


def _freeze_gru(dataset_name: str, group: str, experiment_id: str) -> list[dict]:
    from beaker import Beaker

    beaker = Beaker.from_env(check_for_upgrades=False, default_org="ai1")
    runs = _wandb_runs(group)
    experiment = beaker.experiment.get(experiment_id)
    records = []
    for job in experiment.jobs:
        if str(job.status.current) != "finalized" or job.status.exit_code != 0:
            raise AssertionError(f"Beaker job {job.id} is not a successful final result")
        env = _job_env(job)
        run_id = env["WANDB_RUN_ID"]
        node = runs.pop(run_id)
        if node["state"] != "finished":
            raise AssertionError(f"W&B run {run_id} is {node['state']}, not finished")
        config = json.loads(node["config"] or "{}")
        source = _unwrapped(config, "source")
        target = _unwrapped(config, "target")
        if target.get("dataset") != dataset_name:
            raise AssertionError(f"W&B run {run_id} targets {target.get('dataset')!r}")

        result_dataset = beaker.job.results(job)
        if result_dataset is None:
            raise AssertionError(f"Beaker job {job.id} has no result dataset")
        files = beaker.dataset.ls(result_dataset)
        selected = {
            suffix: [item for item in files if item.path.endswith(suffix)]
            for suffix in ("test_metrics.json", "test_trial_predictions.csv")
        }
        if any(len(matches) != 1 for matches in selected.values()):
            raise AssertionError(f"Beaker result {result_dataset.id} has ambiguous report files")
        source_key = source["key"]
        metrics_bytes = _cached_beaker_file(
            beaker,
            result_dataset,
            selected["test_metrics.json"][0],
            CACHE / "gru" / dataset_name / source_key / "test_metrics.json",
        )
        predictions_bytes = _cached_beaker_file(
            beaker,
            result_dataset,
            selected["test_trial_predictions.csv"][0],
            CACHE / "gru" / dataset_name / source_key / "test_trial_predictions.csv",
        )
        trial_digest, n_prediction_rows = _trial_key_digest(predictions_bytes)
        metrics = json.loads(metrics_bytes)
        summary = json.loads(node["summaryMetrics"] or "{}")
        if not abs(
            metrics["normalized_likelihood"]
            - summary["target/test/normalized_likelihood"]
        ) < 1e-10:
            raise AssertionError(f"W&B/file likelihood mismatch for {run_id}")
        records.append(
            {
                "source_key": source_key,
                "nominal_D": source["nominal_D"],
                "actual_D": source["actual_n_source_subjects"],
                "seed": source["seed"],
                "wandb_run_id": run_id,
                "wandb_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}",
                "evaluation_artifact": _artifact(node, "external-gru-output-"),
                "beaker_job_id": job.id,
                "beaker_result_dataset_id": result_dataset.id,
                "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
                "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
                "ordered_trial_key_sha256": trial_digest,
                "n_prediction_rows": n_prediction_rows,
                "metrics": _report_metrics(metrics),
            }
        )
    if runs:
        raise AssertionError(f"W&B group {group} has runs absent from Beaker: {sorted(runs)}")
    return sorted(records, key=lambda row: (row["nominal_D"], row["seed"]))


def _freeze_q() -> dict[str, dict]:
    runs = _wandb_runs(Q_GROUP)
    records = {}
    for run_id, node in runs.items():
        if node["state"] != "finished":
            raise AssertionError(f"W&B run {run_id} is {node['state']}, not finished")
        config = json.loads(node["config"] or "{}")
        dataset_name = _unwrapped(config, "target")["dataset"]
        if dataset_name in records:
            raise AssertionError(f"Q group contains duplicate runs for {dataset_name}")
        artifact = _artifact(node, "baseline-rl-output-")
        root = CACHE / "q" / dataset_name
        files = _cached_wandb_report_files(artifact, root)
        metrics_bytes = files["test_metrics.json"]
        predictions_bytes = files["test_trial_predictions.csv"]
        trial_digest, n_prediction_rows = _trial_key_digest(predictions_bytes)
        metrics = json.loads(metrics_bytes)
        summary = json.loads(node["summaryMetrics"] or "{}")
        if not abs(
            metrics["normalized_likelihood"]
            - summary["target/test/normalized_likelihood"]
        ) < 1e-10:
            raise AssertionError(f"W&B/file likelihood mismatch for {run_id}")
        records[dataset_name] = {
            "wandb_run_id": run_id,
            "wandb_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}",
            "training_artifact": artifact,
            "slurm_array_job_id": Q_SLURM_ARRAY_JOB_ID,
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
            "ordered_trial_key_sha256": trial_digest,
            "n_prediction_rows": n_prediction_rows,
            "metrics": _report_metrics(metrics),
        }
    return records


def main() -> None:
    source_runs = json.loads((STUDY / "source_runs.json").read_text())["runs"]
    expected_keys = set(source_runs)
    datasets = {
        name: {
            "gru_group": group,
            "beaker_experiment_id": experiment_id,
            "gru": _freeze_gru(name, group, experiment_id),
        }
        for name, (group, experiment_id) in GRU_LAUNCHES.items()
    }
    q_records = _freeze_q()
    if set(q_records) != set(datasets):
        raise AssertionError(f"Q datasets differ from GRU datasets: {sorted(q_records)}")
    for name, dataset in datasets.items():
        if {row["source_key"] for row in dataset["gru"]} != expected_keys:
            raise AssertionError(f"{name} does not contain the exact source grid")
        q = q_records[name]
        for row in dataset["gru"]:
            if (
                row["ordered_trial_key_sha256"] != q["ordered_trial_key_sha256"]
                or row["n_prediction_rows"] != q["n_prediction_rows"]
            ):
                raise AssertionError(f"GRU/Q trial keys do not align for {name}/{row['source_key']}")
        dataset["q"] = q
        dataset["alignment"] = {
            "exact_ordered_trial_keys": True,
            "ordered_trial_key_sha256": q["ordered_trial_key_sha256"],
            "n_trials": q["n_prediction_rows"],
        }

    output = {
        "_meta": build_meta(
            "studies/09-gru-cross-species-transfer/analysis/freeze_matched_half.py",
            WANDB_GROUPS,
            study_root=STUDY,
        ),
        "contract": {
            "condition": "matched_half",
            "metric": "normalized_likelihood",
            "gru_source_hidden_size": 128,
            "gru_embedding_steps": 500,
            "gru_embedding_lr": 0.001,
            "q_model": "ForagerQLearning_L1F1_CK1_softmax",
        },
        "wandb_project": f"https://wandb.ai/{ENTITY}/{PROJECT}",
        "q_group": Q_GROUP,
        "datasets": datasets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
