"""Freeze matched-half GRU and Q results from pinned W&B/Beaker launches."""

from __future__ import annotations

import csv
import hashlib
import json
import netrc
import os
import subprocess
import sys
import time
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
    "gru-lebedeva-matched-half@20260905-232924",
    "gru-beron-matched-half@20260905-232924",
    "gru-kwak-matched-half@20260905-232924",
    "gru-miller-matched-half@20260905-232924",
    "gru-findling-matched-half@20260905-232924",
    "gru-tang-matched-half@20260905-232924",
    "gru-alsio-matched-half@20260905-232925",
    "gru-eckstein-matched-half@20260905-232924",
    "gru-costa-matched-half@20260905-232924",
    "gru-lopez-mouse-matched-half@20260905-232924",
    "q-matched-half@20260905-024031",
    "q-expanded-matched-half@20260906-001656",
]
GRU_LAUNCHES = {
    "grossman": (WANDB_GROUPS[0], "01M1RE7RE42MHTHFDDRYJWTWHV"),
    "chen": (WANDB_GROUPS[1], "01M1RFF0YVREC2924A9Z2Y13XF"),
    "zid": (WANDB_GROUPS[2], "01M1RG1X3W0VK8ZBYQ8ZB4V4BR"),
    "lebedeva": (WANDB_GROUPS[3], "01M1TPMMP0HY1F03AKFKEJSK4S"),
    "beron": (WANDB_GROUPS[4], "01M1TPMFZM1GYD60REY6JHTQSR"),
    "kwak": (WANDB_GROUPS[5], "01M1TPMPJJNBHT2RT5WHR8KRJS"),
    "miller": (WANDB_GROUPS[6], "01M1TPMHVARYQ17AYFSSTFCACS"),
    "findling": (WANDB_GROUPS[7], "01M1TPMS3BQPDYNEV1K98PY43X"),
    "tang": (WANDB_GROUPS[8], "01M1TPMY71W6PEGRPTRHMYQQDM"),
    "alsio": (WANDB_GROUPS[9], "01M1TPNT60G2REHV0BS2JYJH32"),
    "eckstein": (WANDB_GROUPS[10], "01M1TPN53SDJYVN9TE5YQ6ZEXY"),
    "costa": (WANDB_GROUPS[11], "01M1TPN1PNEMN6Q1036ZAXDCNV"),
    "lopez_mouse": (WANDB_GROUPS[12], "01M1TPMW5BXV82QQ0MS3AWWZE3"),
}
Q_LAUNCHES = [
    (WANDB_GROUPS[13], "25580070"),
    (WANDB_GROUPS[14], "25581304"),
]
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


def _download(url: str, path: Path) -> None:
    """Download a potentially large artifact file with bounded retries and resume."""
    partial = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(60, 120),
            ) as response:
                if offset and response.status_code == 200:
                    offset = 0
                elif offset and response.status_code != 206:
                    response.raise_for_status()
                    raise RuntimeError(
                        f"Artifact server did not honor byte range at offset {offset}"
                    )
                response.raise_for_status()
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            partial.replace(path)
            return
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


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
            _download(matches[0]["directUrl"], path)
        output[suffix] = path.read_bytes()
    return output


def _prediction_summary(data: bytes) -> tuple[str, int, list[dict]]:
    digest = hashlib.sha256()
    rows = 0
    sessions: dict[tuple[str, str], list[float]] = {}
    text = data.decode("utf-8").splitlines()
    for row in csv.DictReader(text):
        digest.update(
            f"{row['subject_id']}\t{row['ses_idx']}\t{row['trial']}\t{row['choice']}\n".encode()
        )
        key = (row["subject_id"], row["ses_idx"])
        aggregate = sessions.setdefault(key, [0.0, 0])
        aggregate[0] += float(row["log_likelihood_nats"])
        aggregate[1] += 1
        rows += 1
    per_session = [
        {
            "subject_id": subject_id,
            "ses_idx": ses_idx,
            "n_trials": int(count),
            "mean_log_likelihood_nats": total / count,
        }
        for (subject_id, ses_idx), (total, count) in sorted(sessions.items())
    ]
    return digest.hexdigest(), rows, per_session


def _report_metrics(metrics: dict, per_session: list[dict]) -> dict:
    """Keep only the likelihood values consumed by the committed report."""
    return {
        "n_trials": metrics["n_trials"],
        "mean_log_likelihood_nats": metrics["mean_log_likelihood_nats"],
        "normalized_likelihood": metrics["normalized_likelihood"],
        "per_subject_mean_log_likelihood_nats": {
            item["subject_id"]: item["mean_log_likelihood_nats"]
            for item in metrics["per_subject"]
        },
        "per_session": per_session,
    }


def _job_env(job: dict) -> dict[str, str]:
    return {
        item["name"]: item["value"]
        for item in job["execution"]["spec"]["envVars"]
        if item.get("value") is not None
    }


def _beaker_tasks(experiment_id: str) -> list[dict]:
    result = subprocess.run(
        ["beaker", "experiment", "tasks", experiment_id, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout)


def _freeze_gru(dataset_name: str, group: str, experiment_id: str) -> list[dict]:
    runs = _wandb_runs(group)
    records = []
    tasks = _beaker_tasks(experiment_id)
    if len(tasks) != 15:
        raise AssertionError(f"Beaker experiment {experiment_id} has {len(tasks)} tasks")
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
        if target.get("dataset") != dataset_name:
            raise AssertionError(f"W&B run {run_id} targets {target.get('dataset')!r}")

        source_key = source["key"]
        artifact = _artifact(node, "external-gru-output-")
        files = _cached_wandb_report_files(
            artifact, CACHE / "gru" / dataset_name / source_key
        )
        metrics_bytes = files["test_metrics.json"]
        predictions_bytes = files["test_trial_predictions.csv"]
        trial_digest, n_prediction_rows, per_session = _prediction_summary(
            predictions_bytes
        )
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
                "evaluation_artifact": artifact,
                "beaker_job_id": job["id"],
                "beaker_result_dataset_id": job["result"]["beaker"],
                "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
                "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
                "ordered_trial_key_sha256": trial_digest,
                "n_prediction_rows": n_prediction_rows,
                "metrics": _report_metrics(metrics, per_session),
            }
        )
    if runs:
        raise AssertionError(f"W&B group {group} has runs absent from Beaker: {sorted(runs)}")
    return sorted(records, key=lambda row: (row["nominal_D"], row["seed"]))


def _freeze_q(group: str, slurm_array_job_id: str) -> dict[str, dict]:
    runs = _wandb_runs(group)
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
        trial_digest, n_prediction_rows, per_session = _prediction_summary(
            predictions_bytes
        )
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
            "slurm_array_job_id": slurm_array_job_id,
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
            "ordered_trial_key_sha256": trial_digest,
            "n_prediction_rows": n_prediction_rows,
            "metrics": _report_metrics(metrics, per_session),
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
    q_records = {}
    for group, slurm_array_job_id in Q_LAUNCHES:
        for name, record in _freeze_q(group, slurm_array_job_id).items():
            if name in q_records:
                raise AssertionError(f"Q launches contain duplicate dataset {name}")
            q_records[name] = record
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
        "q_groups": [group for group, _ in Q_LAUNCHES],
        "datasets": datasets,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
