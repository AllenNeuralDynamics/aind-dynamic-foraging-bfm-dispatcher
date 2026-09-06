"""Freeze author-aligned baseline results from pinned W&B groups."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402

from freeze_matched_half import (  # noqa: E402
    ENTITY,
    PROJECT,
    _artifact,
    _cached_wandb_report_files,
    _report_metrics,
    _trial_key_digest,
    _unwrapped,
    _wandb_runs,
)


MODELS_COMMIT = "a75ae23e0d5bce7985d3c4f5f7f30c7a971e0070"
ORIGINAL_MODELS_COMMIT = "25f5f1ce64705edbf266feb8d57ff83a018b12c5"
ZID_PARITY_MODELS_COMMIT = "553d1a9eae919bcec5121dc747d09d5559c31bb6"
GROUPS = {
    "grossman-meta-learning": {
        "group": "grossman-meta-learning@20260905-124420",
        "dataset": "grossman",
        "agent_class": "ForagerGrossmanMetaLearning",
        "author_selected": True,
        "slurm_job_id": "25580843",
        "foraging_models_commit": MODELS_COMMIT,
    },
    "chen-rlck": {
        "group": "chen-rlck@20260905-123624",
        "dataset": "chen",
        "agent_class": "ForagerRLCK",
        "author_selected": True,
        "slurm_job_id": "25580837",
        "foraging_models_commit": ORIGINAL_MODELS_COMMIT,
    },
    "zid-traditional-rlck": {
        "group": "zid-history-kernel@20260905-123624",
        "dataset": "zid",
        "agent_class": "ForagerRLCK",
        "author_selected": False,
        "slurm_job_id": "25580838_0",
        "foraging_models_commit": ORIGINAL_MODELS_COMMIT,
    },
    "zid-history-kernel-foraging": {
        "group": "zid-history-kernel@20260905-151851",
        "dataset": "zid",
        "agent_class": "ForagerZidHistoryKernel",
        "author_selected": True,
        "slurm_job_id": "25580951_1",
        "foraging_models_commit": ZID_PARITY_MODELS_COMMIT,
    },
}
CACHE = STUDY / "analysis" / "_cache_author_baselines"
OUTPUT = STUDY / "analysis" / "author_baseline_results.json"


def _freeze() -> dict[str, dict]:
    matched = json.loads((STUDY / "analysis" / "matched_half_results.json").read_text())
    nodes_by_group = {
        group: _wandb_runs(group)
        for group in sorted({specification["group"] for specification in GROUPS.values()})
    }
    records = {}
    for baseline, specification in GROUPS.items():
        matching = [
            node
            for node in nodes_by_group[specification["group"]].values()
            if node["displayName"] == baseline
        ]
        if len(matching) != 1:
            raise AssertionError(
                f"Expected one {baseline!r} run in {specification['group']!r}"
            )
        node = matching[0]
        if node["state"] != "finished":
            raise AssertionError(f"W&B run {node['name']} is {node['state']}, not finished")
        config = json.loads(node["config"] or "{}")
        model = _unwrapped(config, "model")
        target = _unwrapped(config, "target")
        meta = _unwrapped(config, "meta")
        expected = {
            "dataset": (target.get("dataset"), specification["dataset"]),
            "agent_class": (model.get("agent_class"), specification["agent_class"]),
            "foraging_models_commit": (
                meta.get("foraging_models_commit"),
                specification["foraging_models_commit"],
            ),
        }
        mismatches = {
            key: values for key, values in expected.items() if values[0] != values[1]
        }
        if mismatches:
            raise AssertionError(f"W&B config mismatch for {baseline}: {mismatches}")

        artifact = _artifact(node, "baseline-rl-output-")
        files = _cached_wandb_report_files(artifact, CACHE / baseline)
        metrics_bytes = files["test_metrics.json"]
        predictions_bytes = files["test_trial_predictions.csv"]
        metrics = json.loads(metrics_bytes)
        summary = json.loads(node["summaryMetrics"] or "{}")
        if not abs(
            metrics["normalized_likelihood"]
            - summary["target/test/normalized_likelihood"]
        ) < 1e-10:
            raise AssertionError(f"W&B/file likelihood mismatch for {node['name']}")
        trial_digest, n_prediction_rows = _trial_key_digest(predictions_bytes)
        q = matched["datasets"][specification["dataset"]]["q"]
        if (
            trial_digest != q["ordered_trial_key_sha256"]
            or n_prediction_rows != q["n_prediction_rows"]
        ):
            raise AssertionError(f"Author/Q trial keys do not align for {baseline}")
        records[baseline] = {
            **specification,
            "wandb_run_id": node["name"],
            "wandb_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{node['name']}",
            "output_artifact": artifact,
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "predictions_sha256": hashlib.sha256(predictions_bytes).hexdigest(),
            "ordered_trial_key_sha256": trial_digest,
            "n_prediction_rows": n_prediction_rows,
            "metrics": _report_metrics(metrics),
        }
    return records


def main() -> None:
    groups = sorted({specification["group"] for specification in GROUPS.values()})
    output = {
        "_meta": build_meta(
            "studies/09-gru-cross-species-transfer/analysis/freeze_author_baselines.py",
            groups,
            study_root=STUDY,
        ),
        "contract": {
            "condition": "matched_half",
            "metric": "normalized_likelihood",
            "fit_scope": "one independent fit per target subject",
            "optimizer": "differential_evolution",
            "foraging_models_commits": sorted(
                {specification["foraging_models_commit"] for specification in GROUPS.values()}
            ),
        },
        "wandb_project": f"https://wandb.ai/{ENTITY}/{PROJECT}",
        "records": _freeze(),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
