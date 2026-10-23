"""Freeze Study 13's source GRUs (D=614, H128, 3-way head) into ``source_runs.json``.

Reads the pinned W&B group over GraphQL (no SDK, so it also runs off-cluster), keeps the
D=614 / H128 cells, and records the run id plus the immutable ``gru-output-*`` artifact
id/digest that later extraction must load.

Cross-checks against picking up the wrong runs:
  * every run is ``data.ignore_policy=include`` (the 3-way head);
  * digests already in the manifest must be unchanged;
  * the engaged-trial L/R held-out likelihoods must equal Study 02's frozen
    ``scaling.json`` seed values for the same cell.

Needs WANDB_API_KEY (or api.wandb.ai in ~/.netrc).
"""

from __future__ import annotations

import json
import netrc
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

STUDY = Path(__file__).resolve().parents[1]
STUDIES = STUDY.parent
MANIFEST = STUDY / "source_runs.json"
STUDY02_SCALING = STUDIES / "02-gru-scaling-law-ignore" / "analysis" / "scaling.json"

# Pinned allowlist: model key -> (W&B group, hidden_size, subject_embedding_size).
# Never discovered live. The group also holds the H16/H64/H256 cells of the same launch.
WANDB_GROUPS = {
    "C3": ("nxd-3way@20260707-075401", 128, 4),
}
STUDY02_CHECKED = {"C3": "614x128"}
# Study 02 rounds its frozen seed values to 6 significant digits.
STUDY02_TOLERANCE = 5e-6
SEEDS = {0, 1, 2}
HELDOUT_LL_KEYS = ("heldout/final/eval_likelihood", "heldout/eval_likelihood")
HELDOUT_LR_KEYS = ("heldout/final/eval_likelihood_LR_engaged", "heldout/eval_likelihood_LR_engaged")
HELDOUT_PR_AUC_KEYS = ("heldout/final/engage_ignore_pr_auc", "heldout/engage_ignore_pr_auc")

RUNS_QUERY = """query Runs($entity:String!,$project:String!,$filters:JSONString){
  project(name:$project,entityName:$entity){
    runs(filters:$filters,first:100){edges{node{
      name state group config summaryMetrics
      outputArtifacts(first:30){edges{node{
        id state digest fileCount size artifactSequence{name}
      }}}
    }}}
  }
}"""


def _wandb_key() -> str:
    if key := os.environ.get("WANDB_API_KEY"):
        return key
    # Explicit path: netrc.netrc() with no argument rejects a ~/.netrc owned by another uid.
    credentials = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
    if not credentials or not credentials[2]:
        raise RuntimeError("WANDB_API_KEY is unset and api.wandb.ai is absent from ~/.netrc")
    return credentials[2]


def _wandb_runs(entity: str, project: str, group: str) -> list[dict]:
    response = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", _wandb_key()),
        json={
            "query": RUNS_QUERY,
            "variables": {
                "entity": entity,
                "project": project,
                "filters": json.dumps({"group": group}),
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return [edge["node"] for edge in payload["data"]["project"]["runs"]["edges"]]


def _unwrapped(config: dict, key: str):
    value = config.get(key, {})
    return value.get("value", value) if isinstance(value, dict) and "value" in value else value


def _first(summary: dict, keys: tuple[str, ...], run: str) -> float:
    value = next((summary[k] for k in keys if summary.get(k) is not None), None)
    if value is None:
        raise AssertionError(f"{run} has none of {keys}")
    return float(value)


def _gru_artifact(node: dict) -> dict:
    matches = [
        edge["node"]
        for edge in node["outputArtifacts"]["edges"]
        if edge["node"]["artifactSequence"]["name"].startswith("gru-output-")
    ]
    if len(matches) != 1 or matches[0]["state"] != "COMMITTED":
        raise AssertionError(f"Expected one committed gru-output artifact for {node['name']}")
    return matches[0]


def _freeze_group(
    entity: str, project: str, model_key: str, group: str, hidden_size: int, embedding: int
) -> dict[str, dict]:
    runs = {}
    for node in _wandb_runs(entity, project, group):
        config = json.loads(node["config"] or "{}")
        data = _unwrapped(config, "data")
        architecture = _unwrapped(config, "model")["architecture"]
        if (
            architecture["hidden_size"] != hidden_size
            or architecture["subject_embedding_size"] != embedding
            or float(data["subject_ratio"]) != 1.0
        ):
            continue
        if node["state"] != "finished":
            raise AssertionError(f"{node['name']} is {node['state']}, not finished")
        if architecture["session_encoding_type"] != "scalar":
            raise AssertionError(f"{node['name']} session_encoding_type drifted")
        if data["ignore_policy"] != "include":
            raise AssertionError(f"{node['name']} is ignore_policy={data['ignore_policy']}, not include")
        seed = int(_unwrapped(config, "seed") if "seed" in config else data["seed"])
        summary = json.loads(node["summaryMetrics"] or "{}")
        artifact = _gru_artifact(node)
        key = f"{model_key}-s{seed}"
        if key in runs:
            raise AssertionError(f"Duplicate cell {key} in {group}")
        runs[key] = {
            "seed": seed,
            "run_id": node["name"],
            "artifact_name": f"{artifact['artifactSequence']['name']}",
            "artifact_id": artifact["id"],
            "artifact_digest": artifact["digest"],
            "heldout_likelihood": _first(summary, HELDOUT_LL_KEYS, node["name"]),
            "heldout_likelihood_LR_engaged": _first(summary, HELDOUT_LR_KEYS, node["name"]),
            "heldout_ignore_pr_auc": _first(summary, HELDOUT_PR_AUC_KEYS, node["name"]),
        }
    if {r["seed"] for r in runs.values()} != SEEDS or len(runs) != len(SEEDS):
        raise AssertionError(f"{group} {model_key} D=614 cells: {sorted(runs)}")
    return dict(sorted(runs.items()))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    study02 = json.loads(STUDY02_SCALING.read_text())["grid"]
    for model_key, (group, hidden_size, embedding) in WANDB_GROUPS.items():
        entry = manifest["models"][model_key]
        if entry["group"] != group:
            raise AssertionError(f"{model_key} manifest group {entry['group']} != {group}")
        runs = _freeze_group(
            manifest["entity"], manifest["project"], model_key, group, hidden_size, embedding
        )

        for key, previous in entry["runs"].items():
            if runs[key]["artifact_digest"] != previous["artifact_digest"]:
                raise AssertionError(f"{key} artifact digest changed since it was frozen")
        if model_key in STUDY02_CHECKED:
            expected = sorted(study02[STUDY02_CHECKED[model_key]]["LR_engaged"]["seeds"])
            found = sorted(r["heldout_likelihood_LR_engaged"] for r in runs.values())
            if any(abs(a - b) > STUDY02_TOLERANCE for a, b in zip(expected, found)):
                raise AssertionError(f"{model_key} L/R likelihoods {found} != Study 02 {expected}")

        entry["runs"] = runs
        entry["status"] = "complete"
    manifest["frozen_at_pt"] = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(
        timespec="seconds"
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    main()
