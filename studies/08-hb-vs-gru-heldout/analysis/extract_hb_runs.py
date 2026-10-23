#!/usr/bin/env python
"""Freeze the HB runs that study 08's reports read.

Reads only the groups in ``wandb_keys.WANDB_GROUPS`` (never discovers groups), checks each
group holds exactly the recorded number of runs, requires the contract keys on every finished
run, and writes ``hb_runs.json`` plus ``hb_runs.csv``. Every report producer reads those frozen
files and not W&B, so a report regenerates offline.

Each run carries ``wandb_run_id`` and a ``digest``, the sha256 of that run's extracted record.
Re-running against unchanged W&B state reproduces every digest; a changed digest means the run
changed after it was frozen.

W&B is read through its GraphQL endpoint with ``requests`` (the route the dispatcher's own client
uses), so the script needs only ``requests`` and ``pandas``. The key is read from WANDB_API_KEY.

    WANDB_API_KEY=... python extract_hb_runs.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "util"))
import wandb_keys as K  # noqa: E402
from _meta import build_meta  # noqa: E402

ENTITY, PROJECT = K.PROJECT.split("/")
QUERY = """query($e:String!,$p:String!,$f:JSONString,$c:String){project(name:$p,entityName:$e){
  runs(first:100,after:$c,filters:$f){pageInfo{hasNextPage endCursor}
  edges{node{name displayName group state createdAt heartbeatAt host summaryMetrics config}}}}}"""


def gql(variables: dict) -> dict:
    resp = requests.post("https://api.wandb.ai/graphql", json={"query": QUERY, "variables": variables},
                         auth=("api", os.environ["WANDB_API_KEY"]), timeout=120)
    resp.raise_for_status()
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"W&B GraphQL error: {body['errors']}")
    return body["data"]["project"]["runs"]


def runs_in_group(group: str) -> list[dict]:
    out, cursor = [], None
    while True:
        page = gql({"e": ENTITY, "p": PROJECT, "f": json.dumps({"group": group}), "c": cursor})
        out += [e["node"] for e in page["edges"]]
        if not page["pageInfo"]["hasNextPage"]:
            return out
        cursor = page["pageInfo"]["endCursor"]


def flat_config(raw: str) -> dict:
    cfg = json.loads(raw or "{}")
    return {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in cfg.items()}


def dig(cfg: dict, path: tuple):
    for p in path:
        cfg = (cfg or {}).get(p) if isinstance(cfg, dict) else None
    return cfg


def digest(rec: dict) -> str:
    return hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()


def extract() -> list[dict]:
    rows = []
    for group, spec in K.WANDB_GROUPS.items():
        runs = runs_in_group(group)
        if len(runs) != spec["n_runs"]:
            raise RuntimeError(f"{group}: W&B holds {len(runs)} runs but wandb_keys.WANDB_GROUPS records "
                               f"{spec['n_runs']}. A run was added or removed after the group was pinned; "
                               "update the allowlist deliberately and re-check its launch record.")
        keys = K.NUMPYRO if spec["kind"] == "numpyro" else K.STAN
        required = K.NUMPYRO_REQUIRED if spec["kind"] == "numpyro" else K.STAN_REQUIRED
        for run in sorted(runs, key=lambda r: r["name"]):
            summ = json.loads(run["summaryMetrics"] or "{}")
            cfg = flat_config(run["config"])
            label = f"{group}/{run['name']}"
            rec = {"group": group, "wandb_run_id": run["name"], "kind": spec["kind"], "state": run["state"],
                   "created": run["createdAt"], "host": run["host"],
                   "beaker_experiment": spec.get("beaker_experiment"), "slurm_sweep": spec.get("slurm_sweep"),
                   "counts": spec["counts"]}
            rec.update({name: dig(cfg, path) for name, path in K.CONFIG_FIELDS.items()})
            if run["state"] == "finished":
                for name in required:
                    K.require(summ, keys[name], label)
            if spec["counts"] and run["state"] == "finished":
                # A reported number must be traceable to the seed it ran with. Both are None when the config
                # lacks them, so absence is rejected rather than frozen as null.
                if rec["seed"] is None or rec["data_seed"] is None or rec["seed"] != rec["data_seed"]:
                    raise RuntimeError(f"{label}: resolved seed is missing or inconsistent "
                                       f"(model.seed={rec['seed']!r}, data.seed={rec['data_seed']!r})")
            rec.update({name: summ.get(key) for name, key in keys.items()})
            rows.append(rec)
    for r in rows:
        r["digest"] = digest({k: v for k, v in r.items() if k != "digest"})
    return rows


def main() -> None:
    rows = extract()
    meta = build_meta("analysis/extract_hb_runs.py", sorted(K.WANDB_GROUPS), HERE.parent)
    (HERE / "hb_runs.json").write_text(json.dumps({"_meta": meta, "runs": rows}, indent=2, default=str) + "\n")
    pd.DataFrame(rows).to_csv(HERE / "hb_runs.csv", index=False)
    fin = [r for r in rows if r["state"] == "finished"]
    print(f"{len(rows)} runs in {len(K.WANDB_GROUPS)} groups; {len(fin)} finished")


if __name__ == "__main__":
    main()
