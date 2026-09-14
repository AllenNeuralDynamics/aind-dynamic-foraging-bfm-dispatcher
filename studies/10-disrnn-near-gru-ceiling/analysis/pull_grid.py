"""Pull the three staged near-GRU-ceiling W&B groups into a committed CSV.

W&B ACCESS: ``wandb.Api()`` fails in the Claude Science sandbox (spawns a wandb-core
service subprocess it can't reach — same failure documented in study 03's
``beta_scan_analysis.py`` and study 04's ``run_recovery_analysis.py``). This script hits
the GraphQL endpoint directly with ``requests`` (auth=('api', WANDB_API_KEY)) instead,
which is network-allowlisted and works from the sandbox. On the HPC login node the
``wandb`` SDK works too if you prefer it.

Also pulls the two external reference points this study's report compares against:
  * study 06's tuned disRNN operating point (D=614, mult=1, beta=3e-4, n_steps=100000),
    project AIND-disRNN/disrnn_data_scaling, group mult-d-grid@20260718-151409
  * study 01's GRU ceiling (hidden_size=256, D=614), project AIND-disRNN/mice_data_scaling,
    groups nxd-grid@20260623-102649 / nxd-grid@20260624-141106 / nxd-h512@20260720-195322

    python analysis/pull_grid.py

Writes analysis/grid.csv (this study's 3 stages) and analysis/reference_points.csv
(the two external comparators, independently re-verified — see the r1 report for why
this matters: study 06's own scaling_report.py hardcodes a STALE GRU=0.7268 constant at
D=614 that predates a later re-run of study 01's nxd-grid; the live re-pull here gives
0.7290 (3 seeds, tight agreement), which is the number this study's report uses).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
GRAPHQL_URL = "https://api.wandb.ai/graphql"

STUDY10_PROJECT = "AIND-disRNN/disrnn_near_gru_ceiling"
STUDY10_GROUPS = {
    "stage1-no-penalty": "stage1-no-penalty@20260909-021133",
    "stage2-linear-update-net": "stage2-linear-update-net@20260909-021223",
    "stage3-wide-latent": "stage3-wide-latent@20260909-140640",  # the latent32 RELAUNCH, not the failed latent256 original
}

S06_PROJECT = "AIND-disRNN/disrnn_data_scaling"
S06_GROUP = "mult-d-grid@20260718-151409"

GRU_PROJECT = "AIND-disRNN/mice_data_scaling"
GRU_GROUPS = ["nxd-grid@20260623-102649", "nxd-grid@20260624-141106", "nxd-h512@20260720-195322"]


def _wandb_key() -> str:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise SystemExit("WANDB_API_KEY not set.")
    return key


def _unwrap(v):
    if isinstance(v, dict) and set(v.keys()) == {"value"}:
        return _unwrap(v["value"])
    if isinstance(v, dict):
        return {k: _unwrap(x) for k, x in v.items()}
    return v


def fetch_group_runs(entity: str, project: str, group: str) -> list[dict]:
    """Fetch every run in one W&B group via GraphQL (sandbox-safe route)."""
    key = _wandb_key()
    q = """
    query Runs($entity: String!, $project: String!, $cursor: String, $filt: String!) {
      project(name: $project, entityName: $entity) {
        runs(first: 200, after: $cursor, filters: $filt) {
          edges { node { name state config summaryMetrics } cursor }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """
    filt = json.dumps({"group": group})
    runs, cursor = [], None
    while True:
        r = requests.post(
            GRAPHQL_URL, auth=("api", key),
            json={"query": q, "variables": {"entity": entity, "project": project, "cursor": cursor, "filt": filt}},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if "errors" in payload:
            raise RuntimeError(payload["errors"])
        proj = payload["data"]["project"]
        if not proj:
            return []
        conn = proj["runs"]
        for e in conn["edges"]:
            runs.append(e["node"])
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
    return runs


def parse_study10():
    rows = []
    for stage, group in STUDY10_GROUPS.items():
        for r in fetch_group_runs("AIND-disRNN", STUDY10_PROJECT.split("/")[1], group):
            cfg = _unwrap(json.loads(r["config"]))
            summ = json.loads(r["summaryMetrics"])
            arch = (cfg.get("model") or {}).get("architecture") or {}
            data = cfg.get("data") or {}
            rows.append(dict(
                stage=stage,
                run_name=r["name"],
                state=r["state"],
                seed=data.get("seed"),
                latent_size=arch.get("latent_size"),
                update_net_n_layers=arch.get("update_net_n_layers"),
                heldout_ll=summ.get("heldout/final/eval_likelihood"),
                checkpoint_ll=summ.get("checkpoint/eval_likelihood"),
                final_step=summ.get("_step"),
            ))
    rows.sort(key=lambda x: (x["stage"], x["seed"] or 0))
    out = HERE / "grid.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} — {len(rows)} runs")


def parse_reference_points():
    rows = []
    for r in fetch_group_runs("AIND-disRNN", S06_PROJECT.split("/")[1], S06_GROUP):
        cfg = _unwrap(json.loads(r["config"]))
        summ = json.loads(r["summaryMetrics"])
        pen = (cfg.get("model") or {}).get("penalties") or {}
        beta, upl = pen.get("beta"), pen.get("update_net_latent_penalty")
        ids = cfg.get("resolved_subject_ids") or []
        D = len(ids) or None
        mult = round(upl / beta) if (beta and upl) else None
        if r["state"] == "finished" and D == 614 and beta == 0.0003 and mult == 1:
            rows.append(dict(reference="study06_tuned_D614_mult1_beta3e-4", project=S06_PROJECT,
                              group=S06_GROUP, seed=(cfg.get("data") or {}).get("seed"),
                              heldout_ll=summ.get("heldout/eval_likelihood"), final_step=summ.get("_step")))

    for g in GRU_GROUPS:
        for r in fetch_group_runs("AIND-disRNN", GRU_PROJECT.split("/")[1], g):
            cfg = _unwrap(json.loads(r["config"]))
            summ = json.loads(r["summaryMetrics"])
            arch = (cfg.get("model") or {}).get("architecture") or {}
            ids = cfg.get("resolved_subject_ids") or []
            D = len(ids) or None
            if r["state"] == "finished" and arch.get("hidden_size") == 256 and D == 614:
                rows.append(dict(reference="gru_ceiling_H256_D614", project=GRU_PROJECT, group=g,
                                  seed=(cfg.get("data") or {}).get("seed"),
                                  heldout_ll=summ.get("heldout/final/eval_likelihood"), final_step=summ.get("_step")))

    out = HERE / "reference_points.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} — {len(rows)} rows")


if __name__ == "__main__":
    parse_study10()
    parse_reference_points()
