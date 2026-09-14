"""Pull the wave2-step-budget-and-penalty-extension W&B group into a committed CSV.

Sibling of ``pull_grid.py`` (same schema, same conventions) for the wave-2 extended-penalty-
range grid (8 runs: D in {300, 614} x (mult, beta) in {(1, 1e-4), (0.5, 3e-4)} x seed in {0, 1}).
Kept as a separate file/output rather than appended to ``pull_grid.py``/``grid.csv`` because
wave 2 is a distinct W&B group and its own variant folder -- merging happens at analysis time
(``r3_penalty_edge_extension.py``), not at pull time, so each producer's source data stays
independently re-pullable and auditable.

    python analysis/pull_wave2_grid.py

Writes analysis/grid_wave2.csv -- one row per run: D, mult, beta, seed, state, heldout_ll, eval_ll.

NOTE on sandbox execution: ``wandb.Api()`` fails in the Claude Science Mac sandbox (spawns a
blocked ``wandb-core`` helper subprocess). This script is written for a working wandb SDK
environment (HPC login node, or any machine with normal wandb auth) per pull_grid.py's own
convention. When run from the sandbox, use the GraphQL route documented in the
posthoc-reporting skill's ``references/wandb-graphql-sandbox.md`` instead (same output schema).
"""
from __future__ import annotations

import csv
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
PROJECT = "AIND-disRNN/disrnn_data_scaling"
GROUP = "wave2-step-budget-and-penalty-extension@20260909-021546"


def main() -> None:
    api = wandb.Api()
    rows = []
    for r in api.runs(PROJECT, filters={"group": GROUP}, per_page=200):
        cfg, s = r.config, r.summary
        model = cfg.get("model") or {}
        pen = model.get("penalties") or {}
        beta = pen.get("beta")
        upl = pen.get("update_net_latent_penalty")
        ids = cfg.get("resolved_subject_ids") or []
        row = {
            "run_id": r.id,
            "wandb_run_name": r.name,
            "state": r.state,
            "D": len(ids) or None,           # ACTUAL resolved D, never the nominal ratio
            "seed": (cfg.get("data") or {}).get("seed"),
            "beta": beta,
            "mult": round(upl / beta, 4) if (beta and upl) else None,
            "heldout_ll": s.get("heldout/eval_likelihood"),
            "eval_ll": s.get("checkpoint/eval_likelihood"),
            "final_step": s.get("_step"),
            "heldout_backfilled": bool(s.get("heldout/eval_likelihood_backfilled", False)),
        }
        rows.append(row)

    rows.sort(key=lambda x: (x["D"] or 0, x["mult"] or 0, x["beta"] or 0, x["seed"] or 0))
    out = HERE / "grid_wave2.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    done = sum(1 for r in rows if r["state"] == "finished")
    print(f"wrote {out} — {len(rows)} runs seen ({done} finished, {len(rows)-done} in flight/failed)")


if __name__ == "__main__":
    main()
