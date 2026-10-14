#!/usr/bin/env python
"""Recompute the HB-Stan D=29 fit's R-hat and ESS from its saved draws, with the correct chain axis.

Why this exists. HBStanTrainer logged hb/max_rhat, hb/min_ess_bulk and the per-draw / per-second variants to W&B
from a reshape that treated PyStan's flat draw axis as chain-major. PyStan 3's ``fit[name]`` has a trailing axis of
``num_samples * num_chains`` in DRAW-MAJOR order with the chain index innermost, so that reshape mixed the four chains
and the logged ESS was far too high. The saved draws are unaffected, so the diagnostics are recomputed here from them
and ``convergence.py`` reads this file instead of the W&B summary values for Stan. The W&B values stay in
``hb_runs.json`` as recorded and are known to be wrong for this run.

Inputs (committed): analysis/stan_inputs/{stan_d29_one_stage_fit.npz,.json,MANIFEST.json}; hb_runs.json.
Output: analysis/stan_d29_diagnostics.json. Runs in the ArviZ environment (see environment.lock):

    python recompute_stan_diagnostics.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "util"))
from _meta import build_meta  # noqa: E402

SITES = ("population_mean", "population_scale", "log_sigma_mean", "log_sigma_spread")
PARAMS = ("learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen", "softmax_inverse_temperature", "bias_l")
STAN_GROUP = "stan-arm@20260905-020928"


def chains_first(flat, num_chains, num_samples):
    """``(..., samples*chains)`` PyStan layout (chain innermost) -> ``(..., chain, draw)``.

    Mirrors ``chains_first`` in the wrapper's hb_stan_trainer.py and the models package's Stan driver; kept as a
    copy so this analysis does not import the training runtime.
    """
    flat = np.asarray(flat)
    if flat.shape[-1] != num_chains * num_samples:
        raise ValueError(f"last axis {flat.shape[-1]} != {num_chains} x {num_samples}")
    return np.swapaxes(flat.reshape(*flat.shape[:-1], num_samples, num_chains), -1, -2)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import arviz as az

    inp = HERE / "stan_inputs"
    man = json.loads((inp / "MANIFEST.json").read_text())
    for name, meta in man["files"].items():
        got = _sha256(inp / name)
        if got != meta["sha256"] or (inp / name).stat().st_size != meta["size_bytes"]:
            raise RuntimeError(f"{name}: does not match its pinned sha256/size")
    fit_meta = json.loads((inp / "stan_d29_one_stage_fit.json").read_text())
    runs = {r["wandb_run_id"]: r for r in json.loads((HERE / "hb_runs.json").read_text())["runs"]}
    run = runs[man["wandb_run_id"]]
    if run["group"] != STAN_GROUP or run["kind"] != "stan" or not run["counts"]:
        raise RuntimeError("the pinned W&B run is not the counted Stan run")
    # The identity link: the file's own recorded fit time equals the W&B run's.
    if fit_meta["metrics"]["hb/fit_seconds"] != run["fit_seconds"]:
        raise RuntimeError("saved fit time differs from the W&B run's; the files may belong to another run")
    chains, draws = int(fit_meta["num_chains"]), int(fit_meta["num_samples"])
    if (chains, draws, int(fit_meta["n_subjects"])) != (int(run["num_chains"]), int(run["num_samples"]), int(run["n_subjects"])):
        raise RuntimeError("chain / draw / subject counts disagree between the file and the W&B run")

    z = np.load(inp / "stan_d29_one_stage_fit.npz")
    per_site, wrong_popmean = {}, []
    for site in SITES:
        arr = z[site]
        if arr.shape != (len(PARAMS), chains * draws):
            raise RuntimeError(f"{site}: unexpected shape {arr.shape}")
        rows = []
        for i, param in enumerate(PARAMS):
            right = chains_first(arr[i], chains, draws)
            rows.append({"param": param, "ess_bulk": float(az.ess(right, method="bulk")), "rhat": float(az.rhat(right))})
            if site == "population_mean":
                old = arr[i].reshape(chains, draws)            # what the trainer computed (wrong)
                wrong_popmean.append({"param": param, "ess_bulk": float(az.ess(old, method="bulk")), "rhat": float(az.rhat(old))})
        per_site[site] = {"rows": rows, "min_ess_bulk": min(r["ess_bulk"] for r in rows), "max_rhat": max(r["rhat"] for r in rows)}
    pm = per_site["population_mean"]
    out = {
        "_meta": build_meta("analysis/recompute_stan_diagnostics.py", [STAN_GROUP], HERE.parent),
        "wandb_run_id": man["wandb_run_id"], "n_chains": chains, "draws_per_chain": draws, "n_draws": chains * draws,
        "arviz_version": az.__version__,
        "chain_axis": "PyStan layout: flat axis is draw-major with the chain innermost; reshaped (draw, chain) then transposed",
        "per_site": per_site,
        "population_mean": {"max_rhat": pm["max_rhat"], "min_ess_bulk": pm["min_ess_bulk"]},
        "as_logged_to_wandb_WRONG": {
            "why": "chain-major reshape of a draw-major axis mixes the chains",
            "logged_in_hb_runs_json": {"max_rhat": run["max_rhat"], "min_ess_bulk": run["min_ess_bulk"]},
            "reproduced_here_from_the_same_draws": {"max_rhat": max(r["rhat"] for r in wrong_popmean),
                                                      "min_ess_bulk": min(r["ess_bulk"] for r in wrong_popmean)},
        },
    }
    (HERE / "stan_d29_diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"population_mean (correct axis): max R-hat {pm['max_rhat']:.4f}, lowest bulk ESS {pm['min_ess_bulk']:.0f}")
    w = out["as_logged_to_wandb_WRONG"]
    print(f"as logged (wrong axis): {w['logged_in_hb_runs_json']} | reproduced: {w['reproduced_here_from_the_same_draws']}")


if __name__ == "__main__":
    main()
