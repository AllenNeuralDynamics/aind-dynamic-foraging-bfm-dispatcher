#!/usr/bin/env python
"""Convergence summaries of the saved NumPyro HB posteriors (R-hat, bulk ESS, divergences).

NumPyro HB runs log no R-hat or ESS to W&B (``hb/diagnostics`` is an image), so convergence is
computed here from each run's saved ``one_stage_fit.nc``. Only the sites that carry the
group-level and subject-level parameters are read (``population_mean``, ``population_scale``,
``log_sigma_mean``, ``log_sigma_spread``, ``mu_p``, ``log_sigma``); the session-level sites are the
bulk of the file and do not enter the gates.

Input is a directory of ``.nc`` files named ``<anything>.nc`` plus a manifest mapping each file to
its W&B run. A file is matched to a run by its subject count, never by job order: the subject
count is read from the posterior itself and must equal the count implied by the run's
``subject_ratio`` (0.016/0.049/0.163/0.489/1.0 -> 10/29/99/300/614, from the cohort logs).

    python summarize_hb_posteriors.py <posterior_dir> <manifest.json>

Gates, as set in the study plan: zero divergences, max R-hat < 1.01, min bulk ESS >= 400.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "util"))
from _meta import build_meta  # noqa: E402

N_SUBJECTS = {0.016: 10, 0.049: 29, 0.163: 99, 0.489: 300, 1.0: 614}
GROUP_SITES = ("population_mean", "population_scale", "log_sigma_mean", "log_sigma_spread")
SUBJECT_SITES = ("mu_p", "log_sigma")
GATES = dict(max_rhat=1.01, min_ess_bulk=400, divergences=0)


def summarize(path: Path) -> dict:
    post = xr.open_dataset(path, group="posterior")
    n_subj = int(post["mu_p"].shape[2])
    n_chain, n_draw = int(post.sizes["chain"]), int(post.sizes["draw"])
    ss = xr.open_dataset(path, group="sample_stats")
    div = int(np.asarray(ss["diverging"]).sum())
    out = {"file": path.name, "n_subjects": n_subj, "chains": n_chain, "draws_per_chain": n_draw, "divergences": div}
    for label, sites in (("group_level", GROUP_SITES), ("subject_level", SUBJECT_SITES)):
        sub = post[list(sites)].load()
        rhat = az.rhat(sub)
        ess = az.ess(sub, method="bulk")
        worst = {}
        for s in sites:
            worst[s] = {"max_rhat": float(np.nanmax(rhat[s].values)), "min_ess_bulk": float(np.nanmin(ess[s].values))}
        out[label] = {"per_site": worst,
                      "max_rhat": max(v["max_rhat"] for v in worst.values()),
                      "min_ess_bulk": min(v["min_ess_bulk"] for v in worst.values())}
    out["max_rhat"] = max(out["group_level"]["max_rhat"], out["subject_level"]["max_rhat"])
    out["min_ess_bulk"] = min(out["group_level"]["min_ess_bulk"], out["subject_level"]["min_ess_bulk"])
    out["gates"] = {"divergences": div <= GATES["divergences"], "max_rhat": out["max_rhat"] < GATES["max_rhat"],
                    "min_ess_bulk": out["min_ess_bulk"] >= GATES["min_ess_bulk"]}
    out["passes_all_gates"] = all(out["gates"].values())
    return out


def main(post_dir: str, manifest_path: str) -> None:
    manifest = json.loads(Path(manifest_path).read_text())   # {file: {wandb_run_id, group, subject_ratio, divergences_wandb}}
    rows = []
    for fname, m in sorted(manifest.items()):
        r = summarize(Path(post_dir) / fname)
        expected = N_SUBJECTS[round(float(m["subject_ratio"]), 3)]
        if r["n_subjects"] != expected:
            raise RuntimeError(f"{fname}: posterior has {r['n_subjects']} subjects but run {m['wandb_run_id']} "
                               f"(subject_ratio {m['subject_ratio']}) implies {expected}; the file/run mapping is wrong")
        if m.get("divergences_wandb") is not None and int(m["divergences_wandb"]) != r["divergences"]:
            raise RuntimeError(f"{fname}: {r['divergences']} divergences in the file but W&B says {m['divergences_wandb']}")
        rows.append({**m, **r})
        print(f"{m['wandb_run_id']} D={r['n_subjects']:>3} {m['num_warmup']}/{m['num_samples']} div={r['divergences']:>2} "
              f"rhat={r['max_rhat']:.4f} ess={r['min_ess_bulk']:.0f} "
              f"(group {r['group_level']['max_rhat']:.4f}/{r['group_level']['min_ess_bulk']:.0f}, subject {r['subject_level']['max_rhat']:.4f}/{r['subject_level']['min_ess_bulk']:.0f}) "
              f"gates={'PASS' if r['passes_all_gates'] else 'FAIL'}", flush=True)
    meta = build_meta("analysis/summarize_hb_posteriors.py", sorted({r["group"] for r in rows}), HERE.parent)
    (HERE / "hb_posterior_summary.json").write_text(json.dumps({"_meta": meta, "gates": GATES, "runs": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
