#!/usr/bin/env python
"""Freeze the per-mouse HB outputs the per-mouse and shrinkage reports read.

Inputs (not committed; large or only on Beaker): for each finished NumPyro run, its saved
posterior ``one_stage_fit.nc``, its ``one_stage_fit.json`` (whose ``_meta.subject_ids`` is the order of
the subject axis) and its ``per_subject_matched`` table, all from the run's Beaker result dataset
(W&B blobs are not reachable from the analysis sandbox; the Beaker copies are the files W&B uploaded).

Outputs, committed:
  hb_per_subject_heldout.csv   per held-out mouse likelihood for every finished run (149 mice each)
  hb_subject_posteriors.csv    per training-cohort mouse, per parameter: posterior mean/sd/5%/95% in
                               NATURAL units, for the rungs listed in HEADLINE_RUNS
  hb_subjects_manifest.json    sha256 of every input file and the transform used

Transform (aind_dynamic_foraging_models.hierarchical_bayes.model.hattori2019_session_params): rates are
Phi(mu_p), inverse temperature is Phi(mu_p) * beta_max (10), bias_l is mu_p unchanged. The posterior
mean is taken over the transformed draws, not by transforming the mean of mu_p.

    python freeze_hb_subjects.py <post_dir> <meta_dir> <file_to_run_manifest.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "util"))
from _meta import build_meta  # noqa: E402

PARAMS = ("learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen", "softmax_inverse_temperature", "bias_l")
BETA_MAX = 10.0
HEADLINE_GROUPS = ("one-stage-ladder@20260903-154819", "one-stage-ladder@20260903-154828")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_natural(mu_p: np.ndarray) -> np.ndarray:
    """mu_p (..., 5) -> parameters in natural units, same shape."""
    out = np.empty_like(mu_p)
    out[..., 0:3] = ndtr(mu_p[..., 0:3])
    out[..., 3] = ndtr(mu_p[..., 3]) * BETA_MAX
    out[..., 4] = mu_p[..., 4]
    return out


def main(post_dir: str, meta_dir: str, manifest_path: str) -> None:
    post_dir, meta_dir = Path(post_dir), Path(meta_dir)
    runs = json.loads(Path(manifest_path).read_text())          # {"ladder_0.nc": {wandb_run_id, group, subject_ratio, ...}}
    held, subj, man = [], [], {}
    for fname, m in sorted(runs.items()):
        stem = fname[:-3]
        nc, fit_json, table = post_dir / fname, meta_dir / f"{stem}_one_stage_fit.json", meta_dir / f"{stem}_per_subject_matched.json"
        meta = json.loads(fit_json.read_text())
        ids = [str(s) for s in meta["_meta"]["subject_ids"]]
        mu_p = xr.open_dataset(nc, group="posterior")["mu_p"].load().values   # (chain, draw, subject, 5)
        if mu_p.shape[2] != len(ids):
            raise RuntimeError(f"{fname}: {mu_p.shape[2]} subjects in the posterior but {len(ids)} ids")
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"{fname}: duplicate subject ids")
        tab = json.loads(table.read_text())
        df = pd.DataFrame(tab["data"], columns=tab["columns"]).assign(wandb_run_id=m["wandb_run_id"], group=m["group"])
        df["subject_id"] = df.subject_id.astype(str)
        held.append(df)
        entry = {"posterior_sha256": sha(nc), "fit_json_sha256": sha(fit_json), "per_subject_table_sha256": sha(table),
                 "n_subjects_train": len(ids), "wandb_run_id": m["wandb_run_id"], "group": m["group"],
                 "beaker_experiment": m.get("beaker_experiment"), "subject_ratio": m["subject_ratio"]}
        if m["group"] in HEADLINE_GROUPS:
            nat = to_natural(mu_p.reshape(-1, mu_p.shape[2], 5))           # (chain*draw, subject, 5)
            q05, q95 = np.percentile(nat, [5, 95], axis=0)
            for j, p in enumerate(PARAMS):
                subj.append(pd.DataFrame({"wandb_run_id": m["wandb_run_id"], "D": len(ids), "subject_id": ids, "param": p,
                                          "hb_mean": nat[:, :, j].mean(0), "hb_sd": nat[:, :, j].std(0),
                                          "hb_q05": q05[:, j], "hb_q95": q95[:, j]}))
            entry["subject_posteriors_written"] = True
        man[fname] = entry
        print(f"{m['wandb_run_id'][-8:]} D={len(ids):>3} held-out rows={len(df)} subject posteriors={'yes' if m['group'] in HEADLINE_GROUPS else 'no'}", flush=True)
    pd.concat(held).to_csv(HERE / "hb_per_subject_heldout.csv", index=False)
    pd.concat(subj).to_csv(HERE / "hb_subject_posteriors.csv", index=False)
    meta = build_meta("analysis/freeze_hb_subjects.py", sorted({m["group"] for m in runs.values()}), HERE.parent)
    (HERE / "hb_subjects_manifest.json").write_text(json.dumps({
        "_meta": meta, "transform": {"params": PARAMS, "beta_max": BETA_MAX, "rule": "Phi(mu_p) for the three rates, Phi(mu_p)*beta_max for inverse temperature, mu_p for bias_l; mean over transformed draws"},
        "files": man}, indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
