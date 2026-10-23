"""Single source of truth for the W&B summary keys study 08 reads, and the run groups it may read.

The keys are the contract between the wrapper's HB trainers (``HBTrainer`` for NumPyro and
``HBStanTrainer`` for the Stan arm), which write them into each run's W&B summary, and this
study's extraction. A wrapper schema change is then a one-file diff, and ``require`` turns a
renamed or missing key into a loud error instead of a silently dropped run.

The wrapper commit that defines these keys is pinned in this study's ``environment.lock`` and
stamped into every analysis JSON as ``_meta.wrapper_git_sha``.

Modelled on studies/01-gru-scaling-law/analysis/wandb_keys.py.
"""
from __future__ import annotations

PROJECT = "AIND-disRNN/mice_data_scaling"

# Groups this study may read. Never discover groups live: a run launched later into a pinned
# group, or a new group, must be a visible diff here and not a silent change to a report number.
#   n_runs   number of HB runs the group held when this list was written; extraction fails on a
#            different count.
#   counts   True if the group has at least one finished run, so it can feed a reported number.
#            Groups with counts=False are kept for the crash history only.
#   beaker_experiment / slurm_sweep   tie the group to its launch record under
#            variants/<variant>/launch_record/.
# Generated from the W&B project on 2026-10-13 and cross-checked against run_inventory.csv.
WANDB_GROUPS: dict[str, dict] = {
    "one-stage-ladder@20260902-235911": dict(kind="numpyro", n_runs=1, counts=False,
        beaker_experiment="01M1K112YCMY31GNDKHESW9G6Z",
        note="D~30 attempt 1, cancelled by me (500 warmup)"),
    "one-stage-ladder@20260903-052244": dict(kind="numpyro", n_runs=1, counts=True,
        beaker_experiment="01M1KKHNKZ67YGVVGRDH2G009B",
        note="D~29 at 500 warmup / 500 draws; failed all three gates"),
    "one-stage-ladder@20260903-153959": dict(kind="numpyro", n_runs=4, counts=False,
        beaker_experiment="01M1MPVQNC4HGJ7WT3K5PHY6E3",
        note="first ladder launch, cancelled about 5 min in"),
    "one-stage-ladder@20260903-154025": dict(kind="numpyro", n_runs=1, counts=False,
        beaker_experiment="01M1MPWGDCJZXY92C97S4927M0",
        note="first warmup-4000 launch, cancelled about 5 min in"),
    "one-stage-ladder@20260903-154819": dict(kind="numpyro", n_runs=4, counts=True,
        beaker_experiment="01M1MQB0SRRBDDRKP1SX3T55DV",
        note="ladder D~10/29/99/300 at warmup 2000, 500 draws"),
    "one-stage-ladder@20260903-154825": dict(kind="numpyro", n_runs=1, counts=True,
        beaker_experiment="01M1MQB4MVJ76GK8NQR07WH9HR",
        note="warmup-4000 probe at D~29"),
    "one-stage-ladder@20260903-154828": dict(kind="numpyro", n_runs=1, counts=True,
        beaker_experiment="01M1MQB7NVE8N9THHXVC4W7X8Z",
        note="full cohort D=614 at warmup 2000, 500 draws"),
    "one-stage-ladder@20260903-230522": dict(kind="numpyro", n_runs=1, counts=True,
        beaker_experiment="01M1NGB9S8TAP8MH3R165T6RSC",
        note="sampler-diagnostic probe at D~29 (extra NUTS fields)"),
    "one-stage-ladder@20260904-082628": dict(kind="numpyro", n_runs=2, counts=True,
        beaker_experiment="01M1PGEPRYGY542Y5JBQB17EED",
        note="draws probe, D~10 and D~29 at 2000 draws"),
    "stan-arm@20260905-020928": dict(kind="stan", n_runs=2, counts=True,
        slurm_sweep="4i17yn5r",
        note="first Stan launch: the one completed fit (D~29) and the cancelled serial D~99"),
    "stan-arm@20260905-115323": dict(kind="stan", n_runs=2, counts=False,
        slurm_sweep="inspj29s",
        note="threaded pair, cancelled; node was oversubscribed"),
    "stan-arm@20260905-121427": dict(kind="stan", n_runs=1, counts=False,
        slurm_sweep="5v3nwgzu",
        note="exclusive-node threaded D~99, cancelled"),
    "stan-arm@20260905-174213": dict(kind="stan", n_runs=2, counts=False,
        slurm_sweep="nswlcwu0",
        note="threaded reslice, cancelled"),
    "stan-arm@20260905-180320": dict(kind="stan", n_runs=2, counts=False,
        slurm_sweep="3io4fnwv",
        note="single-thread reslice, cancelled on mislabelled tasks"),
}

# Read from run.config, not the summary.
CONFIG_FIELDS = {
    "subject_ratio": ("data", "subject_ratio"),
    "num_warmup": ("model", "num_warmup"),
    "num_samples": ("model", "num_samples"),
    "num_chains": ("model", "num_chains"),
    # The root `seed` key is not logged to W&B (it is None on every run). The resolved seed is the one
    # Hydra interpolates into the model and data configs, which is what the sampler and the cohort split used.
    "seed": ("model", "seed"),
    "data_seed": ("data", "seed"),
    "dispatcher_commit": ("dispatcher_commit",),
    "wrapper_commit": ("wrapper_commit",),
    "foraging_models_commit": ("foraging_models_commit",),
}

COMMON = {"divergences": "hb/divergences", "fit_seconds": "hb/fit_seconds"}
NUMPYRO = {
    **COMMON,
    "estimator": "hb/estimator",
    "heldout_matched": "heldout/matched_likelihood",
    "heldout_k0": "heldout/few_shot_k0_likelihood",
    "heldout_k1": "heldout/few_shot_k1_likelihood",
    "heldout_k2": "heldout/few_shot_k2_likelihood",
    "heldout_k4": "heldout/few_shot_k4_likelihood",
    "heldout_k8": "heldout/few_shot_k8_likelihood",
    "num_test_subjects": "heldout/num_test_subjects",
    "num_test_trials": "heldout/num_test_trials",
}
# A finished NumPyro run must carry these; a miss means the wrapper schema changed.
NUMPYRO_REQUIRED = ("divergences", "fit_seconds", "heldout_matched", "num_test_subjects", "num_test_trials")
STAN = {
    **COMMON,
    "build_seconds": "hb/build_seconds",
    "n_subjects": "hb/n_subjects",
    "n_draws": "hb/n_draws",
    "real_trials": "hb/real_trials",
    "padded_trials": "hb/padded_trials",
    "max_rhat": "hb/max_rhat",
    "min_ess_bulk": "hb/min_ess_bulk",
    "min_ess_per_draw": "hb/min_ess_per_draw",
    "min_ess_per_second": "hb/min_ess_per_second",
}
STAN_REQUIRED = ("divergences", "fit_seconds", "n_subjects", "max_rhat", "min_ess_bulk")
# NumPyro runs log no r_hat or ESS (hb/diagnostics is an image), so those come from the saved
# posterior in the separate posterior-summary extraction, never from W&B.

_MISSING = object()


def require(summary, key: str, run_label: str):
    """Return ``summary[key]``, raising loudly if it is absent or None."""
    val = summary.get(key, _MISSING)
    if val is _MISSING or val is None:
        raise KeyError(f"{run_label}: expected W&B summary key {key!r} is missing; the wrapper "
                       "schema and analysis/wandb_keys.py have drifted apart")
    return val
