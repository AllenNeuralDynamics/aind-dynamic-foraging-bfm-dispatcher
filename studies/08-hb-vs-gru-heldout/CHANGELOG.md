# Changelog — `studies/08-hb-vs-gru-heldout/`

One entry per merged PR (or coherent unreleased batch).

## [Unreleased] — analysis and reports

Added `analysis/` and `reports/` (r1 to r4), following the posthoc-reporting contract.

- **Frozen inputs.** `wandb_keys.py` pins the 14 W&B groups this study may read; `extract_hb_runs.py` freezes their runs with a per-run
  digest; `summarize_hb_posteriors.py` recomputes R-hat, bulk ESS and divergences from every saved NumPyro posterior;
  `freeze_hb_subjects.py` freezes per-mouse HB held-out likelihoods and subject posteriors; `ml_params/` holds the per-mouse
  maximum-likelihood fits of the classical RL baselines, copied from `/allen` with checksums.
- **Reports.** r1 three-arm held-out comparison, r2 per-mouse paired comparison, r3 shrinkage by parameter, r4 convergence and
  Stan-versus-NumPyro. All `status: draft`.
- **Finding that qualifies r1 to r3.** No finished HB run passes all convergence gates. The D = 99, 300 and 614 rungs did not mix at
  the per-mouse level (zero divergences), although the trainer's own group-level diagnostics would have reported D = 300 as acceptable.
- **Correction.** An earlier session figure for Stan's effective-sample advantage (about 18x per draw) compared Stan's
  population-mean ESS with NumPyro's lowest ESS over every site. Like for like, on the population means, it is about 7.8x. r4 carries the
  corrected figure.
- **Not done here.** `results.md` stubs for settled launch groups; a validator that reads per-launch subdirectories of `launch_record/`.
