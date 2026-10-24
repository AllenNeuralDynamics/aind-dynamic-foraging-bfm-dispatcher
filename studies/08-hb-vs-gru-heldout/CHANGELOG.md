# Changelog — `studies/08-hb-vs-gru-heldout/`

One entry per merged PR (or coherent unreleased batch).

## 2026-10-23: status records that the study is closing

- Wrap-up checks, run on a compute node (`srun`, env `dynamic-foraging-bfm-cpu`) from a clean worktree of this branch:
  - `validate_provenance.py --strict` passes for every launch group: 13 Beaker groups with `--beaker`, 4 of them in W&B project `test`; the 2026-08-29 SLURM relaunch group, offline only; and the 5 Stan groups with `--wandb`. Every Beaker experiment and W&B sweep found for a group is recorded.
  - `make all` twice: the two runs differ only in `_meta`, and the four reports are byte-identical to the committed ones. Against the committed data files, `per_mouse.json`, `shrinkage.json` and `shrinkage.csv` differ only in the 15th–16th significant digit (cross-machine floating-point noise), so they are not recommitted.
- Not done, on purpose: crashed W&B runs are kept, because `wandb_keys.py` keeps the crashed groups for the crash history. The variant-root `sweep_*.yaml` are the templates the README's commands use, and each launch's copy is already in its launch-record folder, so nothing is moved under `_sweeps/`.
- README `## Status` now states the study-level position: all compute done, the negative verdict, the merged PRs, and what is left (the verdict on #72). The Variants index and Status tables list `stan-arm`.
- `stan-arm/notes.md` gives the variant's end state instead of pointing at the launch records; `one-stage-ladder/notes.md` links r4 in place rather than the merged PR.
- `results.md` stubs for every launch with a finished run: the six counted `one-stage-ladder` Beaker launches, the smoke and the three seed-0 repro launches (W&B project `test`), and the counted Stan group. Times are from Beaker and `sacct`; numbers are from `analysis/hb_runs.json` and `analysis/convergence.json`, except the smoke and repro likelihoods, which were read from W&B on 2026-10-23. The three repro runs give the same matched held-out likelihood to every digit.

## 2026-10-14 (third round): Stan diagnostics corrected, freezer inputs pinned

- **Stan R-hat and ESS were wrong, and r4's Stan comparison is redone.** The Stan trainer reshaped PyStan's flat draw axis chain-major; PyStan's axis is draw-major with the chain innermost, so the four chains were mixed. The one finished Stan run (D=29) logged lowest bulk ESS 7007 and R-hat 1.0049 to W&B; from its saved draws with the correct axis they are 1220 and 1.0030. `analysis/recompute_stan_diagnostics.py` recomputes them from `analysis/stan_inputs/` (pinned by sha256; the file's recorded fit time equals the W&B run's), and the wrong-axis computation reproduces the logged values exactly from the same draws. `convergence.py` reads the recomputed values; `hb_runs.json` keeps the logged ones as recorded. Stan against NumPyro at D = 29 is now 1.36x ESS per draw (was 7.8x), 2.3x wall time, 0.59x ESS per second (was 3.4x). Stan's 23 divergences are unchanged. The earlier 7.8x and 3.4x are retracted.
- `freeze_hb_subjects.py` verifies, before reading any input, the posterior (sha256, size, source-dataset W&B run), the fit JSON and the per-subject table against the pinned `hb_posterior_manifest.json`, which now carries the fit-JSON and table hashes. Re-run: the frozen CSVs are byte-identical; swapping two same-shaped tables is refused.
- `summarize_hb_posteriors.py` docstring describes the manifest, not subject count, as what matches a file to a run.

## 2026-10-14 (second round): report location, hashing, wording

- Reports moved from `reports/` to `analysis/reports/`, the layout the study-wrapup convention and the ten other studies with reports use; `update_reports.py`, the image links and this changelog follow. `make all` regenerates them identically.
- `summarize_hb_posteriors.py` and `freeze_hb_subjects.py` hash posteriors with a streaming digest (one input is about 2.4 GB), and the pinned `size_bytes` is now checked.
- `per_mouse.py` and r2 say what is asserted: equal per-mouse scored-trial counts, not trial identity.
- r4: zero divergences at D >= 99 does not identify the cause of the poor mixing, and the report no longer says it is not funnel geometry.

## 2026-10-14: review fixes to the analysis inputs

- `wandb_keys.CONFIG_FIELDS`: the resolved seed is read from `model.seed` (and `data.seed`). The root `seed` key is not logged to W&B, so every frozen run record had `seed: null`. Counted finished runs without a consistent seed are now rejected. Re-frozen: all 25 records now carry seed 0; no other field changed.
- `convergence.py`: the Stan run is selected from the pinned counted group `stan-arm@20260905-020928`, and both comparators assert exactly one match.
- `summarize_hb_posteriors.py` and `hb_posterior_manifest.json`: each posterior file is pinned by sha256 and size, by the Beaker result dataset it came from, and by the W&B run directory inside that dataset (which must equal the run id); draws per chain are checked against the run's `num_samples`. Numbers are unchanged (re-run against the pinned files).
- `per_mouse.py` and r2: the pairing is stated as what is asserted, matching mice and equal per-mouse scored-trial counts. Individual trials are not compared.

## [Unreleased] — analysis and reports

Added `analysis/` and `analysis/reports/` (r1 to r4), following the posthoc-reporting contract.

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
