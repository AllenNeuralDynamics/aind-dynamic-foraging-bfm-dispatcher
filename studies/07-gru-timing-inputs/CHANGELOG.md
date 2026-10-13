# Changelog — 07-gru-timing-inputs

Per-study log; newest first. One entry per merged PR or significant milestone.
Dates in America/Los_Angeles.

## 2026-09-03 — r3 mechanism report (why RT/licks help), on the held-out mice

- Added report **r3** and producer `analysis/why_features_help.py`: two logistic
  regressions (`stay ~ prev_reward * z(prev logRT)` and `* z(prev total licks)`)
  with per-mouse cluster-bootstrap 95% CIs. Finding: prev-trial RT predicts
  switching reward-INDEPENDENTLY (a global engagement signal), lick count is
  reward-GATED (more licking after no-reward → abandon; after reward → mild stay).
- Cohort = the **held-out mice** (pinned in `analysis/provenance/heldout_subjects.txt`:
  eligible cohort minus the union of every GRU training cohort; 157 listed, 125
  enter the regressions — the other 32 are early old-schema sessions lacking the
  `reaction_time` field that `build_sequence` needs to form previous-trial RT). Chosen because the GRU's
  likelihood gain is scored on held-out mice, so the mechanism is shown on the
  population it explains — and the coupling is population-general (the same trend
  holds on the train mice), which is why the model generalizes. Effects are firmer
  than the earlier train-set draw (rewarded-lick z≈+3.9 vs +2.6).
- Note: mouse-clustering is essential — naive trial-level SEs are anti-conservative
  by ~an order of magnitude. The lowest-lick bin dips (very-low-lick = disengaged
  trials switch more), the same engagement signal the RT panel carries.

## 2026-09-03 — Probe/GRU maturity reconciliation + broken report figure paths

- The logistic probe is mature-only + curricula; the GRU runs use
  `data.mature_only: false` (all-stage, via `mice_snapshot_scaling.yaml`). Added
  `analysis/probe_maturity_reconciliation.py` (re-runs the probe's exact fit on
  the same seeded 60-subject cohort, varying only the session filter) with
  `fig_probe_maturity_reconciliation.png` + `probe_maturity_reconciliation.csv`
  and a subsection in r1. Verdict: dropping the maturity filter leaves the Δ
  unchanged (+0.00764 → +0.00834) and drops the baseline (0.7433 → 0.7294) onto
  the GRU OFF baseline (0.7285). Variant A reproduces the committed
  `timing_calibration.csv` to full precision, so the effect is robust to the
  maturity scope. Conclusion unaffected.
- Fixed broken figure paths in the r1/r2 report blocks: study-root figures were
  linked `../fig_*.png` (resolves to `analysis/`) instead of `../../fig_*.png`.
  Corrected in `update_reports.py` so regenerated blocks render.

## 2026-08-26 — Study wrap-up + analysis normalization

- Normalized the study folder to `study-conventions`: added `analysis/`
  (`timing_scaling.py` single producer, `timing_scaling.json`,
  `timing_scaling.csv`, `wandb_keys.py`, `update_reports.py`, `reports/r1`,`r2` +
  `INDEX.md`, `provenance/`), study-root figures
  (`fig_block_decomposition.png`, `fig_threeway.png`,
  `fig_selection_breakdown.png`), `Makefile`, `environment.lock`, `.gitignore`,
  this changelog, and a **Verdict** section in the README.
- Reused shared helpers `studies/util/_meta.py` and `plot_style.py`.
- `environment.lock` pins wrapper `78b8d118` — the WRAPPER_REF the last-scored
  arms (shuffled / RT-only / licks-only) launched with, carrying the timing
  module, the held-out-path fix, the shuffle implementation and the run-seed
  derivation fix.

## 2026-08-23 — Block decomposition + selection-reliability correction

- Single-block arms (`h128-dscan-rt-only`, `h128-dscan-licks-only`, 3 seeds each)
  scored: at D≥100 lick counts carry ~98% of the combined effect, RT-only ~7%
  but statistically real and growing with D. Mildly sub-additive.
- Corrected the D≤30 mechanism: not "scored past the peak" (held-out uses
  `best_eval`, the best-validation checkpoint) but that `best_eval` selects on a
  within-subject signal which has itself collapsed at small D
  (corr with held-out ≈0 at D≈10 → 1.0 at D≈614). D≤30 excluded from estimates.

## 2026-08-22 — Three-way decomposition resolved

- Shuffled control (`h128-dscan-shuffled`) completed the 3-arm × 5-D grid. The
  information contribution (ON−shuffled) is flat ~+0.0075; the apparent
  growth-with-D of the raw ON−OFF curve is a vanishing input-width overfitting
  penalty, not scaling information. Headline figure is the three-way
  decomposition, not the paired curve.
- Paired timing-OFF arms (`h128-dscan-off-bigD`, `h128-dscan-off-smallD`) run at
  every D after the `d100-bridge` OFF arm failed to reproduce study-01's H128
  band — diagnosed as `data.snapshot` pinning changing the rank-based cohort
  draw, not code drift. Convention adopted: always pin `data.snapshot`.
- Held-out-path fix: the held-out loader never derived the timing columns, so the
  primary metric was silently absent for the timing arms; fixed and the finished
  arms re-scored via `resume_heldout_beaker.py` (no retraining).

## 2026-08-21 — Study created + timing arm launched

- New study on branch `feat/rt-lick-inputs` (wrapper + dispatcher). Adds
  previous-trial reaction time (log-encoded) and per-side lick counts as GRU
  inputs; the only model/data change vs study-01 is
  `data.timing_features.enabled: false → true` (input tensor widens 3 → 6, output
  unchanged so the metric stays comparable to study-01).
- Wrapper payload: `utils/trial_timing_features.py`, the float-safe dataset
  builder in `data_loaders/mice.py`, tests, config blocks, and
  `analysis/calibrate_timing_features.py` (the logistic-probe calibration).
- Variants `d100-bridge`, `h128-dscan`, `h128-dscan-onprem` launched; W&B project
  `mice_rt_lick_scaling`.
