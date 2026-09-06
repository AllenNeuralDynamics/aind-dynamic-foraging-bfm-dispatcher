# Study 09: GRU cross-species and cross-dataset transfer

Issues: dispatcher [#32](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/32),
[#126](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/126),
and [#127](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/127);
Stage-A expansion [#134](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/134)
through [#138](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/138);
author-aligned baselines [#131](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/131),
[#132](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/132),
and [#133](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/133);
wrapper [#91](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/91)
and [#92](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/92).

This suite tests whether the GRU core learned from AIND dynamic foraging
transfers to external binary two-arm-bandit behavior. The completed starter
panel is:

1. Grossman, Bari & Cohen: 48 mice, blockwise independent reward probabilities.
2. Chen et al.: 32 mice, eight restless-bandit sessions.
3. Zid et al. Experiment 1: 258 humans, schedule-matched to Chen.

Stage A expands that panel only with complete cohorts that fit the existing v1
or v2 split contracts. The cohort-by-cohort admission audit, exact counts,
licenses, author-selected models, and reproduction feasibility are frozen in
[`DATASET_SURVEY.md`](DATASET_SURVEY.md). Mixed split structures are skipped;
schema v3 is intentionally not implemented.

## Build the suite

The command below downloads pinned public releases, verifies their repository
checksums, converts them to the wrapper's canonical trial table, writes an
explicit split manifest, and fails if subject/session/trial counts differ from
the audited release.

    make suite

Set CACHE_ROOT to place the uncommitted raw and canonical data elsewhere. After
the first successful download, use make rebuild to work entirely from the
checksum-verified local source files.

Generated files are:

- `$CACHE_ROOT/canonical/<dataset>.parquet`: one row per valid decision trial;
- `$CACHE_ROOT/canonical/<dataset>.split.json`: the immutable adaptation/test split;
- `$CACHE_ROOT/canonical/<dataset>.audit.json`: provenance, checksums, and counts.

The admitted dataset keys are `grossman`, `chen`, `zid`, `lebedeva`, `beron`,
`kwak`, `miller`, `findling`, `tang`, `alsio`, `eckstein`, `costa`, and
`lopez_mouse`. Run `make validate` after generation to verify the exact release
counts, binary choices/rewards, v1/v2 contract, deterministic manifest
regeneration, and non-empty adaptation/test partitions. Its committed summary
is `analysis/dataset_suite_validation.json`.

With the Makefile default, `$CACHE_ROOT` is `data-cache` relative to this study directory.

The six required columns are subject_id, ses_idx, trial, animal_response,
rewarded, and earned_reward. Choices and rewards are binary. The adapters retain
source trial numbers, reward probabilities, and available cohort metadata as
additional columns.

## Cohort-specific decisions

The complete expansion audit is in [`DATASET_SURVEY.md`](DATASET_SURVEY.md).
The starter decisions below remain frozen because their completed cells are
reused rather than rerun.

Grossman uses only the dynamicForaging/behavior branch. This is the curated
48-mouse behavior cohort and avoids duplicate sessions also present under neural
or chemogenetics analyses. CSminus no-go trials and CSplus trials without a
left/right decision are excluded. Left is encoded 0 and right 1.

Chen retains all 256 source files. Source choices 1/2 become canonical choices
0/1. All eight sessions remain separate, including short sessions.

Zid uses the official Experiment 1 MATLAB file rather than the equivalent
Python pickle, avoiding executable deserialization of downloaded data. The
first 25 fixed-schedule practice trials are excluded, leaving the 300 main
trials for each participant.

## Frozen split contract

For Grossman and Chen, sessions are ordered chronologically within subject. The
first, third, fifth, and later odd-positioned sessions are adaptation data; the
second, fourth, sixth, and later even-positioned sessions are test data. This is
the current GRU study convention (eval_every_n=2). Therefore a K-session
few-shot run takes the first K sessions *within that odd-positioned adaptation
sequence*, not the first K sessions before the odd/even split.

Zid and Eckstein have one main session per person, so inventing pseudo-sessions would reset
the recurrent and Q-learning states at an artificial boundary. Instead, the
manifest assigns each subject's complete first half to an adaptation prefix and
the second half to a test suffix. Zid therefore uses trials 0–149 and 150–299;
Eckstein uses the analogous subject-specific midpoint because released session
lengths vary. Evaluation runs the prefix to establish the state at the boundary,
then scores the suffix without changing fitted parameters. Observed choices and
outcomes still update the recurrent or Q state online during the suffix; only
the learned parameters remain frozen.

## Benchmark matrix

Each external dataset is held out from foundation-model training. Evaluate:

- zero-shot: frozen GRU core and the prespecified new-subject embedding
  initialization;
- few-shot: frozen GRU core, adapt only the new-subject embedding on K
  adaptation sessions (or a declared prefix budget);
- half-data shot: the full adaptation half, again changing only the embedding;
- subject-level Q-learning: fit parameters on the identical adaptation
  observations and score the identical test observations.

Stage A runs the full matched adaptation half only. There is no K condition for
v2: the complete first-half prefix adapts the embedding or Q parameters, and the
second-half suffix is scored after replaying the prefix to establish the state.

The primary comparison is mean per-trial log likelihood on the test partition,
paired by subject. Report normalized likelihood, Brier score, accuracy, and
calibration as descriptive secondary metrics. Fit preprocessing, hyperparameters,
random seeds, split manifests, source checksums, and model checkpoints are part
of the run provenance.

## Current consolidated report

Issue #126 first evaluates the fully matched half-data condition. The GRU panel
uses the fixed H=128 Study 01 source models at
`D={10,30,100,300,614}` with seeds 0--2. For every cell, only the new-subject
embedding is optimized for 500 steps at learning rate 0.001; the GRU core stays
frozen and the target test partition never selects a checkpoint. Source run IDs
and immutable W&B artifact digests are recorded in `source_runs.json`.

The starter `gru-*-matched-half` variants and expansion dataset variants run as
GPU-only Beaker grids. The common-Q variants run as CPU-only SLURM arrays on
Allen HPC. Both model
families consume the same generated Parquet table and split manifest, and both
emit the wrapper's canonical `test_trial_predictions.csv` and
`test_metrics.json` outputs for an exact trial-key parity check.

GPU tasks mount the checksum-backed canonical dataset
`study09-external-v1v2-20260905`
(`01M1TMMETY8M1V0F0E2V6XP146`). The current GPU image predates the wrapper's
declared `pyarrow` dependency, so tasks also mount dependency bundle
`01M1RDVWF18JF5QMEB618WJPSF`, verify the wheel checksum, and install
`pyarrow==21.0.0` before reading the canonical Parquet table.

The common-Q comparison is consolidated into
[Result 2](analysis/reports/r2-author-aligned-baselines.md).
At D=614, trial-pooled GRU normalized likelihood exceeds Q-learning by 0.01358
on Grossman, 0.00528 on Chen, and 0.00936 on Zid. The frozen input also proves
exact ordered trial-key equality between every GRU cell and its Q baseline.

## Author-aligned baselines

The common Q-learning model remains the controlled baseline across datasets.
Three additional variants test whether that conclusion depends on using a
generic family instead of the model selected by each dataset's authors:

- `grossman-meta-learning`: uncertainty-dependent asymmetric meta-learning;
- `chen-rlck`: the selected four-parameter RL plus choice-kernel model;
- `zid-history-kernel`: both the best traditional RLCK model and the best
  overall history-kernel-2 foraging-RL model.

They use the same subject-level adaptation observations and identical held-out
trial keys as the GRU and common-Q comparisons in Result 2. These fits are
CPU-only SLURM jobs on Allen HPC; they must not be sent to Beaker.

The completed consolidated comparison is
[Result 2](analysis/reports/r2-author-aligned-baselines.md).
The author-selected model beats common Q only for Chen (+0.00392 normalized
likelihood). The D=614 transferred GRU remains above the author-selected model
by +0.01559 on Grossman, +0.00136 on Chen, and +0.03057 on Zid.

No new author-selected model is implemented or run during Stage A. The
feasibility table in `DATASET_SURVEY.md` is the explicit stop gate before Stage
B; new model families require a dataset-by-dataset decision after reviewing the
expanded GRU-versus-common-Q result.

## Subject embedding space

[Result 3](analysis/reports/r3-embedding-space.md) uses the three D=614 source
seeds to ask where unseen subjects land after embedding-only adaptation. PCA is
fit separately to the 614 source-training AIND mice in each seed; the primary
control is the 149 held-out AIND mice, which were also unseen by the frozen GRU
core. A full four-dimensional Mahalanobis-distance analysis accompanies the 2D
PC views.

Held-out AIND mice remain calibrated to the source distribution, with only
4.0%--5.4% outside its empirical 95th percentile. Grossman mice are moderately
shifted (14.6%--22.9%), whereas Chen mice (100%) and Zid humans
(97.3%--98.1%) are strongly displaced in all three seeds. Because Chen and Zid
share the restless random-walk task while Grossman is a mouse blockwise task
closer to AIND dynamic foraging, task structure is a better first explanation
than species alone; the datasets do not isolate those factors experimentally.
