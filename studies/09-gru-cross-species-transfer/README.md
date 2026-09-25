# Study 09: GRU cross-species and cross-dataset transfer

Issues: dispatcher [#32](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/32),
[#126](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/126),
and [#127](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/127);
generalization drivers [#140](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/140);
embedding capacity [#148](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/148);
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

1. Grossman (mouse): 48 mice, blockwise independent reward probabilities.
2. Chen (mouse): 32 mice, eight restless-bandit sessions.
3. Zid (human): 258 humans, schedule-matched to Chen (mouse).

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
`kwak`, `miller`, `findling`, `tang`, `alsio`, `eckstein`, `costa`,
`lopez_mouse`, and `hattori`. Run `make validate` after generation to verify the exact release
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

Grossman (mouse) uses only the dynamicForaging/behavior branch. This is the curated
48-mouse behavior cohort and avoids duplicate sessions also present under neural
or chemogenetics analyses. CSminus no-go trials and CSplus trials without a
left/right decision are excluded. Left is encoded 0 and right 1.

Chen (mouse) retains all 256 source files. Source choices 1/2 become canonical choices
0/1. All eight sessions remain separate, including short sessions.

Zid (human) uses the official Experiment 1 MATLAB file rather than the equivalent
Python pickle, avoiding executable deserialization of downloaded data. The
first 25 fixed-schedule practice trials are excluded, leaving the 300 main
trials for each participant.

Hattori (mouse) uses every mouse in the untreated longitudinal Imaging cohort but
only each mouse's 15th probabilistic-reversal session onward. The paper calls days
1–14 early and day 15 onward late/expert-stage behavior. This mature-only rule
retains 292 dated sessions and 141,488 binary-choice trials from seven mice; the
separate inactivation and paAIP2 cohorts are not mixed into the within-condition
transfer estimand. Source actions `1=right, 2=left` become canonical
`1=right, 0=left`; alarm and miss trials are excluded because they contain no
binary choice.

## Frozen split contract

For Grossman (mouse) and Chen (mouse), sessions are ordered chronologically within subject. The
first, third, fifth, and later odd-positioned sessions are adaptation data; the
second, fourth, sixth, and later even-positioned sessions are test data. This is
the current GRU study convention (eval_every_n=2). Therefore a K-session
few-shot run takes the first K sessions *within that odd-positioned adaptation
sequence*, not the first K sessions before the odd/even split.

Zid (human) and Eckstein (human) have one main session per person, so inventing pseudo-sessions would reset
the recurrent and Q-learning states at an artificial boundary. Instead, the
manifest assigns each subject's complete first half to an adaptation prefix and
the second half to a test suffix. Zid (human) therefore uses trials 0–149 and 150–299;
Eckstein (human) uses the analogous subject-specific midpoint because released session
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
GPU-only Beaker grids. The Bari2019 common-Q variants run as CPU-only SLURM arrays on
Allen HPC. Both model
families consume the same generated Parquet table and split manifest, and both
emit the wrapper's canonical `test_trial_predictions.csv` and
`test_metrics.json` outputs for an exact trial-key parity check.

GPU tasks mount the checksum-backed canonical dataset
`study09-external-v1v2-20260905`
(`01M1TMMETY8M1V0F0E2V6XP146`). The choice-corrected Kwak (mouse) rerun mounts the
immutable replacement `study09-kwak-choicefix-20260906`
(`01M1VGY7MV148S9GWM2HW38FRQ`), which inverts the released `0=right, 1=left`
choice bit to the canonical `0=left, 1=right` convention while retaining the
same split manifest. That manifest was later found to adapt on CNO sessions and
test on DMSO sessions, so every Kwak (mouse) result is quarantined pending a
DMSO/control-only odd/even-session rerun. The current GPU image predates the wrapper's
declared `pyarrow` dependency, so tasks also mount dependency bundle
`01M1RDVWF18JF5QMEB618WJPSF`, verify the wheel checksum, and install
`pyarrow==21.0.0` before reading the canonical Parquet table.
Hattori (mouse) uses the separate mature-only canonical dataset
`study09-hattori-mature-canonical-20260907`
(`01M1ZED14MFW8BC8728QZ3Z30F`), containing only session 15 onward.

The Bari2019 common-Q comparison is consolidated into
[Result 1](analysis/reports/r1-author-aligned-baselines.md).
All 14 raw cohort matrices and all 210 E4 GRU cells are frozen. Kwak (mouse) and
its 15 cells are quarantined, leaving 13 valid cohorts and 195 valid E4 cells.
Exact ordered trial-key equality passes between every GRU cell and its cohort's
Bari2019 baseline; parity alone does not rescue an invalid scientific split.
Across the 13 valid cohorts at D=614, the exploratory subject-paired result favors
GRU for Grossman (mouse), Chen (mouse), Lebedeva (mouse), López-Yépez (mouse),
and Hattori (mouse); it favors Bari2019 common Q for Zid (human), Miller (rat), Findling (human), and
Eckstein (human); Beron (mouse), Tang (macaque), Alsiö (rat), and Costa (macaque) are
unresolved at the unadjusted 0.05 level. Every valid cohort improves in trial-pooled
GRU likelihood from D=10 to D=614,
although several curves peak at D=100 or D=300.

Kwak (mouse) was rerun after correcting the release's reversed left/right choice
labels, but the later split audit found that adaptation used CNO and testing used
DMSO. Those likelihoods and embeddings are retained only for provenance and are
excluded from figures, rankings, direction counts, and inference. Readmission
requires retaining DMSO/control sessions only, adapting on chronological odd
DMSO sessions, and testing on chronological even DMSO sessions.

Zid (human) is the one aggregation reversal: its trial-pooled D=614 score favors GRU by
0.00936, but its arithmetic mean subject difference favors Q by 0.00171 and its
median difference favors Q by 0.00870. All subjects contribute 150 held-out
trials, so this reflects geometric pooled versus arithmetic subject-level
summaries in a heterogeneous distribution, not unequal trial weighting.

## Author-aligned baselines

Bari2019 (`L1F1CK1`) remains the controlled common-Q baseline across datasets.
Author-aligned variants test whether conclusions depend on using a common family
instead of the model selected by each dataset's authors:

- `grossman-meta-learning`: uncertainty-dependent asymmetric meta-learning;
- `chen-rlck`: the selected four-parameter RL plus choice-kernel model;
- `zid-history-kernel`: both the best traditional RLCK model and the best
  overall history-kernel-2 foraging-RL model;
- `lebedeva-pr`: probabilistic reversal model;
- `beron-rflr`: reward-forgetting logistic regression;
- `miller-rhg`: reward-history-gradient model;
- `findling-weber`: Weber-imprecision Bayesian inference;
- `eckstein-rl` and `eckstein-bi`: the two reported co-winning families;
- `hattori-q-learning`: Hattori2019 (`L2F1CK0`), with separate rewarded and
  unrewarded learning rates and no choice kernel.

They use the same subject-level adaptation observations and identical held-out
trial keys as the GRU and common-Q comparisons in Result 1. These fits are
CPU-only SLURM jobs on Allen HPC; they must not be sent to Beaker.

The completed consolidated comparison is
[Result 1](analysis/reports/r1-author-aligned-baselines.md).
The author-selected refit beats Bari2019 common Q for Chen (mouse), Lebedeva
(mouse), Miller (rat), and both Eckstein (human) co-winners; Bari2019 is better
for Grossman (mouse), Zid (human), Beron (mouse), Findling (human), and Hattori
(mouse). Result 1 reports every subject-paired comparison and the known
paper-parity limitations. In particular, Hattori2019 does not reproduce the
paper's cross-validated L2 penalty, so equation parity is high but paper-level
fit-procedure parity is moderate.

## Subject embedding space

[Result 2](analysis/reports/r2-embedding-space.md) uses the three D=614 source
seeds to ask where unseen subjects land after embedding-only adaptation. PCA is
fit separately to the 614 source-training AIND mice in each seed; the primary
control is the 149 held-out AIND mice, which were also unseen by the frozen GRU
core. Separate source-fitted PCA views and full-dimensional Mahalanobis analyses
are reported for E=4 and E=8.

Held-out AIND mice remain calibrated to the source distribution, with only
4.0%--5.4% outside the E=4 empirical 95th percentile and 6.0%--7.4% outside the
E=8 threshold. Every external cohort's median distance exceeds the held-out-AIND
median in all three seeds in both spaces. Raw E4 and E8 distances are not
directly comparable, but the external-cohort distance ranks are strongly stable
(Spearman rho=0.923). Distance is descriptive: species, task structure, reward
contingency, session duration, and adaptation-data volume vary together and
cannot be isolated by this survey.

## Generalization drivers

[Result 3](analysis/reports/r3-generalization-drivers.md) is a cross-cohort
meta-analysis of the frozen GRU, Bari2019 common-Q, embedding, and task-design artifacts.
It compares subject-balanced E4 and E8 D=614 GRU improvement and
dimension-specific embedding displacement with an evidence-backed categorical
distance from the AIND source tasks. For
the seven primary cohorts with complete trial-wise arm probabilities, it also tests
an empirical reward-schedule distance from Grossman (mouse). Primary categorical
inference uses nine cohorts; a separate all-valid sensitivity uses 13 and still
excludes quarantined Kwak (mouse). Species remains descriptive
because species, study, apparatus, and task design are confounded.

## Subject-embedding capacity ablation

[Result 4](analysis/reports/r4-embedding-dimension-transfer.md) compares E=4
versus E=8 transfer in issue #148. Both
source dimensions use D=614, H=128, three seeds, the same source snapshot and
training recipe, and the same external 500-step embedding adaptation. The
strict paired current-code comparison covers Grossman (mouse), Lebedeva
(mouse), Miller (rat), Findling (human), Eckstein (human), and Hattori (mouse). The remaining
seven cohorts compare current E8 with the frozen historical E4 screen on exactly
the same held-out trial keys. The two E8 expansion shards completed as Beaker experiments
`01M1YRP7ASQRQN4Y7MZ2HEQAXG` and `01M1YRT8B20NMJ3JAHNVC0RMF4`; Hattori (mouse)
completed in `01M1ZFJ3BPJSZNJ403DPD3FS6C` (E4) and
`01M1ZFJ1Z9BP5GFCM8RPTG119Z` (E8). Kwak (mouse)
remains quarantined and was not run. All 21 expansion tasks succeeded, their
W&B artifacts were frozen, and all 18 mature Hattori tasks also succeeded. Exact
held-out trial-key parity passed. Across all 13 valid cohorts, E8 has higher mean
held-out likelihood in 10 and lower in Grossman (mouse), Zid (human), and Hattori
(mouse). The seven historical-E4 comparisons remain a weaker causal dimension
ablation than the six paired current-code reruns.
