# Changelog

## 2026-09-07

- Extended E=8 D=614 transfer to the seven remaining valid cohorts in 21
  GPU-only Beaker tasks; froze all artifacts with exact historical-E4 trial-key
  parity and updated Report 1 so every displayed cohort has an E8 overlay.
- Froze all six paired source artifacts, completed the E=8 diagnostic transfer,
  and launched the matched 15-cell E=4 transfer as Beaker experiment
  `01M1YBQERTCR0HF6Y749X7M5HT`.
- Completed the E=4 transfer, verified exact E4/E8/historical held-out trial-key
  parity, and published Result 4. E8 is neutral on Grossman (mouse) and
  Lebedeva (mouse), but improves Miller (rat), Findling (human), and Eckstein
  (human), supporting expansion to the remaining non-quarantined cohorts.

## 2026-09-06

- Prepared the paired current-code E=4 versus E=8 D=614 transfer ablation for
  five diagnostic cohorts and reserved Result 4 for its subject-level report.
- Implemented and ran the primary-set author-selected models for Lebedeva
  (mouse), Beron (mouse), Miller (rat), Findling (human), and both Eckstein
  (human) co-winners; added a 64-particle Findling fitting sensitivity.
- Split the generalization, robustness/scaling, and task-design figures into
  cumulative primary-only, primary-plus-stress-test, and all-valid views so
  each inclusion criterion remains visually legible.
- Quarantined Kwak (mouse) because the frozen manifest adapts on CNO sessions
  and tests on DMSO sessions; documented a DMSO/control-only odd/even rerun and
  excluded the current result from every plot and inference.
- Added primary, stress-test, descriptive-only, and quarantined cohort tiers;
  primary meta-analysis now uses eight cohorts, the all-valid sensitivity uses
  12, and empirical schedule inference uses six.
- Standardized paper labels as `Author (species)` throughout Study 09 reports.
- Added an outcome-blind categorical task-design matrix, nearest-AIND-prototype
  distances, and subject-balanced empirical reward-schedule features for the
  six primary cohorts with complete arm probabilities.
- Related categorical and empirical schedule distance to D=614 GRU advantage
  and adapted embedding displacement with permutation, bootstrap, leave-one-out,
  and feature-wise FDR sensitivity analyses.

## 2026-09-05

- Added Result 3 relating subject-balanced GRU transfer advantage and source-D
  scaling to external embedding displacement and common-Q predictability.
- Renumbered the two live reports after removing the obsolete first-round
  report: the consolidated decision report is now Result 1 and the embedding
  analysis is Result 2.
- Expanded the checksum-pinned binary dataset suite to 13 admitted cohorts
  using only the existing v1/v2 contracts; documented five skipped cohorts.
- Added exact held-out-count/digest validation, one-subject loader and Q
  preflights, a frozen Beaker input bundle, ten GPU-only GRU variants, and one
  CPU-only HPC common-Q array. New author-model work remains behind Stage B.
- Distinguished subject-level means from medians in the consolidated figure
  and reported, for every dataset, how D=614 GRU improvement correlates with
  author-model likelihood.
- Added a source-fitted, seed-separated PCA analysis of D=614 embeddings for
  source and held-out AIND mice plus all three external cohorts, with a full-4D
  source-distance audit and immutable artifact/file provenance.
- Require any preinstalled pyarrow to match the pinned 21.0.0 runtime instead
  of bypassing the wheel-version contract.
- Removed the obsolete first-round report after its GRU-versus-Q content was
  consolidated into the current Result 1, and updated all report-consumer
  references.
- Added deterministic representative behavior-session plots for Grossman (mouse),
  Chen (mouse), and Zid (human) to the current Result 1 using the pinned basic-analysis session
  plotter.
- Consolidated the first-round report into the current Result 1, fixed
  author-line styling, added task and
  subject-count panel labels, and added author-relative paired subject-level
  violin plots with Wilcoxon tests across common Q, published baselines, and
  every GRU source D.
- Refit the Grossman (mouse), Chen (mouse), and Zid (human) author-aligned models subject by subject on
  the matched adaptation halves and added a frozen comparison against common Q
  and D=614 transferred GRU.
- Added author-aligned Grossman (mouse) meta-learning, Chen (mouse) RLCK, and Zid (human) traditional
  RLCK/history-kernel-2 foraging-RL model definitions and HPC variants.
- Pinned the 15 H=128 Study 01 source GRUs and their committed W&B artifacts.
- Added matched-half GPU grids for Grossman (mouse), Chen (mouse), and Zid (human).
- Added the matched subject-level Q-learning CPU runner and HPC SLURM array.
- Kept Zid (human) as one within-session prefix/suffix condition with no session-count K.
- Added a checksum-pinned pyarrow wheel mount for the current GPU image.
- Completed all 45 GRU cells and three matched Q baselines with no failed jobs.
- Added the frozen matched-half result, exact trial-alignment audit, D-scaling
  figure, and GRU-versus-Q report.

## 2026-09-04

- Created the top-three open-data starter suite for Grossman (mouse), Chen (mouse), and Zid (human).
- Pinned repository versions, public licenses, file identifiers, and checksums.
- Added source-specific canonical adapters and exact release-count audits.
- Emit canonical Parquet tables for an interoperable wrapper boundary.
- Use Zid (human)'s official MATLAB representation with a locally pinned SHA-256,
  avoiding executable pickle deserialization.
- Froze odd/even session manifests for the mouse cohorts and a state-preserving
  prefix/suffix manifest for the single-session human cohort.
- Verified all three canonical outputs through the wrapper loader.
