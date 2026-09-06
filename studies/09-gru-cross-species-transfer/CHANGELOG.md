# Changelog

## 2026-09-05

- Added deterministic representative behavior-session plots for Grossman, Chen,
  and Zid to Result 2 using the pinned basic-analysis session plotter.
- Consolidated Result 1 into Result 2, fixed author-line styling, added task and
  subject-count panel labels, and added author-relative paired subject-level
  violin plots with Wilcoxon tests across common Q, published baselines, and
  every GRU source D.
- Refit the Grossman, Chen, and Zid author-aligned models subject by subject on
  the matched adaptation halves and added a frozen comparison against common Q
  and D=614 transferred GRU.
- Added author-aligned Grossman meta-learning, Chen RLCK, and Zid traditional
  RLCK/history-kernel-2 foraging-RL model definitions and HPC variants.
- Pinned the 15 H=128 Study 01 source GRUs and their committed W&B artifacts.
- Added matched-half GPU grids for Grossman, Chen, and Zid.
- Added the matched subject-level Q-learning CPU runner and HPC SLURM array.
- Kept Zid as one within-session prefix/suffix condition with no session-count K.
- Added a checksum-pinned pyarrow wheel mount for the current GPU image.
- Completed all 45 GRU cells and three matched Q baselines with no failed jobs.
- Added the frozen matched-half result, exact trial-alignment audit, D-scaling
  figure, and GRU-versus-Q report.

## 2026-09-04

- Created the top-three open-data starter suite for Grossman, Chen, and Zid.
- Pinned repository versions, public licenses, file identifiers, and checksums.
- Added source-specific canonical adapters and exact release-count audits.
- Emit canonical Parquet tables for an interoperable wrapper boundary.
- Use Zid's official MATLAB representation with a locally pinned SHA-256,
  avoiding executable pickle deserialization.
- Froze odd/even session manifests for the mouse cohorts and a state-preserving
  prefix/suffix manifest for the single-session human cohort.
- Verified all three canonical outputs through the wrapper loader.
