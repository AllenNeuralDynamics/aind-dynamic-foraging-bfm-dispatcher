# Zid history-kernel baselines

Fit both author-selected reference models to the fixed 150-trial adaptation
prefix and score only the held-out 150-trial suffix while replaying the complete
session to preserve model state:

- history-kernel-2 traditional RL (best compare-alternatives model; Eq. 19);
- history-kernel-2 foraging-RL (best model overall; Eq. 22).

Both fits use independent value and history inverse temperatures exactly as in
the paper. This is a two-task CPU-only SLURM array on Allen HPC and must not be
submitted to Beaker.
