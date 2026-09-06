# Expanded common-Q matched-half baseline

Fit the same sticky, forgetful subject-level Q-learning family used in the
completed Grossman, Chen, and Zid screen to each of the ten newly admitted
cohorts. Every fit uses the identical full adaptation half and held-out trials
as its GRU comparison.

This is a CPU-only Allen HPC SLURM array, capped at four concurrent tasks. It
reads the frozen canonical bundle staged at `EXTERNAL_DATA_ROOT`; it must never
be submitted to Beaker. Grossman, Chen, and Zid are deliberately absent because
their completed common-Q cells are reused.
