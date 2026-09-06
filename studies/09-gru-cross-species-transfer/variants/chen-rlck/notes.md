# Chen four-parameter RLCK baseline

Fit Chen et al.'s selected four-parameter RLCK model separately for each mouse
on the same odd-session adaptation half, then score the same even-session test
half used by the GRU and common Q-learning baseline.

The four fitted parameters are reward learning rate, value inverse temperature,
choice-kernel learning rate, and an independent choice-kernel inverse
temperature. This is a CPU-only SLURM job on Allen HPC and must not be submitted
to Beaker.
