# Grossman uncertainty meta-learning baseline

Fit the uncertainty-dependent meta-learning model selected by Grossman et al.
separately for each mouse on the same odd-session adaptation half, then score
the same even-session test half used by the GRU and common Q-learning baseline.

The implementation follows the published asymmetric value update, unchosen-value
retention, expected-uncertainty update, and unexpected-uncertainty modulation of
the negative learning rate. This is a CPU-only SLURM job on Allen HPC and must
not be submitted to Beaker.
