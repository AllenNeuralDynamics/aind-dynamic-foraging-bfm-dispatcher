# Lebedeva (mouse) PR baseline

Fit the five-parameter perseveration/reward-learning model selected by Lebedeva
et al. separately for each mouse on the same odd-session adaptation half, then
score the same even-session test half used by GRU and common Q.

The equation and source-code audit is pinned in the models repository. The
finite optimizer bounds used in place of the authors' unbounded log-odds
weights are reported as a reproduction deviation. This is a CPU-only Allen HPC
job and must not be submitted to Beaker.
