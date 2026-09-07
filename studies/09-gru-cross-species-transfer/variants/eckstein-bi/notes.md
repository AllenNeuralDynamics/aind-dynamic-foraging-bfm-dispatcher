# Eckstein (human) Bayesian-inference co-winner

Fit Eckstein et al.'s four-parameter hidden-state Bayesian filter on each
participant's complete first-half prefix. Replay that prefix with the fitted
parameters to initialize the state belief, then score only the immutable
second-half suffix used by GRU and common Q.

The original study used hierarchical Bayesian population fitting. Study 09 uses
subject-level maximum likelihood to keep the amount of participant data and the
held-out trials matched across models; this deviation is reported explicitly.
This is a CPU-only Allen HPC job and must not be submitted to Beaker.
