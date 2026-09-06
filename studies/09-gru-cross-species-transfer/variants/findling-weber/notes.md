# Findling (human): Weber-imprecision Bayesian inference

Tracks #145. This variant fits the author-selected Weber-imprecision Bayesian
inference model to each subject's schema-v1 adaptation sessions and scores the
same immutable test sessions as the GRU and common-Q baselines.

The fit uses the released 1,000-point Sobol parameter grid and the release's
two-particle likelihood estimator with a recorded seed. Held-out probabilities
use 256 particles to reduce Monte Carlo noise. The Beta belief state resets at
every real session. A particle-count sensitivity check is required before small
differences are interpreted.

The primary run retains the released two-particle fit. A separate sensitivity
run sets `FINDLING_FIT_PARTICLES=64` while leaving the parameter grid,
adaptation/test membership, evaluation particles, and seed unchanged.
