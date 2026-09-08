# Hattori (mouse) author-aligned Q-learning baseline

Fit Hattori et al.'s rewarded/unrewarded learning-rate model with unchosen-value
forgetting separately for each mouse on the same odd-session adaptation half
after excluding sessions 1–14 under the paper's late/expert threshold, then
score the same even-session mature test half used by GRU and common Q.

The equations match Hattori et al. (2023), Eqs. 17–19, and the existing
`Hattori2019` family. The matched-half fit uses bounded differential evolution
without the paper's tenfold-selected L2 penalty, so equation parity is high but
paper-level fitting parity is moderate. This is a CPU-only Allen HPC job.
