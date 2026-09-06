# Miller (rat) RHG baseline

Fit the seven-parameter reward-seeking/habit/gambler's-fallacy model selected by
Miller et al. separately for each rat on the same odd-session adaptation half,
then score the same even-session test half used by GRU and common Q.

The executable equation audit is pinned in the models repository, including
the published `[-V, +V]` policy logits. This is a CPU-only Allen HPC job and
must not be submitted to Beaker.
