# Tang (macaque) block-type-specific RL baseline

Fit the paper's feedback-dependent Rescorla-Wagner equations separately for its
two task domains: object values in What blocks and saccade-direction values in
Where blocks. Each subject therefore has separate positive/negative learning
rates and inverse temperature for the two block types. Values reset at each
80-trial block, consistent with the novel-object design.

The runner converts object-choice probabilities back to canonical right-choice
probabilities before applying the same immutable held-out trial contract used
by GRU and Bari2019 common Q. Tang (macaque) has only two subjects, so the result
is descriptive. CPU-only Allen HPC job; never submit to Beaker.

Tracking: dispatcher #161.
