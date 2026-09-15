# Costa (macaque) feedback-dependent RL baseline

Fit the paper-selected Rescorla-Wagner model with separate rewarded and
unrewarded learning rates independently for each macaque on the matched
odd-session adaptation half, then score the held-out even sessions. Latent
values reset to 0.5 at each real held-out session, exactly as in the Bari2019
common-Q and E4 GRU matched-half comparison.

The equation family is paper-aligned: one rewarded learning rate, one omission
learning rate, and inverse temperature (no unchosen-value forgetting term).
The paper instead fits individual session/schedule phases and resets values at
each 80-trial stimulus block, with limited carryover for repeated stimuli.
That is appropriate for its within-session lesion analysis, but it is not the
same held-out transfer estimand. This variant deliberately uses one
subject-level adaptation fit and the shared session-reset held-out protocol.
CPU-only
Allen HPC job; never submit to Beaker. The job reads the persistent canonical
suite after verifying that its Parquet, split-manifest, and audit-file SHA-256
digests match the local audited copies; this avoids a redundant Zenodo download.

The first completed fit used the invalid original canonical reward mapping and
is excluded. The previous block-reset corrected fit is a paper-protocol
diagnostic only and is not comparable to the E4 GRU/Bari panel. The replacement
matched-half fit must use
`/allen/aind/scratch/han.hou/datasets/study09-costa-reward-fix-20260914/canonical`,
where reward is released column 11.

Tracking: dispatcher #160.
