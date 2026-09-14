# Costa (macaque) feedback-dependent RL baseline

Fit the paper-selected Rescorla-Wagner model with separate rewarded and
unrewarded learning rates independently for each macaque on the matched
odd-session adaptation half, then score the held-out even sessions. Values
reset to 0.5 at every real 80-trial stimulus block.

The equations, block resets, and initial value of 0.5 are paper-aligned. The
paper's phase-, schedule-, and session-specific fits are replaced by one
matched-half parameter set per subject, and cross-block carryover for repeated
stimuli cannot be reconstructed from the retained binary option identity. Those
distinctions remain explicit in the report. CPU-only
Allen HPC job; never submit to Beaker. The job reads the persistent canonical
suite after verifying that its Parquet, split-manifest, and audit-file SHA-256
digests match the local audited copies; this avoids a redundant Zenodo download.

The first completed fit used the invalid original canonical reward mapping and
is excluded. The corrected fit must use
`/allen/aind/scratch/han.hou/datasets/study09-costa-reward-fix-20260914/canonical`,
where reward is released column 11.

Tracking: dispatcher #160.
