# Costa (macaque) feedback-dependent RL baseline

Fit the paper-selected Rescorla-Wagner model with separate rewarded and
unrewarded learning rates independently for each macaque on the matched
odd-session adaptation half, then score the held-out even sessions.

The equations and initial value of 0.5 are paper-aligned. The paper's
phase-specific original fitting procedure is replaced by the shared matched-half
subject fit, and that distinction remains explicit in the report. CPU-only
Allen HPC job; never submit to Beaker. The job reads the persistent canonical
suite after verifying that its Parquet, split-manifest, and audit-file SHA-256
digests match the local audited copies; this avoids a redundant Zenodo download.

Tracking: dispatcher #160.
