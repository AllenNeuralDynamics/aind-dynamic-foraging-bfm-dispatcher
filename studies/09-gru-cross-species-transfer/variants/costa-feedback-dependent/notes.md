# Costa (macaque) feedback-dependent RL baseline

Fit Costa's Rescorla-Wagner family with separate rewarded and unrewarded
learning rates, augmented with a fixed choice bias, independently
for each macaque on the matched odd-session adaptation half. Then score the
held-out even sessions. Latent values reset to 0.5 at each real held-out
session, exactly as in the Bari2019 common-Q and E4 GRU matched-half comparison.

The paper's equation family has one rewarded learning rate, one omission
learning rate, and inverse temperature, with no unchosen-value forgetting or
choice-kernel term. The matched comparison adds one log-odds intercept because
Bari2019 fits an intercept and the GRU can represent a stable subject preference.
For Costa's shape-coded canonical choice, this is a stimulus-shape preference,
not a screen-side preference.

Two requested diagnostics challenge the remaining gap without changing the
held-out estimand:

- rerun Bari2019 common Q from scratch on the same corrected split;
- add Bari2019's one-step choice kernel (`CK1`) to the dual-rate plus bias model,
  while continuing to omit unchosen-value forgetting.

The CK1 sensitivity therefore has five fitted parameters: rewarded learning
rate, unrewarded learning rate, inverse temperature, shape-choice bias, and
choice-kernel relative weight. The choice-kernel step size is fixed at one.

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
