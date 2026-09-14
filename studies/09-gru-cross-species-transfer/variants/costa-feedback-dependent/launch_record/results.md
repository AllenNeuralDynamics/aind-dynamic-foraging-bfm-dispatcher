# Costa (macaque) feedback-dependent RL result

- **W&B group:** `costa-feedback-dependent@slurm-26383388`
- **W&B run:** [256a1vmk](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/256a1vmk)
- **SLURM job:** `26383388`
- **Completed:** 2026-09-14 16:34:47 PT
- **Status:** success
- **Held-out trials:** 162,960
- **Normalized likelihood:** 0.5968516530
- **Accuracy:** 0.7356345115
- **Brier score:** 0.1729063136

The prediction table has the same ordered trial-key digest as the historical
Costa (macaque) split and the corrected common-Q/GRU reruns retain those trial
identities. This result uses rewarded/unrewarded feedback from released column
11 and supersedes every earlier Costa author-model fit.

The model equations, block resets, and initial value are paper-aligned. One
matched-half parameter set per subject replaces the paper's separate
phase-, schedule-, and session-specific fits; repeated-stimulus value carryover
cannot be reconstructed from the retained binary cue identity.
