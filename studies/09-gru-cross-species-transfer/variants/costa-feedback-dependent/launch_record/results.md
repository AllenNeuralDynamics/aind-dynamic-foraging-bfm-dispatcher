# Costa (macaque) feedback-dependent RL result

- **W&B group:** `costa-feedback-dependent-bias@slurm-26386087`
- **W&B run:** [1qyvsk6u](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/1qyvsk6u)
- **SLURM job:** `26386087`
- **Completed:** 2026-09-15 PT
- **Status:** success
- **Held-out trials:** 162,960
- **Normalized likelihood:** 0.5851518426
- **Accuracy:** 0.7187960236
- **Brier score:** 0.1813699142
- **Ordered trial-key digest:** `cfb2fda156b6b3ff4e8e6b555c8b8267ac971964f0957795c9f3db968d09d4a7`

The prediction table has the same ordered trial-key digest and 162,960 held-out
rows as the corrected Costa (macaque) Bari2019 and GRU evaluations. This result
uses rewarded/unrewarded feedback from released column 11 and supersedes the
unbiased matched-half diagnostic for the main report comparison.

The three-parameter learning core follows the paper equations. A fitted
shape-choice log-odds bias is a declared benchmark augmentation; the paper did
not include it. One matched-half parameter set per subject replaces the paper's
separate phase-, schedule-, and session-specific fits. The model uses the same
real-session reset convention as Bari2019 and GRU rather than the paper's more
flexible phase-, schedule-, and session-specific protocol.

## Mechanism checks

- A fresh Bari2019 common-Q rerun reproduced the previous pooled held-out
  normalized likelihood exactly: **0.6310437182** (`26386210_7`, run
  `im1e7wb5`). This rules out a lucky or stale common-Q fit.
- Adding the Bari-style CK1 term to the bias-augmented Costa model improved
  normalized likelihood from **0.5851518426** to **0.6122294730**
  (`26386211`, run `7xjez5i7`). It remains below both Bari2019 common Q
  (**0.6310437182**) and the E4 GRU D=614 mean (approximately **0.6286**).

The CK1 result is a declared sensitivity analysis, not the author-selected
Costa reference. The main author comparison remains the dual-rate model with a
fitted bias; the paper equations omit that bias, but the fair matched-half
comparison requires allowing a stable side preference.
