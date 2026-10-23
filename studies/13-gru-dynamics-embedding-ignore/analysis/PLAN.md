# Study 13 plan

Model **C3 seed 0** first (`nxd-3way-20260707-075401-61a145ca`, H128/E4/D614, 3-way head), on
the 614 training mice's eval sessions. Phases mirror Study 12's letters so every result has
a 2-way counterpart (model C) to set it against. Same rules as Study 12: reuse wrapper code,
new reusable code goes into the wrapper's `code/post_training_analysis/` with tests; split
by curriculum where it matters.

## Conventions for the third action

- **Two readouts.** The head's three logits are re-expressed as
  - `logit_diff` = logit_R − logit_L = log P(R | engaged) / P(L | engaged) — the side
    variable, directly comparable to Study 12's `logit_diff`;
  - `engage_logit` = logsumexp(logit_L, logit_R) − logit_ignore = log P(engaged) / P(ignore)
    — the new variable.

  Side analyses run on engaged trials with `logit_diff`, so their numbers sit next to
  Study 12's. Engagement analyses use `engage_logit` / `p_model1_ignore` over all trials.
- **Time.** `trial_idx` is the GRU timestep, so ignored trials are rows (`action == 2`).
  "Next trial" means the next timestep unless stated as "next engaged trial"
  (`engaged_idx`). The wrapper's history columns (`prev_action`, `history_pattern_*`, …)
  skip ignored trials; any analysis of what the GRU *receives* uses the timestep sequence.
- **Inputs.** An ignored trial reaches the GRU as (choice = 2, reward = 0) on the same
  scalar channel, so the input alphabet is 5 symbols, not 4: L+, L−, R+, R−, ignore.
- **Stay / switch** is defined between consecutive engaged trials; whether an ignore
  intervened is a separate factor.
- **RL baselines** have no ignore action. Compare to them on engaged trials only, and fit
  them to the engaged-trial sequence (as Study 01 did).

## A. Extraction
- **A1** Per-trial table over the whole eval split, ignored trials included. Done when the
  table reproduces all three logged eval likelihoods (3-way, L/R | engaged, engage).
- **A2** (before D) `gru_step` with the ignore input `(2, 0)` and both readouts; parity
  test against the full 3-way model.

## B. Descriptive → r1
- **B1** Example sessions: choices with ignores marked, P(R | engaged), P(ignore). *adapt*
- **B2** Unit heatmap sorted by correlation with `logit_diff`, and separately with
  `engage_logit`. *adapt*
- **B3** PCA exploration with the same panels, plus a P(ignore) colouring and an
  ignored-vs-engaged split; condition grids (B3c) gain the 5th input. *adapt*
- **B4/B5** Update maps and outcome-triggered averages for `logit_diff`, plus the same for
  `engage_logit`, with "after an ignore" as a 5th condition. *adapt*
- **B6** Tracking on engaged trials. *same*
- **B7** Decoders, adding engagement targets (trials since last ignore, recent ignore rate,
  trial position). *adapt*
- **B8** Engagement state: P(ignore) over the session, around ignore bouts (entry and exit),
  and against reward history; does the GRU predict bout exit (cf. Study 02 r3's
  too-sticky ignore state)? *new*

## C. Versus animals and RL → r2
- **C1** Reward-flip impulse responses on P(R | engaged) and on P(ignore) (the latter has no
  RL counterpart); logistic kernels fitted to P(R | engaged). *adapt*
- **C2/C3** GRU vs RL on engaged trials. *same, engaged-only*
- **C4** Behaviour calibration on engaged trials, plus ignore-rate calibration. *adapt*

## D. Dynamics → r3
- **D1–D4** Flow fields, fixed points and timescales per constant input, now 5 inputs;
  where do the ignore-input fixed points sit relative to the L/R ones? *adapt*

## I. Integration axis → r6
- Same questions on engaged trials, and is engagement a separate axis from the value
  axis (cosines between the TDR value axis and an engagement axis)? The distilled model
  (I4) needs a second, engagement coordinate. *adapt + new*

## E, F. Subspaces, context / embedding → r4, r5
- E1/E2 add an engagement subspace and ablate it; F3 asks whether the embedding's
  engagement dimension is separate from its bias / learning-rate dimensions. *adapt*

## Comparison with Study 12 (model C) → r7
- Same-trial comparison of C and C3 on engaged eval trials: P(R | engaged) agreement,
  PCA / CCA alignment of the hidden states, and which Study 12 findings survive.

## — stop and review —

## G. Robustness
- **G1** C3 seeds 1–2. **G2** H256 3-way (`nxd-3way` D=614 H256), if needed.
