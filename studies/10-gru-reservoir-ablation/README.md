# Study 10: GRU reservoir ablation

Tracking: [#151](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/151)

## Question

Are the dynamics of an untrained frozen random GRU reservoir sufficient for
generalization from source-training AIND subjects to held-out AIND subjects, or
does the Study 01 result require learned recurrent dynamics?

## First-pass design

- Positive control: immutable Study 01 v2 trained-GRU curve.
- Reservoir: H=128, D=614, E=4; seeds 0, 1, and 2.
- Native seeded initialization; no reservoir-specific rescaling or tuning.
- Frozen: GRU input/recurrent weights and biases plus session-conditioning
  parameters.
- Trained on source subjects: subject embeddings and final choice readout only.
- Conditioned on each held-out subject: a fresh embedding only, 500 steps at
  learning rate 0.001.
- Same 20260603 data snapshot, held-out cohort, session split, 150,000-step cap,
  and best-source-eval checkpoint policy as Study 01 v2.

The primary paired unit is the held-out subject. For each subject, average the
three seed-paired normalized-likelihood differences (reservoir minus trained
GRU), then bootstrap subjects. The reservoir is non-inferior only if the lower
95% confidence bound is greater than -0.002.

## Execution

Seed 0 began as the end-to-end GPU smoke. At the investigator's request, seeds
1 and 2 were launched before seed 0 completed; all three outputs remain subject
to the same frozen-parameter and held-out-table audits. All training is GPU work
on Beaker hub infrastructure. Report generation is local from committed frozen
inputs.

Seed 0 was submitted at 17:55 PT on 2026-09-07 as Beaker experiment
[`01M1Z86SE16551913JFAA8D189`](https://beaker.org/ex/01M1Z86SE16551913JFAA8D189),
W&B group `frozen-random-core-d614@20260907-175533`.

Seeds 1 and 2 were submitted at 18:41 PT on 2026-09-07 as Beaker experiment
[`01M1ZATF3VZS00RZ82JQTWWQSG`](https://beaker.org/ex/01M1ZATF3VZS00RZ82JQTWWQSG),
W&B group `frozen-random-core-d614@20260907-184115`.

### Split-parity correction

The post-run parity audit found that the current snapshot resolver exchanged
three subjects between train and held-out relative to the immutable Study 01
v2 runs: the reservoir pilot trained on `764791`, `808057`, and `823164` in
place of `722683`, `795395`, and `820243`. Its held-out cohort made the reciprocal
exchange. The 146 shared held-out subjects had identical trial counts, but the
pilot is excluded from the paired confirmatory result because all 149 subjects
and the source-training cohort must match exactly.

The correction pins the ordered 614-subject source cohort and the 149-subject
held-out cohort from the exact Study 01 W&B records in
`reference/study01-v2-exact-split.json`. The rerun sweep passes both lists
explicitly, bypassing selection-policy drift.

The corrected seeds 0-2 were submitted at 04:57 PT on 2026-09-08 as Beaker
experiment
[`01M20E3AD2T3EMMAZ9VEYA8267`](https://beaker.org/ex/01M20E3AD2T3EMMAZ9VEYA8267),
W&B group `frozen-random-core-d614@20260908-045745`. Only this group enters the
confirmatory report.

## Scope boundary

This first pass does not test E=8, a reservoir D curve, tuned reservoir dynamics,
or a recurrence-disabled control. If the reservoir is competitive, the next
control is recurrence-disabled with the same readout/embedding budget.
