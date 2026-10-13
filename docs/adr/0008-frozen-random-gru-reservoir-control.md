# ADR-0008: Frozen-random GRU reservoir control

- Status: accepted
- Date: 2026-09-07
- Tracking: [#151](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/151)

## Context

Study 01 found that a trained multisubject GRU generalizes from source-training
AIND subjects to held-out AIND subjects. A large recurrent network may also be a
useful random feature reservoir, so that result alone does not establish that
the learned recurrent dynamics are necessary.

## Decision

Add one opt-in `model.training.freeze_gru_core` flag, default `false`. When true,
the ordinary seeded GRU initialization is retained, all parameters except source
subject embeddings and the final choice readout receive zero updates, and the
existing held-out procedure still conditions only a fresh subject embedding.

The first experiment is Study 10 at H=128, D=614, E=4 and seeds 0, 1, and 2,
matched to Study 01 v2. It uses the same data snapshot, split, maximum training
budget, checkpoint selection, and held-out conditioning procedure. No
reservoir-specific scaling or spectral-radius tuning is allowed.

For each of the 149 held-out subjects, average the seed-paired difference
`reservoir - trained GRU` in normalized likelihood. Bootstrap subjects. Call the
reservoir non-inferior only when the lower 95% confidence bound exceeds -0.002.

## Consequences

- A competitive reservoir would show that trained recurrent dynamics are not
  required for this source-domain generalization result.
- A worse reservoir would support, but not uniquely prove, a role for learned
  recurrent dynamics.
- Gradient computation still covers the full tree before updates are masked, so
  this control is not expected to be substantially faster. Parameter partitioning
  is deferred unless runtime becomes a practical problem.
- E=8, a full reservoir D curve, reservoir-specific hyperparameter tuning, and a
  recurrence-disabled control are outside the first pass.
