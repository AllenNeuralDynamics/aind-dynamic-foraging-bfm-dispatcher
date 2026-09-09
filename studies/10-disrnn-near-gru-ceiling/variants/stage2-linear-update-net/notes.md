# stage2-linear-update-net

## What differs from the sibling variants

Starting point: stage1-no-penalty's config (interaction penalty already zeroed via
`update_net_latent_penalty_multiplier=0.0`, `beta=3e-4` elsewhere, `latent_size=5`).

The ONE additional change here: `model.architecture.update_net_n_layers=0`, making the
update-net MLPs linear (verified against `disentangled_rnns/library/disrnn.py`'s
`ResMLP`: with `n_layers=0` the hidden-layer loop that applies the nonlinear activation
never executes in `__init__` or `__call__`, so the module reduces to two composed
affine projections — genuinely linear overall, mirroring the existing
`choice_net_n_layers=0` convention). `update_net_n_units_per_layer` stays 16 (only the
nonlinearity is removed, not the intermediate width). `latent_size` stays 5 — stage 3
is the one that widens it.

This variant in isolation answers: on top of the penalty relaxation, how much
additional gap closes from removing the update-net's nonlinearity.

## Expectation

Some additional closure beyond stage 1, since the update net can no longer implement
nonlinear interactions between latents even where the interaction bottleneck no
longer penalizes them informationally — this stage removes the mechanism, not just
the penalty on it.

## Launch

- W&B group: `stage2-linear-update-net@20260909-021223` (project
  `disrnn_near_gru_ceiling`, entity `AIND-disRNN`)
- Beaker experiment: [01M22Q19MRZ3Z2YPRQEG984323](https://beaker.org/ex/01M22Q19MRZ3Z2YPRQEG984323)
  (2 tasks, seed 0/1), cluster `ai1/octo-hub-onprem-h200`
- Both tasks confirmed scheduled (1 GPU each) at launch time; same
  `checkpoint_run_heldout_eval` caveat as stage1 (see study README).

## Result

*(fill in once training finishes — `heldout/final/eval_likelihood` per seed from the
end-of-training `auto_heldout_finetune`, plus the `checkpoint/eval_likelihood`
step-budget curve.)*
