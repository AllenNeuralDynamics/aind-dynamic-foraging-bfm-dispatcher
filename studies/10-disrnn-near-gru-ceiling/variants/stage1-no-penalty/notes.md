# stage1-no-penalty

## What differs from the sibling variants

Starting point: study 06's tuned disRNN operating point (`mult=1`, `beta=3e-4`,
`latent_size=5`, `D=614`, session-conditioning scalar, `n_steps=100000`).

The ONE change here: `model.penalties.update_net_latent_penalty_multiplier=0.0`,
zeroing the update-net-latent ("interaction bottleneck") penalty and leaving every
other penalty (`latent_penalty`, `choice_net_latent_penalty`, `update_net_obs_penalty`,
`subject_penalty`, `update_net_subject_penalty`, `choice_net_subject_penalty`) at
`beta=3e-4`. `latent_size=5` and `update_net_n_layers=5` (nonlinear) are unchanged.

Sibling stage 2 adds a linear update net on top of this; stage 3 further widens the
latent. This variant in isolation answers: how much of the residual GRU gap closes
from the interaction-penalty relaxation alone, with everything else held at the
disRNN operating point.

## Expectation

Some closure of the -0.0047 residual gap (study 06's tuned point vs. GRU_CEIL
~0.7267-0.7268 at D=614), but likely partial — architecture (linear choice net,
small latent, nonlinear update net) still constrains capacity relative to the GRU.

## Launch

- W&B group: `stage1-no-penalty@20260909-021133` (project `disrnn_near_gru_ceiling`,
  entity `AIND-disRNN`)
- Beaker experiment: [01M22PZR6HW7C4ACC09DH6019R](https://beaker.org/ex/01M22PZR6HW7C4ACC09DH6019R)
  (2 tasks, seed 0/1), cluster `ai1/octo-hub-onprem-h200`
- Both tasks confirmed scheduled (1 GPU each) and training past the pre-warmup eval at
  launch time; `checkpoint_run_heldout_eval=true` is a no-op for multisubject disRNN in
  the current wrapper (see study README) — the per-checkpoint curve available is
  `checkpoint/eval_likelihood`, not a per-checkpoint `heldout/*` curve.

## Result

*(fill in once training finishes — `heldout/final/eval_likelihood` per seed from the
end-of-training `auto_heldout_finetune`, plus the `checkpoint/eval_likelihood`
step-budget curve.)*
