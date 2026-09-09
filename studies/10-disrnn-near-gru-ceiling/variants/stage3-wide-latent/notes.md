# stage3-wide-latent

## What differs from the sibling variants

Starting point: stage2-linear-update-net's config (interaction penalty zeroed +
linear update net, `latent_size=5`).

The ONE additional change here: `model.architecture.latent_size=5 -> 256`, chosen to
match study 01's H=256 GRU exactly (the GRU_CEIL reference, ~0.7267-0.7268 at D=614) —
the closest a disRNN-shaped model can get to that architecture's raw capacity.

This is the ceiling probe of the staged ablation: with the penalty, the nonlinearity,
AND the width all relaxed, how much of the residual gap remains. If a gap persists
even here, it is not explained by disRNN's capacity or regularization — something
else in the architecture (e.g. the multiplicative gated update rule itself, vs. a GRU's
gating) would be the remaining candidate.

## Risk flagged before launch

`HkDisentangledRNN.update_latents` builds one separate `ResMLP` per latent in a
Python-level `for net_i in range(latent_size)` loop, unrolled at JAX trace time (not
`vmap`ped). No prior study used `latent_size != 5`, so a ~51x increase in unrolled
per-latent modules is untested — compile time (not necessarily GPU memory, since each
per-latent MLP is tiny: `obs_size+256 -> 16 -> 2`, `n_layers=0`) could be materially
longer than stages 1/2. Sized the Beaker task at 180GiB/24CPU (vs. stages 1/2's
90GiB/12CPU) for headroom, still safely under one H200 GPU's per-slot bundle
(~378GiB/28CPU). Monitor the first scheduled task's wall-clock-to-first-step.

## Launch

- W&B group: `stage3-wide-latent@20260909-021255` (project `disrnn_near_gru_ceiling`,
  entity `AIND-disRNN`)
- Beaker experiment: [01M22Q2AF2JNNJP69RWQ8CG4AA](https://beaker.org/ex/01M22Q2AF2JNNJP69RWQ8CG4AA)
  (2 tasks, seed 0/1), cluster `ai1/octo-hub-onprem-h200`, 180GiB/24CPU/1×H200 per task.
- Both tasks confirmed scheduled (1 GPU each). Compile-time risk check (see above):
  task 000's pre-warmup eval (the first full forward pass through 256 unrolled
  per-latent ResMLPs) completed at 09:17:38, ~2m15s after model init logged at
  09:15:23. Checked directly against stage1 only (its pre-warmup eval logged
  within seconds of model init) — stage2's timing was not checked in this session.
  No OOM; the compile-time risk did not materialize at this scale vs. stage1, but
  per-step throughput (not just first-compile latency) is not yet confirmed either.
- Same `checkpoint_run_heldout_eval` caveat as stage1 (see study README).

## Result

*(fill in once training finishes — `heldout/final/eval_likelihood` per seed from the
end-of-training `auto_heldout_finetune`, and the `checkpoint/eval_likelihood`
step-budget curve.)*
