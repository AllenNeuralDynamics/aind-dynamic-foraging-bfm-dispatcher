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

## Launch (original, latent_size=256 — FAILED)

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

**Result: BOTH tasks FAILED after ~53 min**, at the `train_step` JIT-compile point
(past the pre-warmup eval checked above), with:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1513245575640 bytes
```

(~1.38 TiB requested vs 141GB available on the H200). `hlo_rematerialization.cc:3022`
logged that its own rematerialization pass could only reduce the graph from 1.38TiB to
1.38TiB (`Can't reduce memory use below 99.55GiB by rematerialization; only reduced to
1.38TiB, down from 1.38TiB originally`) — the graph itself, not just per-step
activations, is enormous. So the compile-time-only risk flagged before launch (see
above) was an underestimate: it materialized as a **memory** problem too, not just a
longer JIT compile.

## OOM diagnosis (2026-09-09, same session as the failure)

Two diagnostic Beaker one-offs (not part of the resumable grid) were run to isolate
whether this scales with `batch_size` (fixable in config) or with `latent_size`'s
256x per-latent-module count (a wrapper code issue):

1. **`latent_size=256`, `batch_size` 1024 -> 64, `n_steps=50`** (exp
   [01M23R8FKZSW7D1CDQZFSNDGW4](https://beaker.org/ex/01M23R8FKZSW7D1CDQZFSNDGW4)):
   did NOT reproduce the `RESOURCE_EXHAUSTED` OOM, but hit a *different*,
   batch-size-independent failure at the identical `train_step` JIT-compile point:

   ```
   INVALID_ARGUMENT: Can't pack device memory arguments array of size 1793 which is
   larger than the maximum supported size of 1024
   ```

   This is a fixed PJRT argument-packing ceiling (1024 packed device-memory arguments
   per compiled call), driven by ~7 packed buffers/latent x 256 latents = 1793 — a
   function of `latent_size`, not `batch_size`. This rules out a config-only
   (batch-size) fix for `latent_size=256`: `HkDisentangledRNN.update_latents`
   (`disentangled_rnns/library/disrnn.py`) builds one separate `ResMLP` per latent in
   a Python-level `for net_i in range(latent_size)` loop, unrolled at trace time
   rather than `vmap`ped, and that 256x unrolled-module-count term is what has to
   change. Filed as
   [wrapper#99](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/99)
   (vmap the per-latent loop). A `batch_size=16` confirmation retry was skipped: the
   error text is unambiguous that it is a fixed argument *count*, not a size-scaled
   one, so no batch cut can fix it — per steering, not worth spending a sizing-probe
   budget on a foregone conclusion.
2. **`latent_size=32`, `batch_size=1024` (unchanged), `n_steps=50`** (exp
   [01M23VHJSCFKF6Y80WDZGSJCZ0](https://beaker.org/ex/01M23VHJSCFKF6Y80WDZGSJCZ0)):
   ~7 packed args/latent x 32 = ~224, comfortably under the 1024-arg ceiling and far
   below the 1.38TiB OOM budget. Reached and passed the exact point where the
   `latent_size=256` runs failed — logged `Running warmup training phase` and began
   executing the first training step without an OOM or argument-packing error.
   (Note: this probe ran much longer wall-clock than a "quick" n_steps=50 probe
   should, because `model.training.n_warmup_steps=7500` was left at its full-launch
   value — total steps executed = `n_warmup_steps + n_steps` = 7550, not 50. Future
   sizing probes on this variant should also override `n_warmup_steps` down to keep
   the probe fast; this one was still conclusive because it cleared the failure point
   long before 7550 steps would complete, and Beaker confirmed the job as still alive
   — not crashed — via `exited`/`exit_code`/`finalized` all `None` throughout.)

(A first smoke-test submission,
[01M23R4PMW6C88W4JQM7A1BSEG](https://beaker.org/ex/01M23R4PMW6C88W4JQM7A1BSEG), failed
immediately on a config-key error — `batch_size` lives under `data.batch_size`, not
`model.training.batch_size` — and is not counted as a real sizing probe.)

**Conclusion: latent_size=256 is infeasible on current hardware/wrapper code without
the wrapper vmap fix.** Per explicit steering from Han, stopped chasing 256 via
batch-size tricks and downgraded this variant's target to `latent_size=32` — a
**DOWNGRADED, REDUCED ceiling probe**, not the study's original GRU-H256-matched
target. True GRU-parity capacity (latent_size=256) remains blocked on
[wrapper#99](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/99).

Full evidence trail (trigger symptom/cause/evidence, job refs, deviations):
`launch_record/beaker_resubmit_latent32.json`.

## Relaunch (latent_size=32 — DOWNGRADED reduced ceiling probe)

- `sweep.yaml` / `experiment.yaml` updated: `model.architecture.latent_size` 256 -> 32.
  Everything else unchanged (`batch_size=1024`, `n_steps=100000`,
  `checkpoint_run_heldout_eval=true`, 180GiB/24CPU/1×H200, cluster
  `ai1/octo-hub-onprem-h200`, 2 seeds).
- W&B group: `stage3-wide-latent@20260909-140640` (same project/entity).
- Beaker experiment: [01M23ZX664WP1QJQJNVBYY1V13](https://beaker.org/ex/01M23ZX664WP1QJQJNVBYY1V13)
  (2 tasks, seed 0/1). Both tasks confirmed scheduled with exactly 1 GPU each
  immediately after submission.
- First submission attempt hit a transient `[code=409] a retryable database conflict
  occurred` (resolved-JSON payload was only 5074 bytes, well under the ~40-54KB
  ceiling documented in the beaker-launch skill, so this was a genuine transient
  conflict, not a payload-size issue); retried once and succeeded with no duplicate
  experiment created (verified via `b.workspace.experiments()`).

## Result

*(fill in once training finishes — `heldout/final/eval_likelihood` per seed from the
end-of-training `auto_heldout_finetune`, and the `checkpoint/eval_likelihood`
step-budget curve. NOTE: because this is now `latent_size=32`, not the original
GRU-H256-matched `latent_size=256`, its result answers a narrower question than the
study originally asked — see the study README's downgrade note before interpreting
it as a ceiling on disRNN's GRU-shaped capacity.)*
