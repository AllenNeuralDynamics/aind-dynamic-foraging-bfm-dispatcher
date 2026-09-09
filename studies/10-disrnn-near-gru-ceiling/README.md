# 10 — disRNN near-GRU ceiling

## Question

Study 06 (merged, [PR #69](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/pull/69)) tuned a disRNN operating point (`mult=1`, `beta=3e-4`,
`latent_size=5`, `D=614`) that closes about half the GRU gap in held-out likelihood
(`-0.0114 -> -0.0047` vs. `GRU_CEIL` ~0.7267-0.7268 at D=614) but does not close it
fully. **How much of that residual gap is disRNN's architecture-capacity /
regularization design, versus something else?**

A colleague (Kevin) suggested a near-GRU sanity check: remove disRNN's penalties, make
the update-net MLPs linear, and widen the latent dimensionality substantially (to
match the GRU's capacity) — a disRNN pushed as close to "GRU-shaped" as its skeleton
allows. If the gap closes there, the residual in study 06 is a capacity/regularization
cost, not a fundamentally different function class. If it does not close even here,
something else in the architecture (the multiplicative gated update rule itself, e.g.)
is the remaining candidate.

**This is a STAGED, 3-step ablation**, not all three relaxations at once — doing them
together would not tell you which relaxation mattered. Each stage adds exactly one
relaxation on top of the previous stage's config, so any likelihood movement at a given
stage is attributable to that one change:

1. **stage1-no-penalty** — zero the update-net-latent ("interaction bottleneck")
   penalty only. `latent_size=5`, `update_net_n_layers=5` (nonlinear) unchanged.
2. **stage2-linear-update-net** — stage 1's config + linear update net
   (`update_net_n_layers=0`).
3. **stage3-wide-latent** — stage 2's config + `latent_size` 5 -> 256, matching study
   01's H=256 GRU capacity exactly (the ceiling probe).

## ⚠️ Caveat — this is NOT an interpretable disRNN

Zeroing the interaction-bottleneck penalty, removing the update-net's nonlinearity,
and widening the latent to 256 each individually and especially cumulatively destroy
the disentangling mechanism the disRNN architecture exists for. **Stage 2 and stage 3's
configurations are not interpretable disentangled RNNs** — they are disRNN-shaped
functions pushed toward GRU capacity for exactly one purpose: attributing the
residual gap. **Their results must NOT be folded into study 06's operating-point
verdict.** That is the reason this is its own study (10), not a study-06 wave: study
06's verdict is about the best *interpretable* disRNN; this study is about the ceiling
of the *disRNN-shaped* function class, which is a different question with a different
answer that happens to share code.

## Which penalty knob, and why only that one

`config_disrnn.yaml`'s `model.penalties` block has nine knobs, all defaulting to
`beta`: `latent_penalty`, `choice_net_latent_penalty`, `update_net_obs_penalty`,
`update_net_latent_penalty` (+ its `_multiplier`), `subject_penalty`,
`update_net_subject_penalty`, `choice_net_subject_penalty`. Only
`update_net_latent_penalty` is the **interaction bottleneck** — it caps how much one
latent's update can depend on the *other* latents' state, which is the actual
disentangling mechanism (separate from the multiplier that scales it). The others gate
unrelated bottlenecks (subject-embedding, observation-to-update, latent-to-choice) that
are not part of this ablation's target, so they stay at the tuned `beta=3e-4`
throughout all three stages. The config already ships a multiplier for exactly this
purpose (`update_net_latent_penalty_multiplier`, "scales its base penalty ... and is
then removed" per the config header), so `model.penalties.update_net_latent_penalty_multiplier=0.0`
zeroes ONLY the interaction penalty.

## Known wrapper limitation discovered at launch (affects the step-budget curve)

All three variants set `model.training.checkpoint_run_heldout_eval=true` per this
study's intent (a step-budget curve should be a first-class deliverable, not an
afterthought as in study 06 wave 2). **In practice this flag is a no-op for
multisubject disRNN in the current wrapper**: the trainer logs *"Skipping
PER-CHECKPOINT held-out eval (checkpoint_run_heldout_eval) for multisubject disRNN;
v1 supports seen-subject personalization only"* and proceeds regardless of the flag
value — confirmed in the live logs of all three stages at launch. The per-checkpoint
step curve available is `checkpoint/eval_likelihood` (seen-subject, every
`checkpoint_every_n_steps=10000` steps); the held-out generalization number
(`heldout/final/eval_likelihood`) is only available once, from the end-of-training
`auto_heldout_finetune` (on by default, unaffected by this flag). This is a wrapper v1
limitation (documented in the `wrapper-runtime` skill's "two DIFFERENT held-out
switches" note), not a study-10 config error — noted here so the analysis doesn't
mistake a missing per-checkpoint `heldout/*` curve for a bug.

## Variants index

| variant | what differs | D | seeds | n_steps | status | W&B group | Beaker experiment |
|---|---|---|---|---|---|---|---|
| [stage1-no-penalty](variants/stage1-no-penalty/) | interaction penalty -> 0 (mult=0); latent_size=5, update_net_n_layers=5 unchanged | 614 | 0,1 | 100000 | launched | `stage1-no-penalty@20260909-021133` | [01M22PZR6HW7C4ACC09DH6019R](https://beaker.org/ex/01M22PZR6HW7C4ACC09DH6019R) |
| [stage2-linear-update-net](variants/stage2-linear-update-net/) | stage1 + linear update net (`update_net_n_layers=0`) | 614 | 0,1 | 100000 | launched | `stage2-linear-update-net@20260909-021223` | [01M22Q19MRZ3Z2YPRQEG984323](https://beaker.org/ex/01M22Q19MRZ3Z2YPRQEG984323) |
| [stage3-wide-latent](variants/stage3-wide-latent/) | stage2 + `latent_size` 5 -> **32** (DOWNGRADED; see below) | 614 | 0,1 | 100000 | relaunched (downgraded) | `stage3-wide-latent@20260909-140640` | [01M23ZX664WP1QJQJNVBYY1V13](https://beaker.org/ex/01M23ZX664WP1QJQJNVBYY1V13) |

All three: `ai1/octo-hub-onprem-h200` only (12/16 schedulable, 0 cordoned, 0 queued at
launch), image `han-hou/dynamic-foraging-bfm-wrapper-main-20260902`, W&B project
`disrnn_near_gru_ceiling` (entity `AIND-disRNN`). Runtime refs pinned to full SHAs:
`WRAPPER_REF=9595dd371ab87de49c281d8ca4bb6ae8af7c32e4`,
`DISPATCHER_REF=4b926dc53ee7b4be8c0b55a8f195b7fbde5fd5f3`,
`FORAGING_MODELS_REF=faa0f5ad063e375765aa9c31c7d3fee5eca78ecf`. All six tasks
(2 seeds x 3 stages) confirmed scheduled with exactly 1 GPU each on
`ai1/octo-hub-onprem-h200` at launch; stage 3's first forward pass (256 unrolled
per-latent update-net MLPs, see stage3's notes.md for the compile-time risk this was
watched for) compiled and completed its pre-warmup eval in ~2m15s without OOM.
Timing was directly checked against stage1 only (pre-warmup eval logged within
seconds of model init there); stage2's timing was not checked in this session, so
the compile-time-risk comparison covers stage1 vs. stage3, not stage2.

### ⚠️ stage3-wide-latent DOWNGRADED from latent_size=256 to latent_size=32 (2026-09-09)

The original stage3-wide-latent launch (`latent_size=256`, matching study 01's H=256
GRU capacity exactly — the study's intended ceiling probe) **failed on both tasks**
after ~53 min with `RESOURCE_EXHAUSTED: Out of memory` (~1.38 TiB requested vs 141GB
available on the H200). Root cause (confirmed by a batch-size diagnostic, not just
inferred): `HkDisentangledRNN.update_latents` builds one separate `ResMLP` per latent
in a Python `for` loop unrolled at JAX trace time rather than `vmap`ped — at
`latent_size=256` this is a ~51x increase in unrolled modules vs. every prior study's
`latent_size=5`, and it blows both the GPU memory budget and a fixed PJRT
argument-packing ceiling (1024 packed device-memory arguments per compiled call) that
is **independent of batch size** — a `batch_size=64` retry did not reproduce the OOM
but hit the argument-packing ceiling instead, ruling out a config-only fix.

Per explicit steering, this was not chased further with batch-size tricks. Instead,
**stage3-wide-latent now targets `latent_size=32`** — a value confirmed by a sizing
probe to clear the same failure point, and comfortably under the packing ceiling and
memory budget. **This makes stage3-wide-latent a REDUCED ceiling probe, not the
study's original GRU-H256-matched capacity target.** Any conclusion drawn from its
results about "how much of the residual gap closes at GRU-matched capacity" must be
qualified accordingly: it answers "at latent_size=32" only.

True GRU-parity capacity (`latent_size=256`) requires a wrapper code fix — `vmap`ing
the per-latent update-net loop — tracked in
[aind-dynamic-foraging-bfm-wrapper#99](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/99).
Full diagnostic evidence (OOM log, argument-packing error, sizing-probe results,
Beaker experiment ids) is in `variants/stage3-wide-latent/notes.md` and
`variants/stage3-wide-latent/launch_record/beaker_resubmit_latent32.json`.

## Provenance

Reconcile with:

```bash
python studies/util/validate_provenance.py studies/10-disrnn-near-gru-ceiling --beaker --wandb --strict
```
