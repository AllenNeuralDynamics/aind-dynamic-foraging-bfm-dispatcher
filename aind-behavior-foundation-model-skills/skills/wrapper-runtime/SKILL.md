---
name: wrapper-runtime
description: Understand and work with the aind-dynamic-foraging-bfm-wrapper (formerly aind-disrnn-wrapper) training/analysis runtime — the four run phases, the two DIFFERENT held-out switches, checkpoints/resume/extend, which W&B metric is the figure of merit, bottleneck-sigma conventions, trainer/data-loader architecture, post-training analysis CLI, and testing. Use when interpreting a run's logs or W&B metrics, debugging or extending training code, choosing config keys, or running post-training analysis on a saved run.
---

# Wrapper runtime (training + post-training analysis)

Canonical detail: the wrapper's **living documents** — `code/TRAINING.md` (read
**§1.5 "Run lifecycle & key switches" first**) and `code/POST_TRAINING_ANALYSIS.md`.
They are actively maintained with changelogs; **on conflict, they win** — and if you
change a feature there, you must update them (their own contract).

## Run lifecycle — the four phases (in order)

1. **Warmup** — `n_warmup_steps` of penalty ramp-up; high loss here is normal.
   **W&B `_step` includes the warmup offset**: main-phase progress is
   `(_step − n_warmup_steps) / n_steps`, not `_step / n_steps`.
2. **Main training** — checkpoints every `checkpoint_every_n_steps` (chunked path only).
3. **Artifact upload** — the whole `output_dir` uploads to W&B as
   `<mtype>-output-<run_id>` **once, at the end**. An unfinished run has no
   restorable artifact.
4. **Held-out fine-tune** (multisubject, on by default) — fine-tunes a fresh subject
   embedding per reserved held-out mouse, logs `heldout/*` into the same run.

## The two DIFFERENT held-out switches (recurring trap)

| Switch | Controls | Skipping it means |
|---|---|---|
| `model.training.checkpoint_run_heldout_eval` | per-checkpoint held-out eval *during* training | only the mid-training curve is off |
| `model.training.auto_heldout_finetune.enabled` (default **true**) | the **end-of-training** held-out fine-tune+test | **NO `heldout/*` metrics at all** |

The log line *"Skipping held-out evaluation … seen-subject personalization only"*
refers to the FIRST switch only — the end-of-training fine-tune still runs. Two
sessions have misread this; don't be the third.

**Figure of merit: `heldout/final/eval_likelihood`.** Within-subject
`checkpoint/eval_likelihood` saturates (~0.72–0.75 across model sizes) and cannot
discriminate generalization.

## Checkpoints / resume / extend (three distinct things)

- **Checkpointing** needs `checkpoint_every_n_steps > 0`; writes params +
  `train_state.pkl` under `output_dir/checkpoints/step_*`.
- **Resume (within one experiment)** — preemption recovery; automatic
  (`auto_resume`, default true). Skips warmup (already folded into the checkpoint).
- **Extend (across experiments)** — `model.training.restore_from_run_id=<W&B run>`
  (env `BFM_RESTORE_FROM_RUN_ID` wins) + larger `n_steps`; source run must have
  **finished**. Launch-side detail: beaker-launch / hpc-launch skills.

## Conventions that surprise people

- **disRNN has NO early stopping** — a disRNN sweep must omit `early_stopping` keys
  (Hydra struct mode errors on absent keys). Only `gru_trainer` has it.
- **Bottleneck σ:** small σ = OPEN, σ→1 = CLOSED. Prefer the **threshold-free**
  sparsity readouts (`bottlenecks/<fam>_n_eff_open_frac`, `sigma_median/p10/p90`,
  `total_openness`) — single-threshold `frac_open` saturates misleadingly.
- **`length_bucketing`** (requires `batch_mode: random`): ~1.86× disRNN throughput;
  keep it fixed across a grid so cells stay comparable.
- Configs: `configs/config_{gru,disrnn,baseline_rl}.yaml` document **every key
  inline** — treat them as the per-key reference. New keys are read with
  `getattr(cfg, "key", default)` and documented inline.

## Architecture (where to change what)

- `run_capsule.py` orchestrates: load config → `DatasetLoader.load()` →
  `ModelTrainer.fit(bundle)` → held-out eval/fine-tune → W&B.
- Trainers: `GruTrainer` / `DisrnnTrainer` subclass **`BaseMultisubjectTrainer`** —
  put shared behavior on the base, model-specific bits in the hooks.
  `BaselineRLTrainer` is independent (per-subject differential-evolution fits;
  held-out subjects are *re-fit*, not fine-tuned).
- Data loaders: `mice_snapshot` (standard DB path), `mice` (docDB), synthetic.
  ~20% held-out subjects reserved by rank; `eval_every_n` splits sessions per
  subject; `ignore_policy` `exclude`→2 classes / `include`→3.
- Run outputs land in `/results/` (`inputs.yaml`, `outputs/params.json`,
  `output_summary.json`, `checkpoints/`) — this tree is the contract that
  post-training analysis loads.

## Post-training analysis (no retraining)

Separately, `python -m external_transfer.gru_transfer` (wrapper #104/#105) transfers a pinned
trained GRU to an external bandit dataset by fine-tuning only a new-subject embedding, with an
optional adaptation budget `--adapt-sessions-per-subject K` (K=0 is zero-shot). It is study 09's
Beaker runner; `code/POST_TRAINING_ANALYSIS.md` documents it.

`run_analysis.py` is the unified CLI — **12 sub-commands**, all taking a saved run dir via
`resolve_model_run(model_dir, split=…, checkpoint_policy=best_eval|best_heldout|final)`:

| Family | Sub-commands |
|---|---|
| Behavior / likelihood | `generative`, `from-histories`, `likelihood-comparison`, `likelihood-advantage` |
| RNN state space | `state-space-condition`, `state-space-subject`, `state-space-overview` |
| Baseline-RL Q space | `q-space-condition`, `q-space-subject` |
| Subject embedding | `embedding` |
| Model-specific | `baseline-rl`, `finetune` (training-adjacent — uses trainers) |

`--help` on the sub-command is authoritative for flags; `POST_TRAINING_ANALYSIS.md` is
authoritative for what each one measures.

```bash
cd ../aind-dynamic-foraging-bfm-wrapper/code
python run_analysis.py generative --model-dir <RUN_DIR> \
    --split train --checkpoint-policy best_eval --output-dir <OUT>
```

Always load runs through `resolve_model_run`; never re-implement run discovery.
Analysis code must not import `model_trainers` at module load (two documented
lazy exceptions). Heavy analysis runs on a compute node (sbatch) or Beaker, never
the login node.

## Testing

`unittest`, not pytest, with `code/` on `PYTHONPATH`; never pipe through `tail`
(masks the exit code). Suites self-skip if `jax`/`haiku` aren't importable —
confirm imports before trusting green. There are **known pre-existing failures**
(some `test_disrnn_trainer` NaNs, a dead `ex_model_dir-*` fixture): judge the
*delta* against a baseline run, not absolute green.
