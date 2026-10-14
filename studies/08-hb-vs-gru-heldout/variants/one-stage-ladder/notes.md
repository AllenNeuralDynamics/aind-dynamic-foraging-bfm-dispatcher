# Variant: one-stage-ladder

One-stage three-level `HB-Hattori2019` fits across study 01's D dial, scored on the
held-out cohort with matched conditioning.

| | |
|---|---|
| **W&B project** | `mice_data_scaling` — **study 01's project, not a new one** (see deviation below) |
| **Launch surface** | Beaker (`sweep_beaker.yaml` + `experiment_beaker.yaml`); `production.sbatch` still carries pre-rename paths |
| **Status** | All five rungs (D≈10/30/100/300/614) have run on the current code and finished. **No HB rung passes the convergence gates**, and the D≈99, 300 and 614 chains did not mix, so their held-out numbers are not results; see the rung table and study-08 report r4 ([PR #168](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/pull/168)) |

### Rung state, 2026-10-14

The production grid (`sweep_beaker.yaml`, warmup 2000 / 500 draws / 4 chains) and the full-cohort rung
(`sweep_beaker_d614.yaml`) have both run; each submission has a record under `launch_record/`.
Convergence is recomputed from each run's saved posterior (report r4): worst R-hat and lowest bulk ESS
over the group-level and per-mouse sites.

| rung | ratio | Beaker experiment | state | worst R-hat / lowest ESS / divergences |
|---|---|---|---|---|
| D≈10 | 0.016 | `01M1MQB0SRRBDDRKP1SX3T55DV` (ladder) | finished | 1.018 / 232 / 6 |
| D≈29 | 0.049 | same | finished | 1.045 / 103 / 0 |
| D≈99 | 0.163 | same | finished, chains not mixed | 1.611 / 7 / 0 |
| D≈300 | 0.489 | same | finished, per-mouse chains not mixed | 1.283 / 11 / 0 |
| D=614 | 1.0 | `01M1MQB7NVE8N9THHXVC4W7X8Z` (full cohort) | finished, chains not mixed | 1.892 / 6 / 0 |

Sampler probes at D≈29 (same seed): warmup 4000 (`01M1MQB4MVJ76GK8NQR07WH9HR`), the sampler-diagnostic run
(`01M1NGB9S8TAP8MH3R165T6RSC`, which reproduces the ladder's D≈29 numbers exactly), and 2000 draws at D≈10 and
D≈29 (`01M1PGEPRYGY542Y5JBQB17EED`), the nearest to passing the gates. Two first launches were cancelled about five
minutes in and are recorded as such (`beaker_ladder_try1_cancelled`, `beaker_warmup4000_try1_cancelled`).

**The 2026-08-29 D30 and D101 attempts never completed.** Both crashed on 2026-08-29 in group
`hb-one_stage@20260829-181251` (SLURM 25489921/22/23), seconds after the trainer started.
That group is the *relaunch* recorded in `launch_record/hpc_ladder_relaunch.json`, whose
record documents only the earlier `ModuleNotFoundError: wandb` failure and the resubmission
— not that the resubmission also died. W&B's retained log lines stop at the JAX
backend-init message and the SLURM `.err` files are on HPC, so the cause was not
recoverable when this was written. Do not read the earlier `hb-one_stage-D{30,101}-s0` runs
as results; they carry no likelihood, no sampler config and no commit stamps.

The three `hb-one_stage-D10-s0` runs that *did* complete on 2026-08-29 predate the
`run_hpc` refactor and the seed fix (dispatcher #108), and their sampler settings were never
recorded in W&B, so they are not comparable with anything run since.

## What differs

Only `data.subject_ratio`: 0.016 / 0.049 / 0.163 / 0.489 / 1.0, giving D ≈ 10 / 30 / 100 / 300 / 614
against the ~614 pool (the last is the whole pool). The probe launches also vary the sampler settings
(warmup, draws) at D≈29. Everything else matches study 01's `mice_snapshot_scaling` config — same
snapshot `20260603`, curricula, `min_sessions=10`, `heldout_every_n=5`,
`mature_only=false`, `ignore_policy=exclude`, `eval_every_n=2`.

## What is scored

`heldout/eval_likelihood` and `heldout/test_likelihood` carry the **matched** rung:
adapt each held-out subject on its `eval_every_n=2` train sessions, score its eval
sessions. That is the same protocol as the GRU's `auto_heldout_finetune` y-axis and the
per-mouse MLE baseline, so the three are directly comparable.

`heldout/few_shot_k0_likelihood` carries zero-shot as a second, different claim: what the
cohort prior alone is worth with no data from that subject.

Comparators already on record:

| arm | held-out likelihood |
|---|---|
| per-mouse MLE Hattori (no pooling) | 0.71267 |
| per-mouse MLE CTT (best of family) | 0.71704 |
| GRU D≈30 (seeds 0/1/2) | 0.7248 / 0.7252 / 0.7250 |
| GRU D≈100 (seeds 0/1/2) | 0.7264 / 0.7264 / 0.7260 |

## Deviations from study-conventions

- **Shared W&B project.** The skill says one project per study. This variant logs into
  study 01's `mice_data_scaling` instead, deliberately: the entire point is to land on the
  same scaling curve, and `analyze_scaling.py` reads that project. A separate project would
  make the comparison a manual join.
- **`launch_id` is UTC, not Seattle.** The launcher used `TZ=UTC`; conventions and
  AGENTS.md §7 both call for Seattle. Fixed in the launcher for subsequent launches; this
  group's id is left as-is because the runs are already stamped with it.
- **Resolved, no longer a deviation — sweep configs.** This entry previously read "no
  `sweep.yaml`: these are direct `sbatch` submissions", which stopped being true when the
  Beaker surface landed. `sweep_beaker.yaml` and `sweep_beaker_smoke.yaml` are the sweep
  configs and `launch_beaker_resumable.py` consumes them. Kept as a resolved entry rather
  than deleted, so a reader of the 2026-08-29 records can still see why they have no sweep
  file.

## Known limits

Held-out scoring originally ran one adaptation fit per subject, sequentially: roughly 4 h
per rung over the 153-subject cohort, which forced production down to `matched` + `k=0`.

Adaptation and session scoring are both batched now, so a rung costs minutes and
`production.sbatch` runs the full `k ∈ {0,1,2,4,8}` sweep from the model config rather than
overriding it. The `--few-shot-k 0` in the superseded launch record reflects the old cost.
