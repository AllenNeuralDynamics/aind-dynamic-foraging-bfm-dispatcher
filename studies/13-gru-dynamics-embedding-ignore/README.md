# Study 13 — GRU dynamics and subject embedding with an ignore action

**Question.** What computation does the best 3-way (left / right / ignore) GRU implement —
how its hidden-state dynamics represent engagement alongside the choice, how the subject
embedding reshapes them across mice — and which of Study 12's findings on the 2-way GRU
survive when the network must also predict disengagement?

This is an **analysis-only study** and the 3-way counterpart of
[Study 12](../12-gru-dynamics-embedding/). It trains nothing; its inputs are trained GRUs
from [Study 02](../02-gru-scaling-law-ignore/), pinned in [`source_runs.json`](source_runs.json).
There are no `variants/` and no W&B project of its own.

## Source models

Study 02's `nxd-3way` D=614 / H128 cell: choice/reward inputs only, 4-d subject embedding,
scalar session conditioning, 3 seeds, `best_eval` checkpoint (step 90k).

| key | role | W&B group | hidden | held-out L/R \| engaged (3 seeds) | ignore PR-AUC | source wrapper commit |
|---|---|---|---|---|---|---|
| `C3` | **primary** | [`nxd-3way@20260707-075401`](https://wandb.ai/AIND-disRNN/mice_ignore_scaling) | 128 | 0.7301–0.7304 | 0.635–0.637 | `4530d04` |

**Why C3 is a twin of Study 12's C.** Field by field, the run config differs from C-s0 only
in `data.ignore_policy` (`include`) and the wrapper commit (`4530d04`, 2026-07-05, vs C's
`9595dd3`, 2026-09-05). Snapshot, splits (`eval_every_n=2`, `heldout_every_n=5`), seeds,
architecture, schedule, lr and batch match, and both stop at step 90k. `4530d04` is an
ancestor of `9595dd3` and already contains the length-bucketed batching and snapshot pin
that separate C from Study 01's older H128 cells. Between the two commits the GRU trainer
changed only a log line; the data-loader changes add features these runs do not use, plus
`dac2f72`, which passes the minibatch RNG through merged datasets — it alters minibatch
order (seed-level noise), not which sessions or mice are in each split. *Inferred from the
diffs, not tested by training C on `4530d04`.*

**Metric caveat (from Study 02).** The 3-way likelihood has a 1/3 chance level and a
different trial support, so it is not comparable to 2-way numbers. Compare with Study 12
on L/R likelihood over engaged trials; report engagement separately.

## Pipeline

As in Study 12:

```
A. extract (needs W&B weights + JAX; CPU works: JAX_PLATFORMS=cpu)
   source_runs.json → fetch_models.py → analysis/_cache/models/<key>/run
                    → extract_trial_states.py → analysis/_cache/trial_states_<key>.parquet (gitignored)
                                              → analysis/trial_states_<key>.json (committed)
B. report (local, offline: committed files only)
```

The extractor is the wrapper's `post_training_analysis.gru_trial_states`, generalised to the
3-way head on the wrapper branch `study/13-gru-dynamics-embedding-ignore`: ignored trials
stay in the table as `action == 2` rows at their GRU timestep. Conventions for the third
action (two readouts, timesteps vs engaged trials, the 5-symbol input alphabet) are in
[`analysis/PLAN.md`](analysis/PLAN.md).

## Status

| step | state |
|---|---|
| C3 source runs frozen | done 2026-10-23 (`make freeze-sources`); L/R likelihoods match Study 02's `scaling.json` |
| A1 trial table, C3 seed 0 | done 2026-10-23 on CPU — 4.76M eval trials (8.9% ignored), 8,906 sessions, 614 mice; reproduces all three logged likelihoods to <5e-7 ([`analysis/trial_states_C3-s0.json`](analysis/trial_states_C3-s0.json)). Peak 31 GB RAM: run extractions one at a time |
| Phase B (r1) | next |

**Open check before same-trial comparisons with C.** C3's eval split has 8,906 sessions and
4,338,727 engaged trials; C's has 8,903 and 4,339,402. The splits are drawn the same way,
so the gap is presumably in which sessions/trials survive filtering when ignored trials are
kept — not yet traced.
