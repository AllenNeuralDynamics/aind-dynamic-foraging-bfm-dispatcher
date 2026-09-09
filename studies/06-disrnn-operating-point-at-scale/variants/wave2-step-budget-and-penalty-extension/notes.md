# Wave 2 — step-budget proxy + extended penalty range

Follow-on to the merged `mult-d-grid` (PR #69), motivated by two observations from that
close-out: (1) the D=614 generalization gap does not move monotonically with the tested
penalty range (lightest tested setting has the smallest gap), suggesting the grid's edge,
not interior, may be the true optimum; (2) Kevin's suggestion to treat training-step count
as a tunable.

## Stage A — step-budget proxy analysis (zero new compute, done 2026-09-09)

**What was checked.** Whether the fixed 107.5k-step training budget used throughout the
`mult-d-grid` runs off ran past the useful point, using only already-logged W&B history
(no new runs).

**Important caveat, found while doing this.** The study's held-out-across-mice metric
(`heldout/eval_likelihood`) was **only logged once, at the very end of training** (6
near-identical steps clustered in the final ~50 steps, evidently repeat eval batches for
noise-averaging, not a step-wise curve). It was **not** logged at intermediate checkpoints
in this grid, despite the wrapper config supporting it (`checkpoint_run_heldout_eval: true`
in `config_disrnn.yaml`). So the literal question "would held-out likelihood have peaked
before step 107.5k" **cannot be answered from existing data** — only a proxy is available.

**What IS available for free**: `checkpoint/eval_likelihood` — likelihood on a held-out
*session* split within the same training subjects (`eval_every_n: 2`), logged at ~12
checkpoints across the full run for all 74 analyzable finished runs. This shows a real
early-peak-then-decline pattern: for D>=30, the peak occurs at a median of ~7% of the
training budget (~step 7,500 of 107,500), and the within-training validation likelihood
then **declines** toward the final checkpoint, with the decline growing with D:

| D | median (peak − final) | worst case (peak − final) |
|---|---|---|
| 10  | +0.0028 | 0.0096 |
| 30  | +0.0031 | 0.0057 |
| 100 | +0.0054 | 0.0079 |
| 300 | +0.0071 | 0.0090 |
| 614 | +0.0072 | 0.0110 |

(D=10's curve is dominated by early-training noise rather than late-stage decline — its
median peak fraction is ~35% of the budget, not ~7% — so its numbers are not directly
comparable to D>=30's.)

![Step-budget proxy: within-training validation likelihood peaks early and declines toward the final checkpoint, worse at larger D.](fig_step_budget_checkpoint_proxy.png)

**Reading.** This is suggestive, not conclusive, evidence that step count is a real lever
worth sweeping — the within-training proxy shows genuine overfitting-shaped behavior that
grows with D, right at the study's tuned operating point (D=614). But it measures overfitting
to the SAME mice's held-out sessions, not the study's actual generalization target
(held-out mice). It does not by itself tell us whether stopping earlier would improve
`heldout/eval_likelihood`.

**Action taken**: the extended-penalty-range launch in this same wave (Stage B, below) sets
`checkpoint_run_heldout_eval: true` with intermediate logging enabled, so this wave's new
runs will carry a genuine held-out-vs-step curve going forward — a real step-budget analysis
becomes possible once those runs finish, at no extra launch cost beyond wall-clock time for
the additional periodic held-out evals.

## Stage B — extended penalty range (launched 2026-09-09)

**Question.** `mult-d-grid`'s D=614 generalization gap does not move monotonically with
penalty strength across the tested range: the lightest tested setting (mult=1, β=3e-4) has
the smallest gap (0.0069), the heaviest (mult=10, β=1e-3) has the largest (0.0094) — the
tested range's *edge*, not its interior, may hold the true optimum. Stage B pushes one step
lighter than that tested floor on each penalty axis in turn, at the two largest cohorts
(D=300, 614), to see whether the gap keeps shrinking or turns around.

**Grid launched — 8 tasks** (D × penalty-point × seed):

| D (subject_ratio) | mult | β | seeds |
|---|---|---|---|
| 300 (0.489) | 1 | 1e-4 | 0, 1 |
| 300 (0.489) | 0.5 | 3e-4 | 0, 1 |
| 614 (1.0) | 1 | 1e-4 | 0, 1 |
| 614 (1.0) | 0.5 | 3e-4 | 0, 1 |

Two new penalty points, one axis lighter than `mult-d-grid`'s tested floor at a time
(mult=1,β=3e-4 stays fixed as the *other* axis moves — never re-tested here, since it's
already an existing grid point). The sweep.yaml's `parameters:` block lists both axes as
independent grid values (mult ∈ {1, 0.5}, β ∈ {3e-4, 1e-4}) because the resumable launcher
only supports a strict cartesian product; that renders 16 raw tasks, 8 of which are either a
duplicate of an existing `mult-d-grid` point (mult=1, β=3e-4) or an untested
both-axes-lighter combination (mult=0.5, β=1e-4) outside this wave's one-axis-at-a-time
scope. Those 8 were dropped from the rendered spec before submission — see sweep.yaml's
PRUNING NOTE for the exact mechanics, and `launch_record/beaker_resumable.json` for the
before/after task list.

**Change vs `mult-d-grid`:** `model.training.checkpoint_run_heldout_eval` is `true` here
(was `false` in `mult-d-grid`, the reason Stage A above only had a within-training proxy).
`checkpoint_every_n_steps` is unchanged at 10000, so the new held-out evals land on the same
~12-point cadence and cost only wall-clock time.

**Launch provenance:**
- Beaker experiment: [`01M22QA6NH2MCDCFM8J5YREE61`](https://beaker.org/ex/01M22QA6NH2MCDCFM8J5YREE61) (8 tasks, `ai1/aind-dynamic-foraging-foundation-model`)
- Cluster: `ai1/octo-hub-onprem-h200` only (Han: submit this wave there — 10/16 schedulable,
  0 cordoned, 0 queued, verified at launch time)
- W&B group: `wave2-step-budget-and-penalty-extension@20260909-021546`
- Image: `han-hou/dynamic-foraging-bfm-wrapper-main-20260902` (verified current at launch —
  code/beaker/README.md image table)
- Resolved refs: `WRAPPER_REF=9595dd371ab87de49c281d8ca4bb6ae8af7c32e4`,
  `DISPATCHER_REF=4b926dc53ee7b4be8c0b55a8f195b7fbde5fd5f3`,
  `FORAGING_MODELS_REF=faa0f5ad063e375765aa9c31c7d3fee5eca78ecf`
- Full record: `launch_record/beaker_resumable.json`, `launch_record/experiment_resumable_submitted.yaml`
  (the actual 8-task submitted spec), `launch_record/sweep.yaml` (copy of the sweep used).

**Status (as of launch):** 4/8 tasks `idle` (starting), 4/8 `created` (queued) within seconds
of submission. Results/report land in a future session once the grid finishes — not yet
analyzed here.
