# Beaker scheduling — verified lessons & mechanisms

Deep detail behind the hard rules in SKILL.md. (Absorbed from the former
`docs/beaker-playbook.md`; this file is now the canonical home.)

## Cluster eligibility

Submit only to `hub` clusters (`octo-hub-*`, `octo.hub-*`, `aihub-*`). There are no
non-hub exceptions. A job sent to a non-hub cluster silently never schedules rather than
failing, so the symptom is a task that pends forever, not an error.

Usable for S3/DB-backed training: `octo-hub-aws-h200`, `octo-hub-onprem-h200`,
`octo-hub-aws-l40s`. GCP hub clusters cannot reach the AWS S3 parquet cache and are
unusable for any mice-data run. Canonical table: `code/beaker/README.md`.

## Priority tiers (verified on onprem-H200 + aws-L40s, 2026-06-22)

**Our workspace (`ai1/aind-dynamic-foraging-foundation-model`) has three tiers of
job protection** — pick the tier by how much the job must resist eviction:

| Tier | How to submit | Budget | Eviction |
|---|---|---|---|
| **1. Allocated / protected** | `{priority: normal or high, preemptible: false}` | **8 allocated slots** | never evicted (guaranteed) |
| **2. Unallocated / preemptible-normal** | `{priority: normal/high, preemptible: true}` | **64 unallocated slots** | evicted only by higher priority under contention; auto-resumes |
| **3. Low preemptible (burst)** | `{priority: low, preemptible: true}` | **~unlimited** (best-effort spare GPUs) | evicted by any normal job, and by other low jobs on a full cluster; auto-resumes |

Rules of thumb (updated 2026-11-13): default fan-outs → **tier 2** (`priority: normal`,
`preemptible: true`); the longest must-finish runs → **tier 1**; **tier 3** only when the
64 unallocated slots are exhausted, and only for runs that can afford to lose everything
since their last checkpoint. `maxWorkloadPriority` for this workspace is `high` (verified
2026-07-12), so `high` is available above `normal`. Budgets: **8 / 64** per Han
(2026-11-13); the 2026-06-22 measurement was 4 / 8. The beaker CLI does not expose the
budgets directly — re-derive by watching where `normal`-preemptible starts pending while
physical GPUs are free.

**Why not low (study 17, 2026-11-13).** Low-priority tasks on a full `octo-hub-onprem-h200`
(our 8 allocated + another workspace's 6 low jobs) were evicted (exit 143) every 2-4 min
and never got past setup. On `octo-hub-aws-h200`, 8 low tasks were all evicted at 01:37 PT
when another workspace started 32 normal-priority jobs; they were at 5-8k steps, before
the first 10k-step checkpoint, so resuming restarted them from step 0. When a cluster is
physically full, normal-preemptible tasks queue behind other workspaces' normal jobs (same
priority) rather than evicting them; list both H200 clusters under `constraints.cluster`.

- **Low-priority preemptible jobs burst onto spare idle GPUs *beyond* the
  workspace's unallocated-slot budget** (measured ~14 concurrent on H200), whereas
  `normal`-priority preemptible jobs are **capped at the unallocated budget** (64 since
  2026-11; 8 in June) — free physical slots sit idle while tasks pend only once that cap
  is reached.
- `autoResume` is **auto-applied** to preemptible jobs — do not set it explicitly
  (the v2 schema rejects `preemptible` + `autoResume`:
  `Error: preemptible cannot be set with min_runtime or auto_resume`). Confirm
  with `beaker experiment spec <id>`.
- **Guaranteed slots:** `{priority: normal, preemptible: false}` draws on the
  non-preemptible (allocated) budget — protected indefinitely, never evicted.
- **Budget caps ≠ physical capacity.** Tasks pending while physical slots are free
  ⇒ a budget cap (or GPU over-assignment) is binding, not capacity.

## GPU bundles & over-assignment

GPUs are bundled with host CPU/RAM. A `memory` or `cpuCount` request exceeding
**one** GPU's bundle makes Beaker assign **multiple GPUs** to a `gpuCount: 1` job
(L40S bundle ≈ 93 GiB + 12 CPU/GPU, so `memory: 256GiB` → **3 GPUs**, starving
other allocated tasks). Size to one bundle on the target cluster; big-memory only
where the workload needs it. Check `BEAKER_ASSIGNED_GPU_COUNT` /
`beaker job get` GPUS column on the first scheduled job.

## Cross-cloud S3

GCP clusters (e.g. `gcp-h100`) **cannot reach AWS S3**
(`aind-scratch-data.s3.amazonaws.com` DNS fails cross-cloud) — never route
S3/DB-backed data jobs to GCP. AWS hub clusters (l40s, h200) reach S3 fine. The
trial/session database is public AWS S3
(`s3://aind-scratch-data/aind-dynamic-foraging-cache`, us-west-2).

## Resolved-JSON payload ceiling on `experiment.create()` (hit twice: 2026-07-14, 2026-07-18)

`launch_beaker_resumable.py` renders a sweep grid into **one** Beaker experiment spec with one task
per grid point. **Beaker rejects a spec whose resolved JSON payload is too large**, with a misleading
`[code=409] a retryable database conflict occurred` — not a size/413 error, and retrying the same
oversized payload never helps.

Measured (resolved JSON, i.e. `len(json.dumps(spec))` — **not** the YAML file size; YAML aliasing
collapses repeated env blocks and understates the true payload by ~30%):

| tasks | resolved payload | result |
|---|---|---|
| 12 | 32,584 B | ✅ |
| 15 | 40,447 B | ✅ |
| 18 | 54,405 B | ❌ 409 |
| 80 | 197,546 B (2,469 B/task) | ❌ 409 (also exceeded the client's 5 s default timeout first, masking the size problem) |

**The ceiling is between 40 KB and 54 KB (likely 48 KiB).** Stay safely under it: **chunk any grid
above ~15 tasks into ≤~15-task pieces** (≈10 tasks/~25 KB gives a comfortable margin).

**Fix, two parts:**
1. `code/beaker_client.py` `get_beaker_client()` sets `beaker._timeout = 60` (beaker-py's default
   ~5 s is too short for a large-payload `experiment.create()` call) — matches the pre-existing
   `check_gpu_availability.py` workaround.
2. Render the full grid once with `--no-submit` (keeps one shared W&B group + one set of pinned
   SHAs across the whole logical launch), then split the rendered `tasks:` list into ≤15-task chunks
   and submit each chunk directly via `beaker.experiment.create()` — N Beaker experiment IDs, one W&B
   group. Verify no duplicate experiments after any submit that hit a transient 409-then-retry (check
   `b.workspace.experiments()` for stray task-count matches). Worked examples: study 05
   `subject-capacity` (18→2×9 tasks) and study 06 `mult-d-grid` (80→8×10 tasks) `notes.md`.

## A heavily-preempted run can exit 0 with its final metric MISSING (study 06, 2026-07-25)

**`exit_code == 0` does not guarantee the run's metrics reached W&B.** Verified on 5
study-06 tasks: each was preempted ~5x, W&B marked the run `crashed` on a heartbeat
timeout, and the **final summary write was silently dropped** — the run object sat
frozen at a mid-training step with **no `heldout/*` key at all**, while the job itself
ran to completion (logs end `All done, goodbye`, wandb reports syncing its files, exit 0).

Why it is silent and does not self-heal:

- **Beaker considers the task done** (exit 0), so `autoResume` never retries it.
- **W&B keeps `state == "crashed"`** from the earlier preemption, so any analysis
  filtering on `state == "finished"` drops the cell without complaint.
- The loss is *not* visible from either side alone — Beaker says success, W&B says
  crashed, and neither is wrong.

**Detection: watch the gap between Beaker-finished and W&B-finished counts.** In study
06 this showed up as a persistent, *growing* 4-5 run discrepancy (Beaker 52-55 vs W&B
48-50). Do not hand-wave it as preemption-labelling noise (it was, wrongly, on first
pass) — cross-reference explicitly: for every task whose latest attempt exited 0, pull
its `WANDB_RUN_ID` from the job's env vars and check that run's state *and* whether the
metric key is actually present. `run.summary` having 127 keys means nothing if none of
them is the one you need.

**Recovery is usually free** — the per-subject `run_table` artifact typically survives,
so the scalar can be recomputed exactly with no GPU: see mechanism (4) in
`resume-extend-rescore.md`. Only fall back to a GPU re-score (mechanism 3) if that
artifact is missing.

Expect this on any long, heavily-preempted low-priority fan-out, and check for it
**before** concluding a grid is complete.

## Verify mechanisms with data before asserting

When explaining *why* infra/scheduling/quota behaves a certain way, **pull the
actual data first** (`beaker experiment/job get --format json`, `cluster get`,
the W&B API) and cite the field. Label observed fact ("verified: …") vs inference
("likely, unconfirmed: …"); don't present a plausible hypothesis as a conclusion;
when two variables changed at once, isolate them before attributing cause. (Born
from a turn where several confident-but-wrong infra explanations had to be
retracted.)
