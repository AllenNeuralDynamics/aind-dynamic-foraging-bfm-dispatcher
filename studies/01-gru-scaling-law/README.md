# Study: Mouse data-scaling law

*Folder `01-gru-scaling-law` — formerly `data-scaling-law`. W&B projects `mice_scaling`
+ `mice_data_scaling`; runs logged before the 2026-07-11 rename carry
`meta.study="data-scaling-law"` (uniform across the sampled runs, but filter by W&B
**project** rather than `meta.study` to be safe).*

**Question.** Does training a behavior foundation model on more mice improve
prediction of mice it has *never seen*? Within-subject per-trial likelihood
saturates (~0.73–0.75 flat across H128/H256, even H2–H4), so model size is not the
bottleneck and within-subject likelihood is the wrong figure of merit. The
foundation-model metric is **held-out-mouse generalization vs the number of
training mice (D)**.

**Design (minimal; Kaplan/Chinchilla practice — fix N & HPs, vary one axis).**
- Fixed: GRU `hidden_size=128`, `session_encoding_type=scalar` (session conditioning ON),
  `lr=1e-5`, `batch_size=2048`. Training protocol differs by variant (see Variants): v1 used
  `n_steps=300000` but early-stopped ~40k inside pretrain (SC never engaged); v2 uses λ-forward
  (full SC @50k) + `n_steps=150000` + gated early-stop @70k.
- Swept (science axis): `data.subject_ratio ∈ {0.016, 0.049, 0.163, 0.489, 1.0}`
  → D ≈ {10, 30, 100, 300, ~614} training mice (scalar ratio ⇒ natural curriculum
  composition at every D).
- Replicates: `seed ∈ {0, 1, 2}` → 3 **nested** ladders (loader uses permutation-prefix
  sampling: D=10 ⊂ D=30 ⊂ … for a fixed seed).
- Held-out cohort is **fixed** (`heldout_every_n=5`, every 5th ranked subject per
  curriculum) and constant across all D and seeds — mice never in any training set.
- y-axis: held-out-mouse likelihood from a **single final** held-out fine-tune+test
  at the end of each run (`auto_heldout_finetune`, enabled) — fine-tune only the
  subject embedding on a held-out mouse's sessions, then predict its other sessions.
  Per-checkpoint held-out passes are OFF (`checkpoint_run_heldout_eval=false`) to keep
  eval-during-training minimal. The same fine-tune+test can be re-run offline from the
  saved checkpoint if we change the protocol.

15 runs total (5 D × 3 seeds).

## Summary (verdict, 2026-06-24)

Per-result writeups live in [`analysis/reports/`](analysis/reports/INDEX.md) (r1–r9);
this is the study-level cover.

**TL;DR.** Training on more mice improves prediction of *unseen* mice, but the gain
**saturates by ~100 mice** and the absolute ceiling is low — per-trial L/R choice
likelihood is near a **predictability ceiling**. Session conditioning (v2) is
neutral-to-slightly-negative at small D, then adds a **small, highly-significant gain
that grows with D** (robust from D≈100); ~¾ of that gain persists on mature-only
sessions, so it's mostly *not* a curriculum-stage artifact. The population-mean
("average mouse") already predicts a new mouse to within ~0.3% of full adaptation, so
per-mouse few-shot adaptation is **not** where scale pays off. Effects are tiny
per-trial but, being consistent across ~149 mice, are real evidence (a +0.001
per-trial-normalized LL ≈ ~2× per-session likelihood ratio). **The population GRU beats
every per-mouse classical RL model we fit — Bari, Hattori, and compare-to-threshold — at
every (variant, D). Against the *strongest* of them (compare-to-threshold) the margin at
D=614 is +0.0113 (100% of mice, Wilcoxon p~1e-25) — ~8× the SC effect and the dominant
signal in this study** (r8).

> **Correction (2026-07-13).** This summary previously reported **+0.0136**, measured
> against Bari alone. Bari is **not** the strongest classical model: compare-to-threshold
> beats it (0.71704 vs 0.71491 pooled). Fitting the full suite (issue #20) shows the
> honest best-of-breed margin is **+0.0113** — the old figure overstated the gap by ~20%.
> The GRU still wins on every model and every cell, and the conclusion is unchanged; the
> claim is now "beats the best of three" rather than "beats Bari", which is both smaller
> and much harder to dismiss. Hattori is the *weakest* of the three: asymmetric learning
> rates hurt held-out generalization here.

**On effect sizes (Kevin Miller).** LL is per-trial-normalized (NL = exp(mean_t log p_t)).
A *consistent* Δ=+0.001 ≈ +0.0014 nats/trial → ~0.7 nats over a ~500-trial session →
**~2× per-session likelihood ratio**, compounding across sessions/mice — so the small
SC / data-scaling deltas are genuine model evidence (per-mouse pairing p~1e-24), not
noise, even though the metric is headroom-poor.

**Verdict.** On **per-trial choice likelihood** the system is **near a predictability
ceiling**: a new mouse is predicted to ~99.7% of its adapted likelihood from the
population mean; data-scaling rises fast then saturates by ~100 mice; per-mouse
adaptation adds ~+0.002 (flat in D); SC adds a small, real, mostly-not-stage gain that
grows with D. The **N×D joint scan (r7)** confirms D saturates at every N and the N-axis
adds only +0.004→+0.006 (a weak Chinchilla-style interaction). **But the GRU beats the
per-mouse RL baseline by +0.0136 at D=614 on 100% of held-out mice (p~3e-26) — ~9× the
SC effect (r8).** This doesn't *invalidate* the foundation-model idea (effects are real
and compound, and the cross-mouse vs per-mouse cognitive-model gap is substantial) but
this **metric/task lacks the headroom** to demonstrate further big-data scaling within
the GRU. Tests that could: headroom-ier targets — 3-way output incl. ignored trials,
lick/RT modeling, OOD task/rig transfer, and a fair-comparison population RL baseline
(hierarchical Bayesian fit with a D-axis; see `TODO / follow-ups`).

**Status (2026-06-24).** Done: r1–r6 + bootstrap + N×D joint scan (r7) + simple per-mouse
RL baseline (r8) + generative behavioral-match vs D (r9). In progress: mature-only
few-shot. Deferred: mature-only retrain; regularized few-shot; hierarchical-Bayesian
population RL (the D-aware fair-comparison baseline).

**Provenance.** Training variants: `mice-data-scaling-gru` (exp 01KVQ7EJ3C5YJ8FJVNJB8C8N36),
`v2-sc-active@20260622-144622` (exp 01KVRMSAAJTRSJMFV5JT7JAP6X), `nxd-grid@20260623-102649`.
Offline analyses (wrapper 4f29680 / few-shot knob bb4b052), one run/cell deduped:
`heldout-rerun-*` (adapted), `heldout-zeroshot-*` (zero-shot), `heldout-rerun-*-fewshot-k*`,
`heldout-rerun-*-mature2@` (mature), `generative-*`. RL baseline:
`rl-baseline-simple@20260624-171829` (run `cdq292n5`, HPC CPU, 149 held-out mice, 1.01M
eval trials). Per-arm launch breadcrumbs in [`analysis/provenance/`](analysis/provenance/).
Report run: https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/0fhvwwfu

## Cohort sampling, seeds & nesting (verified 2026-06-23)

- **One `seed` controls *both* data selection and training.** `subject_sample_seed`
  defaults to `seed` (`mice.py`: `data_cfg.get("subject_sample_seed", data_cfg.get("seed"))`)
  and the sweeps never set it separately — so each seed is a *different random subset of
  mice* **and** a different training init. This is intentional: seed-averaging then captures
  **sampling variability** (which D mice you happened to grab — the honest error bar at small
  D), not just training noise. To make `seed` training-only, pin `data.subject_sample_seed`
  to a constant and vary `seed`.
- **`subject_ratio` → actual D is fuzzy** (e.g. ratio `0.049` → **29 or 30** mice depending
  on seed). The per-curriculum sample *count* is deterministic (`round(ratio × pool)`), but a
  selected mouse that passes the `min_sessions ≥ 10` filter (which counts *all* sessions) yet
  **contributes no trials** after the scoped trial fetch is dropped (`mice.py:115-138`), so
  the surviving count depends on *which* mice a seed picked. We therefore use the **actual
  resolved D** (`len(resolved_subject_ids)`) as the x-axis, never the nominal ratio.
- **Verified empirically (both v1 and v2):**
  - v1 and v2 train on the **identical mouse set** in all 15 (ratio, seed) cells → the
    held-out v1-vs-v2 comparison is matched cohorts (apples-to-apples).
  - Cohorts are **nested across ratios** for each seed (seed0: `10⊂29⊂99⊂300⊂614`,
    seed1: `10⊂30⊂101⊂301⊂614`, seed2: `10⊂30⊂101⊂300⊂614`) — the permutation-prefix design
    holds, and the post-fetch drop preserves nesting (the drop is per-mouse, D-independent).

## Variants

Each run/condition of this study lives in `variants/<name>/` (its own `sweep.yaml`,
`experiment.yaml`, `notes.md`, launch record). Shared tooling stays at the study root
(`analyze_scaling.py`, `heldout_offline.yaml`, base configs in `code/config`). All
variants log to **one** W&B project (`AIND-disRNN/mice_data_scaling`) so they compare
side-by-side; each launch is its own group (see Provenance below).

| variant | what differs | status | W&B group (launch) | Beaker exp |
|---|---|---|---|---|
| [`v1-pretrain-phase`](variants/v1-pretrain-phase/notes.md) | early-stop fired ~40k (pretrain) → session conditioning never engaged | ✅ done | `mice-data-scaling-gru` | `01KVQ7EJ3C5YJ8FJVNJB8C8N36` |
| [`v2-sc-active`](variants/v2-sc-active/notes.md) | λ forward (full SC @50k) + gated early-stop @70k; `n_steps=150k` | ✅ done | `v2-sc-active@20260622-144622` | `01KVRMSAAJTRSJMFV5JT7JAP6X` |
| [`e8-d614-smoke`](variants/e8-d614-smoke/notes.md) | Non-scientific one-task plumbing check for an eight-dimensional subject embedding at D=614 | passed (exit 0; held-out metric landed) | `e8-d614-smoke@20260906-192034` | [`01M1WTNQSMRKSKCWD1Q6CXRPHT`](https://beaker.org/ex/01M1WTNQSMRKSKCWD1Q6CXRPHT) |
| [`e4-e8-d614-source`](variants/e4-e8-d614-source/notes.md) | Paired current-code E={4,8}, D=614 source cores; three seeds per dimension for held-out AIND and Study 09 transfer | ✅ done (6/6 succeeded) | `e4-e8-d614-source@20260906-195409` | [`01M1WWK9HP269XQS440Y29631F`](https://beaker.org/ex/01M1WWK9HP269XQS440Y29631F) |
| [`nxd-h512`](variants/nxd-h512/notes.md) | H=512 capacity row for the r7 N×D grid (same recipe as `nxd-grid`/`v2-sc-active`); D=614 excluded, host-RAM OOM never resolved on HPC or Beaker | ✅ done, D∈{10,30,100} only | `nxd-h512@20260720-195322` | n/a (HPC SLURM) |
| [`rl-baseline-simple`](variants/rl-baseline-simple/notes.md) | independent per-subject Bari RL; **skips the train fit** and scores reserved held-out mice under `heldout/*` | ✅ done (superseded by `rl-baseline-bari`) | `rl-baseline-simple@20260624-171829` | HPC CPU, run [`cdq292n5`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/cdq292n5) |
| [`rl-baseline-bari`](variants/rl-baseline-bari/notes.md) | Bari (`ForagerQLearning` L1F1_CK1). **Fits BOTH cohorts** (`skip_train_fit=false`): 614 train + 149 held-out | ✅ done | `rl-baseline-bari@20260713-005938` | HPC CPU, run [`bg3nzqz9`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/bg3nzqz9) |
| [`rl-baseline-hattori`](variants/rl-baseline-hattori/notes.md) | Hattori (`ForagerQLearning` L2F1, asymmetric α⁺/α⁻). Both cohorts | ✅ done | `rl-baseline-hattori@20260713-005950` | HPC CPU, run [`unhmbrk4`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/unhmbrk4) |
| [`rl-baseline-ctt`](variants/rl-baseline-ctt/notes.md) | Compare-to-threshold (`ForagerCompareThreshold`). Both cohorts. **Strongest classical model** | ✅ done | `rl-baseline-ctt@20260713-010000` | HPC CPU, run [`lmg1i9yd`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/lmg1i9yd) |

## Provenance & tracking (one launch = one "pseudo-sweep")

Every launch is uniquely + readably identifiable, with platform-native ids saved alongside.
The Beaker launchers (`launch_beaker_resumable.py`, `launch_beaker.py`) and the HPC launcher
(`launch_hpc.py`) stamp the shared study/variant/launch metadata; see `AGENTS.md` §8.

- **W&B group = `<variant>@<launch_id>`** (`launch_id` = Seattle timestamp). Distinguishes
  repeats of a variant; `launch_id` is also folded into run ids (unique per launch).
- **`meta.{study,variant,launch_id,label,config_hash}`** in run config — portable across
  CO/Beaker/HPC (set via `DISRNN_META_*` env; `--label` adds a human tag).
- **Platform-native cross-refs** next to `CO_COMPUTATION_ID`: `BEAKER_EXPERIMENT_ID`,
  `BEAKER_JOB_ID`, `wrapper_commit`, `dispatcher_commit` (+ the native W&B `sweep_id` on the
  agent route).

So post-hoc: filter `meta.study`/`meta.variant` to aggregate, `group`/`meta.launch_id` to pin
the exact launch, and the Beaker ids to jump to the run on the cluster.

## Launch (resumable, one autoResume task per grid point)

Pick a variant and point the launcher at its `sweep.yaml` + `experiment.yaml`:

```bash
cd <dispatcher repo>
export PATH="$PWD/.venv/bin:$PATH"   # launcher calls bare `wandb`/`beaker`
V=variants/v2-sc-active              # or variants/v1-pretrain-phase, variants/nxd-grid
python code/launch_beaker_resumable.py \
  --sweep studies/data-scaling-law/$V/sweep.yaml \
  --experiment studies/data-scaling-law/$V/experiment.yaml \
  --output-dir /tmp/dsl_$(basename $V)
```

Each grid point becomes a preemptible Beaker task with `autoResume` (Beaker restarts
it in place with the same `/results`; the trainer resumes from its last full-state
checkpoint). H128 fits both `octo-hub-onprem-h200` (template default; free slots,
reaches the AWS DB) and a 48 GB `octo-hub-aws-l40s`. 15 tasks ride the preemptible
(unallocated) quota and drain in waves; preemption auto-recovers, so even the long
large-D runs need no allocated slots.

`rl-baseline-simple` is the exception: it is CPU-only differential-evolution RL
that skips the training-subject fit and fits only the reserved held-out cohort,
launched via `code/launch_hpc.py` on Allen HPC; see its variant notes.

Code versions: pinned per variant in each `experiment.yaml` `WRAPPER_REF` (e.g. v2 uses
`65c3350`; offline analyses use `4f29680`/`bb4b052`), `DISPATCHER_REF=study/data-scaling-law`.

## Pre-launch check

Confirm the largest run converges (the original bug was undertraining): watch
`checkpoint/train_likelihood` of the D≈614 task — it should plateau ≳0.75 by 300k. If
still climbing, raise `lr`/`n_steps`. Also confirm the smallest D (ratio 0.016) still
spans all three curricula in the loader's `[select 6/6]` log (rounding can drop a
small curriculum; raise the smallest ratio if so).

## Analyze

```bash
.venv/bin/python studies/data-scaling-law/analyze_scaling.py            # all variants
.venv/bin/python studies/data-scaling-law/analyze_scaling.py --variant v2-sc-active
```
Writes `scaling_results.csv` (+ `scaling_curve.png` if matplotlib present): held-out
likelihood (`heldout/eval_likelihood` / `heldout/test_likelihood`, from each run's final
`auto_heldout_finetune`) vs actual #training mice (`len(resolved_subject_ids)`), mean over
seeds, power-law fit `L = E + (Dc/D)^α`. **Grouped by `meta.variant`** — every variant gets
its own curve/fit and they overlay on the plot (v1 vs v2 …); CSV columns include
`variant`/`launch_id`/`group`. `--variant <name>` restricts to one.

### Syncing the offline W&B runs (this study runs `WANDB_MODE=offline`)

The runs log offline to `/results/wandb/offline-run-*`, and `/results` is **each Beaker
task's result dataset — persisted to Beaker/S3, not a local path**. So to get the runs
into W&B (for `analyze_scaling.py`, which reads the W&B API) you must first **fetch the
result datasets from Beaker**, then `wandb sync`:
```bash
# pull all task result folders from S3 (one folder per task)
beaker experiment results 01KVQ7EJ3C5YJ8FJVNJB8C8N36 -o /tmp/dsl_results
# sync each task's offline run into W&B (login node reaches W&B fine)
for d in /tmp/dsl_results/*/wandb/offline-run-*; do wandb sync "$d"; done
```
(Per-task result-dataset id is also at `jobs[].result.beaker` in
`beaker experiment get <exp> --format json`; fetch one with `beaker dataset fetch <id>`.)
Authoritative metrics also live in each dataset's `/results/run/outputs`
(`output_summary.json`, checkpoint metrics) independent of W&B.

### Optional: re-run held-out offline

The same fine-tune+test can be re-run from a saved checkpoint without retraining (e.g. to
change the protocol) — mount the run's `/results` and:
```bash
python code/run_heldout_subject_finetuning.py --config configs/config_heldout_subject_finetuning.yaml \
   source_run.run_dir=<mounted /results/run> source_run.checkpoint_policy=best_eval
```

## Results — first run (2026-06-22)

> **→ For the comprehensive, current results (v1 vs v2, zero-shot, few-shot, SC-stage verdict,
> bootstrap CIs), see the [Summary](#summary-verdict-2026-06-24) above and
> [`analysis/reports/`](analysis/reports/INDEX.md) (r1–r9).** The section below is
> the **v1 historical record** (the no-session-conditioning regime).

Experiment `01KVQ7EJ3C5YJ8FJVNJB8C8N36` (onprem-H200, offline W&B; 15 runs all completed,
synced to W&B `AIND-disRNN/mice_data_scaling` as `mice-data-scaling-gru-*-r1`).

Held-out-mouse generalization vs # training mice D (`heldout/eval_likelihood`, mean over 3 seeds):

| D (train mice) | mean held-out LL | per-seed |
|---|---|---|
| ~10  | 0.7219 | 0.7202, 0.7216, 0.7238 |
| ~30  | 0.7250 | 0.7248, 0.7250, 0.7252 |
| ~100 | 0.7262 | 0.7260, 0.7264, 0.7264 |
| ~300 | 0.7267 | 0.7265, 0.7267, 0.7268 |
| ~614 | 0.7268 | 0.7266, 0.7268, 0.7268 |

**Finding: held-out generalization saturates.** It rises ~0.722 → 0.727 from D≈10→100, then is
**flat from D≈100 to 614** — only ~+0.005 total over a 60× increase in training mice. Diminishing
returns set in by ~100 mice; a 614-mouse model generalizes no better than a ~100-mouse one.

**⚠️ Major caveat — this is the pretrain-phase / no-session-conditioning regime.** Early stopping
(min_delta 0.003, patience 2) fired at **step ~40k for every run**, which is deep inside *pretrain*
(pretrain=90k; session-conditioning warm-up only spans 90k→150k). So session conditioning **never
engaged** — this curve is effectively "more mice, no session conditioning," exactly where saturation
is most expected. **The question of whether more mice help *with* session conditioning active is
still open.** To answer it, re-run with early stopping gated to start after warm-up (≥150k) or
disabled, so runs train through the session-conditioning schedule.

> **Resolved by v2-sc-active** (λ-forward + gated early-stop, so SC fully engages): SC active gives a
> **small, highly-significant gain that grows with D** (+0.001→+0.0015 at D≥100, p~1e-24 per mouse),
> ~¾ of which persists mature-only (mostly not a curriculum-stage artifact). Data-scaling still
> **saturates by ~100 mice** even with SC. Full numbers in [`analysis/reports/`](analysis/reports/INDEX.md).

## Early stopping (manual, consistent across D)

Constant lr=1e-5, **no LR scheduler** (training is stable; scheduler = marginal gain + extra
HP). Stop each run when its within-subject **eval_LL** plateaus, then run held-out offline on
its `best_eval` checkpoint. Criterion (applied identically to every D so the comparison is fair):
- **min-delta ε = 0.003** eval_LL to count as a new best (below this is eval noise),
- **patience = 2 checkpoints (20k steps)** with no new best → `beaker job cancel` that run,
- hard overfit guard: cancel if eval_LL drops > 0.01 below the running best.
`best_eval` is the safety net — the held-out fine-tune always uses the best checkpoint, so ε
only controls compute saved, not result quality. (Within-subject eval saturates ~0.74 across
H2–H256 — a noise/feature ceiling, not capacity; see TODO.)

## TODO / follow-ups

**→ [FUTURE_DIRECTIONS.md](FUTURE_DIRECTIONS.md)** — validating the foundation-model/scaling thesis
(per-trial LL is near a predictability ceiling, so D-alone saturating isn't conclusive):
zero/few-shot adaptation-efficiency metrics, joint N×D (IsoFLOP) scaling, OOD task/rig transfer,
and quantifying the current saturation.

- **Hidden-size scan on the HELD-OUT metric.** Within-subject eval is flat H2→H256 (capacity-
  saturated), so model size looks irrelevant there — but capacity may matter more for
  representing a *new* mouse. After the D-curve is in, scan hidden_size {16, 64, 128, 256} at a
  fixed large D and compare **held-out** generalization (not within-subject LL). Keep H128 as
  the default for the main D-sweep.
- N (model-size) × IsoFLOP scaling only if the D-curve shows real headroom.
- **DONE for GRU (PR #41, opt-in / default-off):** native **early stopping**
  (`training.early_stopping {enabled,metric,min_delta,patience,overfit_guard}` — stops at the
  eval-LL plateau and `break`s so finalization + held-out still run, `best_eval` used) and
  **length-bucketed batching** (`training.length_bucketing` + `length_bucket_grid` — draws each
  batch from one grid-rounded session-length bucket and trims the unroll). Padding context:
  sessions median 521 vs T_max 2207 (p95 846), so ~50–75% of unroll compute was padding ⇒ ~2.6x
  predicted. **Measured** (two identical D≈614/H128 runs, W&B `bench_padding`, only `length_bucketing`
  differing): steady-state **0.72 → 0.33 s/step ≈ 2.2x speedup** with an **identical loss curve at
  matched steps** (±0.0015 noise) — slightly under prediction due to grid=128 rounding + fixed
  per-step overhead. Both enablers are on for the 15-run sweep.
- **TODO — port both to disRNN** (needed once disRNN runs are wanted). Early stopping: add the
  same checkpoint-loop hook to `disrnn_trainer`. Length bucketing: the batch sampler
  `session_regularized_training._sample_batch` is **shared**, so it likely "just works" once
  `disrnn_trainer` sets `dataset_train.length_bucketing` on its train set — verify against
  disRNN's warmup/pretrain + session-reg schedule. Defer for now.
- **Selection vs trial-validity filter mismatch.** `min_sessions≥10` (Step 2) counts *all*
  sessions, but selection is followed by a post-fetch drop of subjects that "contribute no
  trials" (`mice.py:115-138`) → actual D is seed-dependent (29 vs 30) and < nominal. Current
  approach (plot actual resolved D) is scientifically fine, so this is optional. If we want a
  deterministic ratio→D map: either (a) **filter the pool on usable-trial sessions up front**
  (count sessions with ≥1 valid trial under the same `ignore_policy` before selection — most
  principled, but needs a trial-level pass over all subjects), or (b) **select → drop →
  top-up** from the next permutation-prefix subjects to hit `n_take` survivors (cheap, keeps
  nesting). First run a diagnostic: how many subjects pass `min_sessions` but yield 0 trials?
  If just the 1–2 we see, leave it; if many, the `min_sessions` filter is mis-specified → do (a).
- **Use PAIRED / repeated-measures analysis (the cohorts support it).** Two free power gains:
  (1) **v1 vs v2 (SC off/on):** every (D, seed) cell uses the *identical* mouse set, so test
  the **15 paired differences** (v2−v1) with a paired t / Wilcoxon — cancels cell-to-cell and
  which-mice variance, far more powerful than two independent groups of 15 (doable now from the
  aggregate held-out LL). (2) **Across D (scaling trend):** the held-out *test* cohort is fixed
  at every D, so each held-out mouse is measured repeatedly across D ⇒ a per-held-out-mouse
  repeated-measures / mixed model is both more powerful and *more correct* (adjacent-D training
  cohorts are nested, so run-level means are NOT independent — treating them as independent
  overstates df). This needs **per-held-out-subject (or per-session) likelihoods saved**, not
  just the run-level aggregate — small logging add for future runs.

## Status log

- 2026-06-22: study scaffolded; nested-sampling patch landed in wrapper
  (`study/data-scaling-law`, `41efc09`); resume path re-validated on Beaker.
- 2026-06-22: fixed resumable-mode `inputs.yaml` bug (`b0c7f11`) that made the
  held-out fine-tune silently skip; re-confirmed held-out eval logs (≈0.70 on a
  tiny D≈30 probe). Launched the 15-run sweep (scalar, lr=1e-5, 300k, lean) as
  experiment `01KVPPMQ38NNT00870Q1QAT0XF` on onprem-H200 (preemptible/autoResume).
  Watching large-D `train_likelihood` to validate lr=1e-5 convergence.
- 2026-06-22: cancelled `01KVPPMQ38…` to fold in PR #41 (early stopping +
  length-bucketing). Padding bench (`bench_padding` W&B project, D≈614, identical
  except `length_bucketing`) confirmed **~2.2x speedup** (steady-state 0.72 →
  0.33 s/step) with an **identical loss curve at matched steps** (±0.0015, noise).
  Enabled `length_bucketing=true` + `early_stopping.enabled=true` in the sweep
  (`1265b07`) and relaunched as experiment `01KVQ3EXZ6PNVQETACT42TGB58` (15
  autoResume tasks, onprem-H200).
- 2026-06-22: concise W&B run names — replaced the full subject-ID list in the run
  name with the subject count (`_n<count>subj`; full list stays in
  `config.resolved_subject_ids`), wrapper `ed8f50e`. Relaunched as
  `01KVQ42H3ZT0EJQGXD3KRBKV3V` — but that launch surfaced two W&B failure modes:
  (1) reusing a deterministic `WANDB_RUN_ID` from a cancelled run + a changed
  `wrapper_commit` → `ConfigError`; (2) 13 simultaneous `wandb.init` calls
  overwhelming the backend → 90s init timeout (only 1/15 runs came up).
- 2026-06-22: hardened `start_wandb_run` (wrapper `ef862fd`): per-run staggered
  init, `init_timeout=300`, retry+backoff, and `allow_val_change=True` on the
  SHA stamp (knobs `WANDB_INIT_TIMEOUT`/`WANDB_INIT_STAGGER`). Relaunched as
  `01KVQ4VSN5HXB91YH0MF7EZ5K0` — `allow_val_change` killed the ConfigError, but
  all 13 still hit `wandb.init` within a 20s window, hung the full 300s, and
  retried in sync (0/15 runs came up). W&B itself was healthy (status page green,
  direct API reads <1s) — the bottleneck is the onprem cluster's shared egress
  under a 15-way simultaneous init burst (single/few runs connect fine).
- 2026-06-22: the 180s stagger relaunch (`01KVQ5BH567MF1FEG8YS7Y5M9N`) **refuted
  the concurrency hypothesis** — inits spread over 146s, yet even near-solo inits
  (40s isolation) still timed out at 300s. Real root cause: the **onprem-H200
  cluster's network path to W&B run-creation degraded ~07:49** (W&B status green,
  my-machine API reads <1s, bench connected fine at 07:08). Not W&B, not
  concurrency, not code (auth succeeds; the run-*create* call hangs).
- 2026-06-22: reverted the speculative stagger/timeout/retry (wrapper `bdb326d`;
  kept `allow_val_change` — a real fix). **Moved the sweep to AWS L40s**
  (`octo-hub-aws-l40s`): H128 fits 48GB, reaches the AWS DB, and AWS→W&B works.
  Node = 1 machine / 4 GPU slots / ~373 GiB → `memory=90GiB`, `cpuCount=8` packs
  4 concurrent (rest queue + drain via autoResume). **Relaunched as experiment
  `01KVQ682F0XPAH4QD58SAKQ4R7`** (`WRAPPER_REF=bdb326d`). 3-h cron repointed.
  Throughput note: only ~4 run at once on the single L40s node, so 15 drain in
  waves (early stopping shortens each).
- 2026-06-22: ran offline on L40s (`01KVQ6K7…`), trained fine. Then probes settled
  the real root cause: on **onprem-H200 a single `wandb.init` succeeds (~1.5s) but
  14 concurrent ones all time out** — a W&B **run-creation throttle under a
  concurrent-init burst**, not a dead path. (The earlier "L40s also fails even
  staggered" was a red herring: L40s has a *separate* W&B-reachability problem, so
  staggering couldn't be validated there — running that test on L40s instead of
  H200 wrongly sank the concurrency theory.) Online needs init concurrency limited
  (stagger/serialize); offline sidesteps it entirely since init is local.
- 2026-06-22: final config — **onprem-H200 + `WANDB_MODE=offline` + full 8+4 quota**.
  The resumable launcher only uses the 8 preemptible slots (autoResume), so the
  rendered spec is post-edited to flip the **4 largest-D runs** (ratio 1.0 ×3 seeds
  + ratio 0.489 seed 0) to **allocated/non-preemptible** (uses the 4 allocated
  slots; no autoResume, but protected — relaunch-to-resume if one dies).
  **Experiment `01KVQ7EJ3C5YJ8FJVNJB8C8N36`** (15 tasks, ~12–14 concurrent). Offline
  runs are persisted to each task's **Beaker result dataset (S3)**, not a local path
  — `beaker experiment results <exp>` to fetch, then `wandb sync` (see "Syncing the
  offline W&B runs"). Track health via Beaker logs, not the live W&B project.
- 2026-06-22: wrapper `6ede321` — `start_wandb_run` now tries online with jittered
  retry, then **falls back to `WANDB_MODE=offline`** so training never blocks on a
  flaky online run-creation handshake (knobs `WANDB_INIT_ATTEMPTS`/`WANDB_INIT_TIMEOUT`;
  post-finish sync deliberately not added — it'd run on the cluster where the same
  path may be down). NOT adopted by the running sweep: it stays pinned to
  `WRAPPER_REF=bdb326d` + explicit `WANDB_MODE=offline` so all 15 tasks (and any
  caretaker relaunch of a dead allocated task) behave identically. **Future launches**
  can bump `WRAPPER_REF=6ede321` and drop `WANDB_MODE=offline` from the template to get
  adaptive online-with-offline-fallback automatically.
- 2026-06-22 (SETTLED): all 15 runs completed & synced; results above. **wandb root
  cause definitively resolved** — after the cluster freed up, **14 truly-simultaneous
  online `wandb.init`s on onprem-H200 all succeeded** (14/14, started within 1s of each
  other, ~1.3s each). So it was **NOT concurrency** (the earlier 8-vs-14 framing was
  wrong) — it was a **transient degradation of the cluster→W&B run-creation path during
  ~07:49–08:30 UTC** that happened to coincide with our 14-task launches and has since
  cleared. Offline mode was the right mitigation regardless; the `6ede321`
  retry→offline-fallback safeguard handles any recurrence. Concurrency is not a real
  constraint for these launches.
- 2026-06-23: **v2-sc-active complete (15/15).** v1-vs-v2 matched-pair held-out analysis done.
  Cell-level (n=15): mean Δ(v2−v1)=+0.00074, paired t p=0.0015, Wilcoxon p=0.0043. Per-held-out-
  mouse (offline re-runs, n=149/D): SC gain grows with D — ~0 at D≤30 (slightly hurts at D=30),
  +0.0010 (D=100) → +0.0015 (D=614), Wilcoxon p~1e-20–1e-24. Report run:
  https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/0fhvwwfu ; see the Summary above.
- 2026-06-24: **rl-baseline-simple complete (1/1).** Bari L1F1_CK1, one DE fit per held-out mouse on
  its train sessions, scored on its eval sessions. n=149, 1.01M eval trials. Trial-weighted pooled
  LL **0.7143**; per-subject mean **0.7211** ± 0.0052 SE. **GRU beats per-mouse RL by +0.0136 at
  v2 D=614 (100% of mice, Wilcoxon p=3e-26)** — ~9× the SC effect, ~2× the within-GRU
  D=10→614 gain. Even v2 D=10 wins on 97% of mice. Run
  [`cdq292n5`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/cdq292n5);
  see r8 in [`analysis/reports/`](analysis/reports/INDEX.md) and analysis/rl_baseline_verdict.md.
