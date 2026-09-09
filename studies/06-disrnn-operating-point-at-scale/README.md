# Study: disRNN operating point at scale

*Folder `06-disrnn-operating-point-at-scale`. W&B project
[`disrnn_data_scaling`](https://wandb.ai/AIND-disRNN/disrnn_data_scaling) (shared with study 05).*

**Umbrella question.** How should the disRNN's bottleneck be configured **as the training cohort
grows**? Studies 03 and 05 each pinned one axis and swept the other; this study is the home for the
**penalty × D** surface they leave open — and for any future "how to configure the disRNN at scale"
scan (multiplier, embedding width, lr, …). One study, one row of the verdict table per scan.

## Why this study exists — it reconciles 03 and 05

The two disRNN studies reach opposite-sounding conclusions about the *same* lever, because each held
the other axis fixed:

- **[Study 03](../03-disrnn-beta-scan)** — penalty/multiplier scan at **fixed D=100**. Verdict:
  *sparsity is free* — held-out transfer is flat across the multiplier, so mult=2 was recommended as
  the operating point.
- **[Study 05](../05-disrnn-scaling-law)** — D-scan at **fixed penalty** (mult=2, β=1e-3). Verdict:
  held-out **peaks at ~100 mice then declines**, falls below the best per-mouse RL baseline at
  D=614, and the GRU beats it at every cohort size. But finding #2 flags the cause:
  *a single-seed probe at D=614 with **mult=1, β=3e-4 → 0.7211** jumped +0.0057 above the scaling
  curve, beat the RL baseline, and closed most of the GRU gap.* The decline may be an artifact of
  **holding a too-strong penalty fixed as the cohort grows.**

Neither study can settle this, because it lives on the axis each one froze. "Sparsity is free"
(03, D=100) and "sparsity costs half the GRU gap" (05, D=614) are the **same surface read at two
values of D**. This study measures that surface.

> Evidence for the 2D question has already been leaking across studies — study 05 carries a
> `mult-beta-d614` variant (a penalty scan at D=614) that really belongs here. This study
> **consolidates** it: the penalty×D question gets one home instead of a scatter of one-off variants.

## Verdict

*One row per scan; appended as each variant's report lands.*

| Scan | Question | Headline | Report |
|---|---|---|---|
| penalty selection (existing 03+05 data, **zero new compute**) | Can β be picked from data we already have; does the pick depend on D? | **No** — free at D=100, but the generalization gap **grows with D** (β=3e-4: +0.0027→+0.0083) and the held-out-optimal β is also the most overfit β. Motivates scanning β jointly with D, not fixing it. | [r1](analysis/reports/r1-penalty-selection.md) |
| `mult-d-grid` (D×mult×β, 80 runs, 2 seeds) | Does 05's peak-then-decline vanish at some point on the penalty×D surface? | **Yes** — at mult=1, β=3e-4, held-out transfer rises monotonically to 0.7221 at D=614, clears the best per-mouse RL baseline, and halves the GRU gap (−0.0114 → −0.0047). The decline was a penalty artifact: the right β must scale *down* as the cohort grows, not stay fixed. ✅ complete 2026-08-02 | [r2](analysis/reports/r2-scaling-surface.md) |

## Scans (variants)

### `mult-d-grid` — wave 1 (✅ complete 2026-08-02, 80/80 usable → [r2](analysis/reports/r2-scaling-surface.md))

Originally scoped as a single-operating-point D-scan at the wave-2 winner (mult=1, β=3e-4). **r1
overturned that**: the in-sample-vs-held-out selection plot shows β=3e-4 is also the *most overfit*
penalty, and the generalization gap widens with D — so β cannot be fixed a priori and scanned around;
it needs to be part of the grid itself. Launched as **D{10,30,100,300,614} × mult{1,2,5,10} ×
β{3e-4,1e-3} × seed{0,1} = 80 runs**, n_steps=100000 (GRU-parity budget). β=3e-3 dropped — r1 already
shows it underfits at both D=100 and D=614. Submitted as 8 Beaker experiments (10 tasks each) sharing
one W&B group `mult-d-grid@20260718-151409`, split to stay under Beaker's ~48 KiB payload ceiling on
`experiment.create()` (see [variant notes](variants/mult-d-grid/notes.md) and the beaker-launch
skill's `scheduling-lessons.md` for the general fix). 21/80 jobs were running within minutes of
submission on the low-preemptible burst tier.

**Primary hypothesis (H1) — CONFIRMED.** If 05's decline is a penalty artifact, some point on the
grid is **flat-or-rising** across D (no peak-then-decline) and reaches the RL baseline / closes
most of the GRU gap at D=614. It is: mult=1, β=3e-4 rises monotonically to 0.7221 at D=614, clears
the RL baseline (0.7170), and closes about half the GRU gap. See [r2](analysis/reports/r2-scaling-surface.md)
for the full surface, the β×D crossover that explains it, and the per-series breakdown of which of
the other 7 settings genuinely decline vs. are merely flat.

**Secondary (H2), same rollouts — not yet run.** Study 05 finding #5: the disRNN's generative
switch-curve is *too flat* — bottlenecks prune history-dependence. A more-open bottleneck should
**restore curve shape**. Testable via a follow-on `generative-*` rollout variant on the grid's
winning checkpoint (mult=1, β=3e-4, D=614) — deferred to a future session; this wrap-up closes
wave 1 (H1) only.

**Seed-noise bars to clear** (measured in study 05, same config family): held-out SD ≈ 0.0005;
generative history-curve corr SD 0.0008–0.0020 at D ≥ 100.

### `wave2-step-budget-and-penalty-extension` — wave 2 (🚧 in progress)

Follow-on to `mult-d-grid`'s close-out, motivated by two observations: (1) the D=614
generalization gap does not move monotonically with penalty strength across the tested
range — the lightest tested setting (mult=1, β=3e-4) has the smallest gap, the heaviest
(mult=10, β=1e-3) the largest, suggesting the tested range's edge, not interior, may be the
true optimum; (2) training-step count as a tunable.

- **Stage A** (zero new compute, done 2026-09-09): a step-budget proxy analysis of existing
  `mult-d-grid` runs. Found the grid's true held-out metric was only logged once, at the very
  end of training, so the literal question can't be answered from existing data — only a
  within-training (same-mice held-out session) proxy is available, which shows an
  early-peak-then-decline pattern growing with D.
- **Stage B** (launched 2026-09-09): an 8-task grid extending the penalty range one step
  lighter than `mult-d-grid`'s tested floor on each axis in turn (mult=1,β=1e-4 and
  mult=0.5,β=3e-4), at D∈{300,614}, 2 seeds — testing whether the D=614 gap keeps shrinking
  below the tested floor or turns around. Sets `checkpoint_run_heldout_eval=true`, but
  **correction (same day):** this flag is a no-op for multisubject disRNN/GRU in the current
  wrapper (`is_multisubject` runs unconditionally skip the per-checkpoint held-out eval,
  confirmed in this launch's own live logs) — these runs do **not** get a held-out-vs-step
  curve, only the same single end-of-training held-out value as `mult-d-grid`. The
  step-budget question from Stage A remains open pending a wrapper fix + relaunch. Beaker
  experiment [`01M22QA6NH2MCDCFM8J5YREE61`](https://beaker.org/ex/01M22QA6NH2MCDCFM8J5YREE61),
  W&B group `wave2-step-budget-and-penalty-extension@20260909-021546`. See
  [variant notes](variants/wave2-step-budget-and-penalty-extension/notes.md) for the full
  grid and provenance. Results not yet analyzed.

## Relation to other studies

- **[03](../03-disrnn-beta-scan)** — the D=100 slice of this surface (penalty axis only).
- **[05](../05-disrnn-scaling-law)** — the fixed-penalty slice (D axis only); this study's wave 1
  directly re-runs 05's `dscan-mult2` at the tuned penalty. When r1 lands, 05's verdict gets a
  forward-pointer noting the decline may be operating-point-dependent, corrected here.
- **[01](../01-gru-scaling-law)** — the GRU curve every disRNN curve overlays.

## Reproduce

```
make -C studies/06-disrnn-operating-point-at-scale all   # regenerate reports from committed grid
```
