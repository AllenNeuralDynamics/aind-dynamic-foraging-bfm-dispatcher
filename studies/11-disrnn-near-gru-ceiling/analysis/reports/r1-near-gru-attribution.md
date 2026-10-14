---
id: r1
slug: near-gru-attribution
status: live
authors: [han]
wandb_groups:
  - stage1-no-penalty@20260909-021133
  - stage2-linear-update-net@20260909-021223
  - stage3-wide-latent@20260909-140640
inputs:
  script: analysis/attribution_report.py
  data: analysis/grid.csv
  reference_data: analysis/reference_points.csv
  figure: analysis/fig_gru_ceiling_attribution.png
reproduce: python analysis/pull_grid.py && python analysis/attribution_report.py
---

# r1 — Staged attribution of the residual GRU gap

**Question.** Study 06's tuned disRNN operating point (`mult=1`, `beta=3e-4`, `latent_size=5`,
D=614, n_steps=100000) closes about half the GRU gap but doesn't eliminate it. Zeroing the
interaction-bottleneck penalty, linearizing the update net, and widening the latent — each one
relaxation closer to a GRU-shaped function — should attribute how much of that residual gap is
disRNN's capacity/regularization design versus something else in the architecture.

## ⚠️ Before the numbers: two things had to be re-verified, and both moved the answer

**1. This study deliberately switches the GRU comparator from study 06's H128 reference to
study 01's H256 reference — this is a different model, not a corrected number for the same
one.** Study 06's `analysis/scaling_report.py` hardcodes `GRU = {..., 614: 0.7268}`, which is
study 01's **original fixed-hidden_size=128** curve at D=614 (`studies/01-gru-scaling-law/
README.md`'s v1 historical record). Study 01's later N×D joint scaling grid
(`analysis/reports/r7-nxd-joint-scaling-grid.md`, `nxd-grid@20260623-102649`, 3 seeds, all
`state=="finished"`, all early-stopped at step 90505) reports H128 and H256 **separately** at
D=614: H128=**0.7282**, H256=**0.7290** (seeds 0.72910/0.72896/0.72901, SD<0.0001). Both are
distinct from, and both postdate, the 0.7268 constant. **This report uses H256's 0.7290**, not
because 0.7268 was wrong for what it measured, but because this study's stage 3 was always
designed as a GRU-**H256**-matched capacity probe (the study README's downgrade note explains
why the executed stage 3 ended up at latent_size=32 instead of the planned 256) — H128 is
simply the wrong comparator for that question. Comparing against H256 revises the residual gap
from the commonly quoted **-0.0047** (against H128's 0.7268) to **-0.0069** (`0.7221 - 0.7290`,
against H256's 0.7290). Every number below is computed against the H256 comparator; study 06's
own README/reports still compare against the H128 figure, which is correct for what *that*
study was asking but is not restated here.

**2. `stage1-no-penalty` seed 0 diverged late in training and is excluded from all summary
statistics.** Its `checkpoint/eval_likelihood` step curve tracks the healthy seed almost exactly
through step 87500 (0.7307 vs. 0.7295) then collapses to 0.6811 by step 97500 and stays there —
a late-training instability, not a data-pull artifact (both `checkpoint/eval_likelihood` and
`heldout/final/eval_likelihood` collapsed together, consistent with a bad final checkpoint feeding
the held-out fine-tune). Averaging it in would report `stage1-no-penalty`'s mean as 0.7006 — a
**24σ**-scale outlier relative to every other seed in this study (seed SD elsewhere ≈0.0004-0.0013)
that would make the interaction-penalty relaxation look like it *widens* the gap by 4x. It is
excluded from the trend line and shown separately in the figure (an open red marker), per the
rule that an excluded row never enters a plotted summary.

## Verified numbers

<!-- BEGIN result-2 -->
| quantity | value | source |
|---|---|---|
| study06 tuned point (D=614, mult=1, β=3e-4, n_steps=100000) | seed0=0.72247, seed1=0.72170, **mean=0.72208** | `disrnn_data_scaling` / `mult-d-grid@20260718-151409` |
| GRU ceiling (H=256, D=614) | 0.72910/0.72896/0.72901, **mean=0.72902** | `mice_data_scaling` / `nxd-grid@20260623-102649` |
| baseline residual gap | **-0.00694** | study06 tuned − GRU |
| stage1-no-penalty | seed0=**0.67838 (COLLAPSED, excluded)**, seed1=0.72291 | `disrnn_near_gru_ceiling` / `stage1-no-penalty@20260909-021133` |
| stage2-linear-update-net | seed0=0.72083, seed1=0.72269, mean=**0.72176** | `disrnn_near_gru_ceiling` / `stage2-linear-update-net@20260909-021223` |
| stage3-wide-latent (latent=32) | seed0=0.72251, seed1=0.72313, mean=**0.72282** | `disrnn_near_gru_ceiling` / `stage3-wide-latent@20260909-140640` |
<!-- END result-2 -->

## Gap-closure attribution

![Cumulative held-out likelihood across the staged ablation, against the study-06 tuned disRNN and study-01 GRU (H=256) reference lines. The excluded stage1 seed-0 collapse is shown as an open marker below the main trend.](../fig_gru_ceiling_attribution.png)

<!-- BEGIN result-1 -->
| stage (cumulative) | mean held-out LL | remaining gap to GRU | gap closed vs. baseline | % of baseline gap closed |
|---|---|---|---|---|
| disRNN tuned (study 06, baseline) | 0.7221 | -0.0069 | — | 0% |
| +stage1 (interaction penalty → 0)ᵃ | 0.7229 | -0.0061 | +0.0008 | **+11.9%** |
| +stage1+2 (+ linear update net) | 0.7218 | -0.0073 | -0.0003 | **-4.7%** |
| +stage1+2+3 (+ latent 5→32) | 0.7228 | -0.0062 | +0.0007 | **+10.6%** |
<!-- END result-1 -->

ᵃ stage1's single reliable seed (seed 1); seed 0 collapsed and is excluded (see above).

<!-- NOT auto-regenerated: hand-written interpretation, not a table fill-in (per Copilot
review on PR #163). attribution_report.py's verify_verdict_numbers() re-derives the closure
%s, the -0.0069 baseline gap, and the 0.7218 stage1+2 mean quoted below from the CSVs and
raises make-all-failing AssertionError if a CSV change makes this prose stale; a CSV change
should still prompt a human read-through of the rest of this paragraph's claims. -->

<!-- BEGIN verdict -->
**Verdict: at most ~12% of the residual gap closes, and it does not close monotonically.**
Zeroing the interaction penalty alone (stage 1) recovers essentially all of the closure this
ablation ever achieves. Adding the update-net linearity (stage 2) does not extend it further —
the stage1+2 mean (0.7218) is *below* stage 1's healthy seed and statistically indistinguishable
from the study-06 baseline itself (a 0.0003 difference against a between-seed spread of
0.0009-0.0019 in this study, and ≈0.0005 in study 06). Widening the latent from 5 to 32 (stage 3)
recovers the closure back to roughly stage 1's level (10.6% vs. 11.9%) but does not exceed it.
**None of the three stages, individually or cumulatively, comes close to eliminating the -0.0069
gap** — at latent_size=32, ~89% of the residual gap remains.

**This cannot rule out that GRU-matched capacity (latent_size=256) would close much more.**
Stage 3 as executed is a **downgraded, reduced ceiling probe** at latent_size=32 (a 6.4x widening
from the baseline's 5, but only 1/8 of the H=256 target) — see the study README's downgrade note.
The flat, small effect observed from a 6.4x latent widening is at least *consistent with*
capacity/regularization not being the dominant remaining bottleneck at this scale, but the study
cannot distinguish that from "the effect only shows up much closer to latent_size=256." True
GRU-parity capacity requires the wrapper's `update_latents` `vmap` fix
([wrapper#99](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/issues/99)).
<!-- END verdict -->

## Generalization gap (checkpoint/eval_likelihood − heldout/final/eval_likelihood)

<!-- BEGIN result-3 -->
| stage | seed0 gap | seed1 gap | mean |
|---|---|---|---|
| study06 tuned (baseline) | 0.0067 | 0.0071 | 0.0069 |
| stage1-no-penalty | 0.0044 (collapsed run, not meaningful) | 0.0063 | — (use seed1: 0.0063) |
| stage2-linear-update-net | 0.0073 | 0.0059 | 0.0066 |
| stage3-wide-latent | 0.0060 | 0.0065 | 0.0063 |
<!-- END result-3 -->

All four settings' generalization gaps sit in a tight 0.0059-0.0073 band (0.0063-0.0069 using
only healthy runs). **Relaxing disRNN's penalty, nonlinearity, and width toward a GRU-shaped
function does not measurably change the train/held-out generalization gap at this scale** — the
small held-out gains in the table above come from the model fitting and transferring slightly
better overall, not from suppressing overfitting relative to the tuned disRNN baseline. This
mirrors study 06 r2's own finding that the generalization gap is flat across its penalty×D
surface at D=614.

## Caveats

- **Stage 2 and stage 3's configurations are not interpretable disentangled RNNs.** See the study
  README's caveat — these results characterize a capacity ceiling, not a candidate operating
  point, and must not be folded into study 06's verdict.
- **Stage 3 is latent_size=32, not the originally planned latent_size=256** (see study README
  downgrade note and `variants/stage3-wide-latent/notes.md` for the full OOM/argument-packing
  diagnostic). Any claim about "how close disRNN can get to GRU capacity" is bounded by this —
  the study answers "at latent_size=32," not at GRU-matched capacity.
- **stage1-no-penalty is effectively n=1** after excluding the collapsed seed. A third seed would
  materially strengthen this stage's estimate; none was launched.
- **This report's H256 GRU comparator (0.7290) is deliberately different from study 06's own
  H128 comparator (0.7268)** (`analysis/scaling_report.py`'s hardcoded `GRU` dict, and the prose
  in `r2-scaling-surface.md` quoting "-0.0047"/"0.7268") — not a correction of it. Study 06 was
  asking a different question (does the tuned disRNN close the gap to a GRU at the *same*
  hidden_size budget it was tuned against) and 0.7268 is the right number for that; this study
  needs the H256 figure specifically because stage 3's capacity-match target was always
  latent_size=256. Both numbers are independently real study-01 N×D-grid cells (H128=0.7282,
  H256=0.7290 — this 0.7282 vs. study 06's quoted 0.7268 is itself a later, separately re-run
  N×D-grid value for the same H128 size, not reconciled here since it doesn't change this
  study's own H256-referenced analysis). Reconciling study 06's own report to the later H128
  grid value is out of scope here.
