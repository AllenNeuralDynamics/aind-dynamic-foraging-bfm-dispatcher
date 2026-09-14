---
id: r3
slug: penalty-edge-extension
status: live
authors: [han]
wandb_groups:
  - mult-d-grid@20260718-151409                                # study 06 wave 1, 80 runs
  - wave2-step-budget-and-penalty-extension@20260909-021546     # study 06 wave 2, 8 runs
inputs:
  script: analysis/r3_penalty_edge_extension.py
  pull_script: analysis/pull_wave2_grid.py
  data_wave1: analysis/grid.csv
  data_wave2: analysis/grid_wave2.csv
  summary: analysis/r3_summary.json
  figure: reports/fig_r3_penalty_edge_extension.png
reproduce: make -C studies/06-disrnn-operating-point-at-scale r3
---

# r3 — Pushing past `mult-d-grid`'s tested penalty floor: one axis keeps helping, the other reverses

**Question.** `mult-d-grid`'s tuned operating point (mult=1, β=3e-4) sat at the *lightest* corner of
the tested penalty range, and its D=614 generalization gap did not fall monotonically with penalty
strength across that range — hinting the true optimum might sit beyond the tested floor, not inside
it. Wave 2 tests one step lighter on each penalty axis independently, at D∈{300,614}, 2 seeds:
**mult=1, β=1e-4** (every penalty term lighter) and **mult=0.5, β=3e-4** (only the interaction term
lighter). Does held-out likelihood keep rising? Does the generalization gap keep shrinking, or turn
around?

**Launch verification (independent of chat-reported numbers, per this study's standing convention).**
Pulled all 8 wave-2 runs directly from W&B via GraphQL (`wandb.Api()` is blocked in this sandbox —
see `pull_wave2_grid.py`'s docstring for the HPC-side equivalent). **8/8 `state=="finished"`**, final
step 107,559–107,562 each (full 107.5k-step budget), zero backfilled rows. Confirms the reconciliation
reported at launch close-out.

**Verdict: no single answer — the two axes disagree.** Lightening the **global penalty β** (mult=1
fixed) keeps helping past the tested floor: held-out likelihood rises monotonically at **both**
D=300 and D=614 as β drops from 1e-3 → 3e-4 → 1e-4, with the new point beating the old tuned point
by a margin that clears seed noise (D=614: +0.0020, ≈5σ against this cell's own combined SEM;
D=300: +0.0038, ≈2.4σ). Lightening only the **interaction multiplier** (β=3e-4 fixed, mult=1→0.5)
does the opposite: held-out likelihood **drops** relative to mult=1 at both D, and mult=1 turns out
to be an interior peak on this axis, not an edge — the "does the edge keep improving" hypothesis is
confirmed on the β axis and rejected on the multiplier axis.

![Going lighter than mult-d-grid's tested floor on the β axis keeps raising held-out likelihood at both D=300 and D=614 with the generalization gap flat, not shrinking further; going lighter on the interaction-multiplier axis alone reverses — mult=1 is an interior peak, and mult=0.5 is worse on both held-out likelihood and the gap.](../reports/fig_r3_penalty_edge_extension.png)

## What the extension says

**1. The β axis has not converged — mult=1, β=1e-4 is the new best point tested, at both cohort
sizes.**

| D | β=1e-3 | β=3e-4 (old tuned pt.) | β=1e-4 (new) | Δ (new − old) |
|---|---|---|---|---|
| 300 | 0.7167 | 0.7192 | **0.7229** | +0.0038 (z≈2.4) |
| 614 | 0.7160 | 0.7221 | **0.7241** | +0.0020 (z≈5.2) |

(mult=1 throughout; n=2 seeds/cell; z = Δ / combined SEM of the two cells being compared.) The
D=614 jump clears seed noise convincingly; the D=300 jump is directionally identical and a smaller,
but still a 2.4σ, effect. **β=1e-4 beats every one of the 10 (mult, β) settings tested across both
grids at both D** — it is not a tie with the old point, it is a new, unambiguous best.

**2. But this gain does not come from suppressing overfitting further — the generalization gap is
flat, not shrinking, once β is already at 3e-4.** At D=614 the gap barely moves as β drops from 3e-4
to 1e-4 (0.0069 → 0.0069, no measurable change); at D=300 it if anything ticks up slightly (0.0081 →
0.0084), well inside the seed SEM (~0.0014–0.0016) at that cohort size. The gap's real drop happened
earlier, going from β=1e-3 to β=3e-4 (already reported in r2); past that point, a lighter β keeps
raising **both** in-sample and held-out likelihood together rather than closing the gap between
them — the same "fits and transfers, not just fits-less-badly" pattern r2 found for the original
tuned point (r2 point 5), now shown to extend past the original grid's edge too.

**3. The interaction-multiplier axis has converged, and mult=1 is an interior optimum, not an edge.**
At fixed β=3e-4, held-out likelihood across mult∈{10,5,2,1,0.5} is **not** monotonic in mult — it
dips through mult=5→2, peaks sharply at mult=1, then drops again at the new mult=0.5 point:

| D | mult=10 | 5 | 2 | 1 (old tuned pt.) | 0.5 (new) |
|---|---|---|---|---|---|
| 300 | 0.7170 | 0.7184 | 0.7183 | **0.7192** | 0.7168 |
| 614 | 0.7191 | 0.7183 | 0.7164 | **0.7221** | 0.7207 |

mult=1 is simultaneously the best point on held-out likelihood *and* the smallest generalization gap
on this axis (D=614 gap: 0.0071, 0.0079, 0.0084, **0.0069**, 0.0076 for mult=10→0.5) — going lighter
here reverses both metrics rather than extending an edge trend. The effect is directionally
consistent at both D but individually weaker than the β-axis result (D=614 z≈-1.0, D=300 z≈-1.5 for
mult=0.5 vs mult=1) — read it as "no evidence lighter helps, some evidence it hurts," not as a
clean reversal on its own; the interior-peak shape of the other four points (10,5,2,1) is the
stronger part of the claim.

**4. The new point closes more of the GRU gap.** Against study 01's GRU curve (D=300: 0.7267;
D=614: 0.7268), the old tuned point trailed by 0.0075 (D=300) / 0.0047 (D=614); the new β=1e-4 point
trails by only **0.0038 / 0.0027** — closing roughly half the remaining gap at each cohort size. The
mult=0.5 point does not help here: it trails by 0.0099 (D=300, now also nominally *below* the
per-mouse RL baseline of 0.7170 by 0.0002 — within noise, but no longer a clean win) / 0.0061
(D=614).

## Reading — what to do next

The β axis is not exhausted: two points (3e-4, 1e-4) is not enough to know whether held-out
likelihood keeps rising, saturates, or eventually turns over the way the multiplier axis did — a
plausible next step is one more point lighter still (e.g. β≈3e-5) to bracket where this axis
actually peaks, if a future wave has budget for it. The multiplier axis, by contrast, looks settled:
mult=1 is bracketed by worse points on both sides (2 and 0.5) at both D, which is the shape of a
genuine interior optimum rather than an artifact of the tested range's edge.

## Caveats

- **n=2 seeds/cell throughout** (both grids) — every comparison here is a 2-vs-2 seed comparison;
  the β-axis D=614 result is the only one clean enough (z≈5) to trust without more seeds, the rest
  are suggestive-but-not-definitive on their own and are reported with that hedge.
- **D=300 leans on `mult-d-grid`'s backfilled, early-stopped runs; D=614 does not.** Four of
  `mult-d-grid`'s 6 historically-backfilled cells fall inside the (D, mult, β) settings this report
  compares, and **all four are at D=300**: the old tuned point's seed-1 side
  (`…-33c0e6f5`, crashed at step 78,400 of 107,560), mult=2's seed-0 side (`…-d9e2902c`, step
  63,680), and **both** seeds of mult=5 (`…-5d00f24c` step 59,900, `…-eb2f9dc2` step 50,250) — every
  one recovered from the per-subject table by `backfill_lost_heldout.py`, not a fresh end-of-training
  value. Every D=614 cell used here, including the old tuned point, mult=2, mult=5, and mult=10, is
  `state=="finished"` on both seeds with zero backfilled rows. Given Stage A's finding that
  within-training likelihood keeps declining toward the final checkpoint, an early-stopped comparator
  is if anything *biased toward whatever point it sits at* (not systematically toward or against the
  new wave-2 points), so this does not undermine the D=300 β-axis win, but it does mean the D=300
  numbers — including the multiplier-axis interior-peak shape in panel b — are not all on fully
  equal training-budget footing. D=614 is the cleaner comparison and the one the ≈5σ claim rests on.
- **Different axes, not a single "penalty knob."** β scales five of six penalty terms uniformly;
  mult scales only the interaction term on top of β. The two "one step lighter" points therefore are
  not comparable points on one 1-D line — this is deliberate (Stage B tested each axis independently,
  see wave2 variant notes), but a reader should not average them into a single "did lighter help"
  verdict.
