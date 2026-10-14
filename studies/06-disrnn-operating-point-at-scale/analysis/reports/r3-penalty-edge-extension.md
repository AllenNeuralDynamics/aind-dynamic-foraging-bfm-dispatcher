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
by a margin that clears seed noise (D=614: +0.0020, ≈5σ against the combined SEM of the old
(β=3e-4) and new (β=1e-4) two-seed cells; D=300: +0.0038, ≈2.4σ by the same combined-SEM
calculation). Lightening only the **interaction multiplier** (β=3e-4 fixed, mult=1→0.5)
does the opposite: held-out likelihood **drops** relative to mult=1 at both D, and mult=1 turns out
to be an interior peak on this axis, not an edge — the "does the edge keep improving" hypothesis is
confirmed on the β axis and rejected on the multiplier axis.

![Going lighter than mult-d-grid's tested floor on the β axis keeps raising held-out likelihood at both D=300 and D=614 with the generalization gap flat, not shrinking further; going lighter on the interaction-multiplier axis alone reverses — mult=1 is an interior peak, and mult=0.5 is worse on both held-out likelihood and the gap.](../reports/fig_r3_penalty_edge_extension.png)

## What the extension says

**1. The β axis has not converged — mult=1, β=1e-4 is the new best point tested, at both cohort
sizes.**

<!-- BEGIN result-1 -->
| D | β=1e-3 | β=3e-4 (old tuned pt.) | β=1e-4 (new) | Δ (new − old) |
|---|---|---|---|---|
| 300 | 0.7167 | 0.7192 | **0.7229** | +0.0038 (z≈2.4) |
| 614 | 0.7160 | 0.7221 | **0.7241** | +0.0020 (z≈5.2) |
<!-- END result-1 -->

(mult=1 throughout; n=2 seeds/cell; z = Δ / combined SEM of the two cells being compared.) The
D=614 jump clears seed noise convincingly; the D=300 jump is directionally identical and a smaller,
but still a 2.4σ, effect. **β=1e-4 beats every one of the 10 (mult, β) settings tested across both
grids at both D** — it is not a tie with the old point, it is a new, unambiguous best.

**2. But this gain does not come from suppressing overfitting further — the generalization gap is
flat, not shrinking, once β is already at 3e-4.** At D=614 the gap barely moves as β drops from 3e-4
to 1e-4 (0.0069 → 0.0069, no measurable change); at D=300 it if anything ticks up slightly (0.0081 →
0.0084), well inside the seed SEM (~0.0014–0.0016) at that cohort size. **⚠ D=300 gap caveat:** the
β=3e-4 cell behind the 0.0081 figure includes one backfilled seed whose `eval_ll` and `heldout_ll`
are not from the same training step — see "D=300's generalization-gap cells mix training steps for
backfilled rows" under Caveats below before reading this comparison. D=614 is unaffected. The gap's real drop happened
earlier, going from β=1e-3 to β=3e-4 (already reported in r2); past that point, a lighter β keeps
raising **both** in-sample and held-out likelihood together rather than closing the gap between
them — the same "fits and transfers, not just fits-less-badly" pattern r2 found for the original
tuned point (r2 point 5), now shown to extend past the original grid's edge too.

**3. The interaction-multiplier axis has converged, and mult=1 is an interior optimum, not an edge.**
At fixed β=3e-4, held-out likelihood across mult∈{10,5,2,1,0.5} is **not** monotonic in mult — it
dips through mult=5→2, peaks sharply at mult=1, then drops again at the new mult=0.5 point:

<!-- BEGIN result-2 -->
| D | mult=10 | 5 | 2 | 1 (old tuned pt.) | 0.5 (new) |
|---|---|---|---|---|---|
| 300 | 0.7170 | 0.7184 | 0.7183 | **0.7192** | 0.7168 |
| 614 | 0.7191 | 0.7183 | 0.7164 | **0.7221** | 0.7207 |
<!-- END result-2 -->

mult=1 is simultaneously the best point on held-out likelihood *and* the smallest generalization gap
on this axis (D=614 gap: 0.0071, 0.0079, 0.0084, **0.0069**, 0.0076 for mult=10→0.5) — going lighter
here reverses both metrics rather than extending an edge trend. The effect is directionally
consistent at both D but individually weaker than the β-axis result (D=614 z≈-1.0, D=300 z≈-1.5 for
mult=0.5 vs mult=1) — read it as "no evidence lighter helps, some evidence it hurts," not as a
clean reversal on its own; the interior-peak shape of the other four points (10,5,2,1) is the
stronger part of the claim. **⚠ D=300 interior-peak caveat:** the mult=1 and mult=2 cells above each
include one backfilled seed, and the mult=5 cell's *both* seeds are backfilled — for all of them
`eval_ll` and `heldout_ll` come from different training steps (see Caveats below). The D=300 side of
this table's shape should be read with that in mind; D=614, where the z≈-1.0 figure above comes
from, is fully backfill-free.

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
- **D=300's generalization-gap cells mix training steps for backfilled rows (disclosed per a
  Copilot review on [PR #164](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/pull/164),
  kept as-is by explicit decision — not excluded, not recomputed).** The gap reported above is
  `eval_ll − heldout_ll` per run (`r3_penalty_edge_extension.py:cell_stats`), and for the same 4
  backfilled rows named in the bullet above, these two terms are **not from the same training
  step**. `heldout_ll` was recovered by `backfill_lost_heldout.py` from the per-subject held-out
  table, which only exists because the held-out fine-tune stage ran — and that stage runs at the
  true *end* of training. `eval_ll` (`checkpoint/eval_likelihood`), by contrast, was read from the
  same frozen W&B summary that lost its final write, so it reflects an earlier, mid-training step:
  `final_step` 50,250–78,400 of the 107,560-step budget for these four rows (47–73% of the full
  schedule), not the true end. The reported gap for **3 of the D=300 cells compared in this report**
  — mult=1/β=3e-4 (shared by both tables above, 1 of 2 seeds backfilled), mult=2/β=3e-4 (1 of 2 seeds
  backfilled), and mult=5/β=3e-4 (**both** seeds backfilled) — therefore subtracts a true-end-of-training
  `heldout_ll` from a mid-training `eval_ll`, not a matched pair. Since within-training eval
  likelihood declines toward the final checkpoint (Stage A), a mid-training `eval_ll` reads *higher*
  than a true-end-of-training value would, which **inflates** the computed gap for these cells
  relative to a step-matched comparison. This affects the absolute gap values quoted at point 2
  (0.0081, 0.0084) and the gap column of the point-3 table at D=300 only — **every D=614 cell used
  in this report is backfill-free, so the D=614 gap figures (0.0069→0.0069; 0.0071/0.0079/0.0084/
  0.0069/0.0076) are unaffected.** Per Han's explicit decision, these D=300 numbers are reported
  as-is, with this caveat, rather than excluding the backfilled rows or trying to recompute a
  matched-step gap — the "gap is flat, not shrinking" (point 2) and "mult=1 is an interior peak"
  (point 3) claims at D=300 should be read with this known imperfection in mind.
- **Different axes, not a single "penalty knob."** β scales five of six penalty terms uniformly;
  mult scales only the interaction term on top of β. The two "one step lighter" points therefore are
  not comparable points on one 1-D line — this is deliberate (Stage B tested each axis independently,
  see wave2 variant notes), but a reader should not average them into a single "did lighter help"
  verdict.


## Code-ref confound check

**Trigger.** A [Copilot review comment on PR #164](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/pull/164#discussion_r4264704530)
flagged that the wave-2 runs (wrapper `9595dd37`, dispatcher `4b926dc5`, foraging-models `faa0f5ad`)
and the `mult-d-grid` comparator cell (wrapper `eb5a6bb0`, dispatcher `f8c9e9ec`, foraging-models
`54a8d3a4`) were launched at different code refs, and asked that this either be re-run at matched
refs or be verified not to matter before the penalty deltas above are trusted.

**Scope checked.** Every line-level diff between the two ref pairs (122 wrapper commits, 265
dispatcher commits) that touches a file plausibly on the training/eval path for
`model=disrnn, data=mice_snapshot_scaling, data.ignore_policy=exclude` — the exact config both
compared cells use (confirmed from both variants' committed `sweep.yaml`). Diffs were pulled via the
GitHub compare API and read in full; nothing below is inferred from commit titles or file names
alone.

**File-by-file verdict (wrapper, `eb5a6bb0..9595dd37`):**

| File | Verdict | Evidence |
|---|---|---|
| `model_trainers/disrnn_trainer.py` | safe | Only an import-path swap to the vendored `data_loaders.disrnn_dataset` (see vendoring row below), a pure extraction of the existing subject-count lookup into `_resolve_multisubject_subject_count` (same logic, same inputs), and a `try/except` wrapped around the *post-hoc* `plot_update_rules` call so a plotting failure no longer kills an already-trained-and-evaluated run. None of this touches the loss, penalty application, optimizer, or training loop. |
| `configs/config_disrnn.yaml` | safe | The entire +16-line diff is one new, additive `data.timing_features` block defaulting `enabled: false`. No existing default value changed. |
| `data_loaders/mice.py` | safe for this config | Same import swap. Adds optional `session_split_by_subject` / `trial_split_by_subject` params (default `None`) for an external-target-transfer feature — with both `None` (the case here), the original `compute_train_eval_session_ids` + `rnn_utils.split_dataset` branch is preserved verbatim, just moved under an `if ... is None` guard. Dataset construction is routed through a new `_create_disrnn_dataset` dispatcher that sends continuous-feature requests to a new float-safe builder and everything else (our case — only `animal_response`/`rewarded`, both integer) to the unchanged `dl.create_disrnn_dataset`. The new `_augment_features_with_timing` call is a confirmed no-op here: `resolve_timing_config(None)` returns `TimingConfig(enabled=False)`, and neither variant's `sweep.yaml` sets `data.timing_features.*`. |
| `post_training_analysis/heldout_finetuning.py` | safe for this config | Same import swap. The new `target_data`/`target_bundle` external-target path only engages when a `target_data` config key is present (absent here) — the default branch reproduces the prior call sequence (`_load_heldout_snapshot_selection` → `_build_local_heldout_bundle`) unchanged, with a no-op timing-feature attach for the reason above. The new `selection_policy` field defaults to `"fixed_final"` (an explicit value check, not a behavior change for the default). The new end-of-run `_build_final_target_predictions` + consistency assertion block executes for `ignore_policy=="exclude"` runs, but it only **independently recomputes and asserts equality against** the already-computed `heldout/final/eval_likelihood`; it does not change how that reported number is produced, and a mismatch would raise (crash the run), not silently alter it — since all 8 wave-2 runs finished and reported a value, this path did not perturb the metric. |
| `training_runner.py` | safe | Only change is adding `"hb"` to the multisubject-personalized model-type set and a new `model_type == "hb"` branch; the `disrnn` path is untouched. |
| `utils/multisubject.py` | safe (reasoned) | `merge_datasets_with_subject_index` now forwards `first_dataset.rng` to the merged dataset's constructor when present. However `disentangled_rnns.library.rnn_utils.DatasetRNN.__init__` sets `self.rng = rng if rng is not None else np.random.default_rng()` (confirmed from the pinned library source — the library pin itself, `aind-disentangled-rnns@a9b9978`, is unchanged across this diff), and **neither commit ever threads the run's own seed into any `DatasetRNN.rng`** — at baseline the merged dataset got its own fresh unseeded `np.random.default_rng()`; at head it reuses the per-subject dataset's own (also freshly, independently, unseeded) `np.random.default_rng()`. Batch-sampling randomness under `batch_mode=random` was already not seed-controlled in either commit, so this changes *which* unseeded generator object is used, not whether the result is seed-reproducible — no systematic bias, same class of inherent run-to-run stochasticity as any two independently launched runs. |
| `utils/run_helpers.py` | safe | `DISRNN_META_*`/`BFM_META_*` and `DISRNN_RESTORE_FROM_RUN_ID`/`BFM_RESTORE_FROM_RUN_ID` env-var renames (both read with the new name winning, old name as fallback — no override present for either run) and sibling-checkout dir-name tolerance (`aind-disrnn-dispatcher` vs `aind-dynamic-foraging-bfm-dispatcher`). Affects provenance-stamp plumbing and restore-from-run resolution only, not training/eval computation. |
| `evaluation/disrnn_evaluation.py` | safe | Import-path swap only. |

**Vendoring fidelity (`aind_disrnn_utils.data_loader` → `data_loaders/disrnn_dataset.py`, wrapper
commit `bdbde77b`, PR [#67](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-wrapper/pull/67)).**
Diffed the vendored module against the actual pinned upstream source
(`aind_disrnn_utils@74de874d`, the exact commit `pyproject.toml` depended on before the vendoring
commit removed it). The initial vendoring commit is a byte-identical copy (function bodies unchanged).
Two follow-on commits in the same diff window (`eeac1aa3`, `9afd8e4a`) replace the per-session
`df.query("ses_idx == @ses_idx")` loop with a single `groupby("ses_idx", sort=False)` for performance;
`sort=False` preserves the first-appearance group order that defines `.unique()`'s session-column
index, and the diff's own comments flag the one known divergence (NaN `ses_idx` values, which
`groupby` drops and `.unique()` keeps) as deliberately preserved rather than fixed, keeping it a pure
performance change for the ordinary (no-NaN) case. Both commit messages state the equivalence was
checked directly: *"dataset tensors, metadata, and the add_model_results dataframe still compare
equal to the pinned upstream under both ignore policies."* This covers exactly the two `ignore_policy`
values (`include`, `exclude`) relevant here. **Caveat: this is the author's own commit-message claim,
not independently re-executed in this check** — no compute access was used here to re-run the
old-vs-new builder on real data; the claim is accepted on the strength of the diff reasoning above
(order-preservation under `sort=False`, no-NaN assumption) plus the explicit author verification,
not independently reproduced.

**Dispatcher (`f8c9e9ec..4b926dc5`), config files reachable by default (not overridden) from either
run's `sweep.yaml`:**

| File | Verdict | Evidence |
|---|---|---|
| `config/data/mice_snapshot_scaling.yaml` | safe | +22/-0 diff is the identical additive `timing_features` block (default `enabled: false`); no existing key's value changed. |
| `config/model/disrnn.yaml` | safe | Only a comment fix and `DISRNN_OUTPUT_DIR` → `BFM_OUTPUT_DIR` env-var rename, with the old name kept as a fallback (`${oc.env:BFM_OUTPUT_DIR,${oc.env:DISRNN_OUTPUT_DIR,/results}}`) — resolves to the same path either way. |
| `config/config.yaml` | safe | Same env-var-rename-with-fallback pattern, no value change. |

**`foraging_models` SHA (`54a8d3a4` vs `faa0f5ad`): not applicable to this path.** Grepped every
`.py` file under the wrapper's `code/` tree at the wave-2 commit for `foraging_models` /
`dynamic_foraging_models` references: the package is imported only by the baseline-RL trainer, the
new `hb_trainer.py`, and generative/likelihood-comparison analysis modules. Neither
`disrnn_trainer.py` nor `heldout_finetuning.py` reference it. The foraging-models ref mismatch cannot
affect a disRNN run's training or held-out-eval numerics.

**Overall verdict: CONFIRMED SAFE** for the specific config `model=disrnn,
data=mice_snapshot_scaling, data.ignore_policy=exclude` that both the `mult-d-grid` comparator cell
and the 8 wave-2 runs use. Every change exercised by this config between the two ref pairs is either
(a) purely additive and gated behind a config key neither run's `sweep.yaml` sets, (b) a refactor
whose output-equivalence is demonstrated or directly reasoned from the diff, or (c) unrelated to the
disRNN training/held-out-eval code path. The r3 penalty-axis deltas above are not confounded by the
code-ref mismatch flagged in the Copilot review.

**What was not independently re-executed** (provisional, flagged per this check's own standard):
the vendoring-equivalence claim rests on the wrapper-repo author's own commit-message verification,
not a fresh run of old-vs-new `create_disrnn_dataset` on real data from this session; and the
`multisubject.py` rng-forwarding analysis rests on reading the pinned `disentangled_rnns` library
source rather than executing it. Both are reasoned to be safe from the diff evidence above, but
neither was re-run end-to-end as part of this check.
