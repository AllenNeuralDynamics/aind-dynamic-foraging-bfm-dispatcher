# Timing-inputs study — results index

Study cover (question, verdict, provenance): [../../README.md](../../README.md).

| id | slug | status | one-line headline |
|---|---|---|---|
| [r1](r1-block-decomposition.md) | block-decomposition | live | prev-trial RT + licking adds a real, FLAT ~+0.0075 held-out likelihood carried ~98% by lick counts; the raw curve's growth-with-D is a vanishing overfit tax, not scaling information |
| [r2](r2-selection-reliability.md) | selection-reliability | live | D≤30 cells are untrustworthy because `best_eval` selects on a within-subject signal that has itself collapsed (corr with held-out ≈0 at D=10, →1.0 at D=614) |
| [r3](r3-why-features-help.md) | why-features-help | live | two logistic regressions on the HELD-OUT mice: prev-trial RT predicts switching reward-INDEPENDENTLY (engagement); lick count is reward-GATED (no-reward→abandon); coupling generalizes from train → why the held-out gain works; per-mouse cluster bootstrap |

## Conventions

**r1 and r2** are produced by a single script (`analysis/timing_scaling.py`, which
pulls the live W&B grid, writes `analysis/timing_scaling.json` +
`timing_scaling.csv` + the three study-root figures, and calls
`analysis/update_reports.py` to regenerate the `<!-- BEGIN result-N -->` /
`<!-- END result-N -->` blocks). Prose outside those markers is human-edited.
Regenerate with `make -C studies/07-gru-timing-inputs` (needs `WANDB_API_KEY`).

**r3** is a behavioural-mechanism report produced separately by
`analysis/why_features_help.py` — it reads the foraging DB directly (not W&B),
writes `fig_why_features_help.png` + `why_features_help.csv`, and is fully
hand-authored prose (no BEGIN/END blocks). Needs DB access + `statsmodels`,
`scikit-learn`, and `matplotlib`.

**Metric.** Unlike study-02, this study leaves the output unchanged (2-way choice,
`n_classes=2`), so held-out MOUSE likelihood (`heldout/final/eval_likelihood`) is
directly comparable across all five arms and to study-01. Within-subject
likelihood is a selection-reliability diagnostic only (r2), never a headline.
