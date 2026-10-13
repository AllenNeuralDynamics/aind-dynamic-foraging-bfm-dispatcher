---
aliases:
  - data-scaling reports index
  - data-scaling results index
tags:
  - index
  - report
  - data-scaling
---

# Data-scaling study — results index

Study cover (TL;DR, Design, Verdict, Status, Provenance): [../../README.md](../../README.md).

| id | slug | status | one-line headline |
|---|---|---|---|
| [r1](r1-heldout-scaling-curve.md) | heldout-scaling-curve | live | cell-level v1-vs-v2 scaling, 15 matched pairs; SC gain +0.00074 (Wilcoxon p=0.0043) |
| [r2](r2-per-mouse-repeated-measures.md) | per-mouse-repeated-measures | live | n=149 mice/D paired; large-D Wilcoxon p~1e-24; 95% mice improve at D=614 |
| [r3](r3-bootstrap-cis.md) | bootstrap-cis | live | 1000-resample CIs — late gain D=100→614 excludes 0 (small real slope) |
| [r4](r4-zeroshot-vs-adapted.md) | zeroshot-vs-adapted | live | adaptation buys ~+0.002 (flat in D); zero-shot scales then saturates |
| [r5](r5-fewshot-adaptation.md) | fewshot-adaptation | live | K=1 crater (~−0.02), recovers by K=4; D- and variant-independent (protocol artifact) |
| [r6](r6-sc-stage-mature-only.md) | sc-stage-mature-only | live | ~¾ of SC gain persists on mature-only eval — not purely a curriculum-stage artifact |
| [r7](r7-nxd-joint-scaling-grid.md) | nxd-joint-scaling-grid | live | 4×4 N × D grid; E=0.729, α=1.30 (N), β=0.56 (D); D saturates by ~100 at every N |
| [r8](r8-gru-vs-rl-baseline.md) | gru-vs-rl-baseline | live | GRU beats **all three** per-mouse RL models at every (variant, D); vs best (CTT) v2 D=614 **+0.0113** (100% mice) — the old Bari-only +0.0136 overstated it by 20% |
| [r9](r9-generative-behavioral-match.md) | generative-behavioral-match | live | 2nd-order: switch-curve corr 0.96→0.98 vs D; 3-trial-back history corr 0.94→0.99 (detailed n=3); saturates by D≈100 |
| [r10](r10-embedding-decodes-rl-params.md) | embedding-decodes-rl-params | live | **subject embedding decodes classical-RL params**: biasL R²≈0.72, learn_rate ≈0.66 — consistent across CTT *and* Bari; curriculum-only ≈0 (not a task artifact); near-blind to forget_rate (0.08) |
| [r11](r11-embedding-dimension-capacity.md) | embedding-dimension-capacity | live | E8 gives a statistically detectable but tiny source gain (+0.00029); PCs 5–8 carry 18.7% of embedding covariance |

## Conventions

Each report carries YAML frontmatter (`id`, `status`, `wandb_groups`, `inputs`, `reproduce`) and uses `<!-- BEGIN result-N -->` / `<!-- END result-N -->` markers around any region a script regenerates. See [[posthoc-analysis]] for the full contract.

**Provenance gaps surfaced by this migration** (ad-hoc producers, no committed script):

- r4 data (`zeroshot_vs_d.json`)
- r5 data (`fewshot_curve.json`)
- r6 data (`mature_sc_verdict.json` / `.md`)

These are committed artifacts without committed producers — future work to wire dedicated regenerators.

**Cross-producer artifact** (one producer per artifact violated):

- `fig_nxd_scaling.png` is written by `nxd_scaling.py` and then overwritten with an RL-overlay variant by `rl_baseline.py`. Migration target: move RL-reading into `nxd_scaling.py` so the figure has a single producer.

## Related

- [[posthoc-analysis]] — conventions used here (frontmatter, markers, _meta blocks, enforcement).
- [[study-organization]] — folder layout for `studies/<study>/variants/<variant>/` and W&B group naming.
- [[README]] — study-level cover (`studies/data-scaling-law/README.md`).
