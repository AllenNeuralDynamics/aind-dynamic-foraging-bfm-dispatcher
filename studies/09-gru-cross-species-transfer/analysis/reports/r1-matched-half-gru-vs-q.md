---
id: r1
slug: matched-half-gru-vs-q
status: superseded
authors: [han, codex]
wandb_groups:
  - gru-grossman-matched-half@20260905-022602
  - gru-chen-matched-half@20260905-024731
  - gru-zid-matched-half@20260905-025752
  - q-matched-half@20260905-024031
inputs:
  successor: analysis/reports/r2-author-aligned-baselines.md
reproduce: make -C studies/09-gru-cross-species-transfer r2
---

# Result 1 — Superseded

This result has been consolidated into
[Result 2 — GRU, common Q, and author-aligned baselines](r2-author-aligned-baselines.md),
which uses the same frozen matched-half data and now contains the full source-D comparison.
