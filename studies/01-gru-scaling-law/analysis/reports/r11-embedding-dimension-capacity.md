---
id: r11
slug: embedding-dimension-capacity
status: planned
authors: [han, codex]
wandb_groups:
  - e4-e8-d614-source@20260906-195409
inputs:
  script: analysis/report_embedding_dimension.py
  freezer: analysis/freeze_embedding_dimension.py
  data: analysis/embedding_dimension_results.json
  figures:
    - analysis/fig_embedding_dimension_source.png
    - analysis/fig_embedding_dimension_spectrum.png
reproduce: make -C studies/01-gru-scaling-law r11
---

# Result 11 — is E=4 a source subject-embedding capacity bottleneck?

This report compares newly paired E=4 and E=8 GRUs trained from scratch on the
same 614 AIND source mice. The three source seeds share the `20260603` snapshot,
H=128 core, training recipe, held-out mice, adaptation protocol, and current
code. The historical E=4 result is not the primary comparator because it would
confound embedding dimension with software and execution drift.

The primary outcome is normalized likelihood on the immutable held-out half of
149 AIND mice after adapting only their subject embeddings. We also report the
test-minus-adaptation likelihood gap and the covariance spectrum across the 614
learned source embeddings. The spectrum asks whether PCs 5–8 carry variation;
it does not by itself prove that the GRU uses that variation predictively.

<!-- BEGIN result-11 -->

Awaiting the three E=4 and three E=8 source artifacts.

<!-- END result-11 -->

## Interpretation boundary

Embedding axes can rotate, permute, and rescale with compensating downstream
weights. We therefore interpret the covariance spectrum within each fitted
model, not individual coordinate identities or raw variance magnitudes between
E=4 and E=8. External predictive value is tested separately in Study 09.
