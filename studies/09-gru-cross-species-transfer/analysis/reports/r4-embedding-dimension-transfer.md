---
id: r4
slug: embedding-dimension-transfer
status: planned
authors: [han, codex]
wandb_groups: []
inputs:
  script: analysis/report_embedding_dimension.py
  freezer: analysis/freeze_embedding_dimension.py
  data:
    - analysis/embedding_dimension_results.json
    - source_runs_e4_pair.json
    - source_runs_e8.json
  figures:
    - analysis/fig_embedding_dimension_transfer.png
    - analysis/fig_subject_embedding_dimension_delta.png
reproduce: make -C studies/09-gru-cross-species-transfer r4
---

# Result 4 — does a wider subject embedding improve external transfer?

This report compares paired current-code E=4 and E=8 source GRUs at D=614 and
H=128. Both dimensions use the same three source seeds, AIND source snapshot,
training recipe, external adaptation observations, 500-step adaptation at
learning rate 0.001, and immutable held-out trials.

The diagnostic cohorts are Grossman (mouse), Lebedeva (mouse), Miller (rat),
Findling (human), and Eckstein (human). The source core stays frozen; only each
external subject embedding is adapted.

<!-- BEGIN result-4 -->

Awaiting the three E=8 and three paired E=4 source artifacts and their matched
Study 09 transfer runs.

<!-- END result-4 -->
