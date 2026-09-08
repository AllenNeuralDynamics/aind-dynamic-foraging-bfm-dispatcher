---
id: r1
slug: frozen-random-reservoir
status: draft
authors: [han]
wandb_groups:
  - frozen-random-core-d614@20260907-175533
inputs:
  script: analysis/report_reservoir.py
  data: analysis/reservoir_results.json
  reference: reference/study01-trained-gru.json
  figure: analysis/fig_reservoir_vs_trained.png
reproduce: make -C studies/10-gru-reservoir-ablation r1
---

# Result 1 - frozen-random GRU reservoir

<!-- BEGIN result-1 -->
Awaiting the seed-0 validation and seeds-1/2 completion launch.
<!-- END result-1 -->

## Interpretation boundary

This first pass tests sufficiency in the source domain only. A competitive
reservoir would show that trained recurrent dynamics are not necessary for this
particular held-out-subject result; it would not establish that learned dynamics
are irrelevant to external-task transfer or richer behavioral targets.
