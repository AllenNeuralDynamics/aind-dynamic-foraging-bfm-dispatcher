# Results: exact-split D=614 sensitivity replication

- W&B group:
  [`frozen-random-core-d614@20260908-045745`](https://wandb.ai/AIND-disRNN/gru_reservoir_ablation/groups/frozen-random-core-d614%4020260908-045745)
- Beaker experiment:
  [`01M20E3AD2T3EMMAZ9VEYA8267`](https://beaker.org/ex/01M20E3AD2T3EMMAZ9VEYA8267)
- Resources: three independent one-H200 GPU tasks, 12 CPU, 90 GiB each.
- Cluster/priority: `ai1/octo-hub-onprem-h200`, normal, non-preemptible.
- Submitted: 2026-09-08 04:57 PT.
- Settled: 2026-09-08 14:40 PT (last Beaker job finalized).
- Status: all three cells completed successfully at 90,000 training steps.
- Feeds: [Report 1](../../../analysis/reports/r1-frozen-random-reservoir.md), exact paired comparison.

The exact-split reservoir mean likelihood is **0.711126** (SD 0.001322), compared
with **0.728229** (SD 0.000063) for the paired Study 01 trained GRU. The paired
subject-level difference is -0.016328 (bootstrap 95% CI [-0.017289, -0.015381]).

Every frozen-parameter audit passed. Native initial parameter trees, all 149
held-out subject keys, and all held-out trial counts match the paired Study 01
positive-control runs. This is a sensitivity replication and is not pooled as
three additional scaling-curve seeds.
