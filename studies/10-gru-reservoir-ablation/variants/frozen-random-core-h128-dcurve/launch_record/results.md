# Results: H128 reservoir scaling curve

- W&B group:
  [`frozen-random-core-h128-dcurve@20260908-111444`](https://wandb.ai/AIND-disRNN/gru_reservoir_ablation/groups/frozen-random-core-h128-dcurve%4020260908-111444)
- Beaker experiment:
  [`01M213NMJA2PSGPWX5EN28FYZM`](https://beaker.org/ex/01M213NMJA2PSGPWX5EN28FYZM)
- Resources: 12 independent one-H200 GPU tasks, 12 CPU, 90 GiB each.
- Status: all 12 cells completed successfully at 90,000 training steps.

| nominal D | realized D by seed | frozen-reservoir mean likelihood |
|---:|---|---:|
| 10 | 10, 10, 10 | 0.711852 |
| 30 | 29, 30, 30 | 0.711091 |
| 100 | 99, 101, 101 | 0.711088 |
| 300 | 300, 301, 300 | 0.711413 |

Every run passed the full frozen-parameter audit: the GRU and session-conditioning
parameters were bitwise unchanged; source embeddings and readout parameters changed.
The D=614 curve cell is recorded separately as the accepted pilot in Report 1.

