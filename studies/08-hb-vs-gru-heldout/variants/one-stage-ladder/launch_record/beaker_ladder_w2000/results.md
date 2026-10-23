# beaker_ladder_w2000 launch

- W&B group: `one-stage-ladder@20260903-154819`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker experiment: https://beaker.org/ex/01M1MQB0SRRBDDRKP1SX3T55DV (`ai1/octo-hub-onprem-h200`, normal priority, 1 GPU / 128 GiB per task)
- Submitted: 2026-09-03 15:48 PT
- Settled: 2026-09-06 12:44 PT (last of four tasks)
- Status: success (4/4)
- Key numbers: matched held-out likelihood / worst R-hat / lowest bulk ESS / divergences, 2000 warmup / 500 draws / 4 chains:
  - D≈10: 0.70973 / 1.018 / 232 / 6
  - D≈29: 0.70984 / 1.045 / 103 / 0
  - D≈99: 0.71073 / 1.611 / 7 / 0 (chains not mixed; not a result)
  - D≈300: 0.71024 / 1.283 / 11 / 0 (per-mouse chains not mixed; not a result)
- Feeds reports: r1, r2, r3, r4
- Notes: no rung passes the convergence gates. Supersedes `beaker_ladder_try1_cancelled`. The four tasks shared one cluster, so their wall times are contended and are not a timing measurement.
