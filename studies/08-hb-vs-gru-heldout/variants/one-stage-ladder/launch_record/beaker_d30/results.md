# beaker_d30 launch

- W&B group: `one-stage-ladder@20260903-052244`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker experiment: https://beaker.org/ex/01M1KKHNKZ67YGVVGRDH2G009B (`ai1/octo-hub-onprem-h200`, normal priority, 1 GPU / 128 GiB per task)
- Submitted: 2026-09-03 05:22 PT
- Settled: 2026-09-03 11:58 PT
- Status: success (1/1)
- Key numbers: D≈29 at 500 warmup / 500 draws; matched held-out likelihood 0.70981; worst R-hat 1.025, lowest bulk ESS 177, 8 divergences. Fails the convergence gates (R-hat ≤ 1.01, ESS ≥ 400).
- Feeds reports: r4
- Notes: supersedes `beaker_d30_attempt1_cancelled`. Its sampler settings are lighter than the ladder's, so it is a sampler data point, not a rung.
