# beaker_draws2000 launch

- W&B group: `one-stage-ladder@20260904-082628`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker experiment: https://beaker.org/ex/01M1PGEPRYGY542Y5JBQB17EED (`ai1/octo-hub-onprem-h200`, normal priority, 1 GPU / 128 GiB per task)
- Submitted: 2026-09-04 08:26 PT
- Settled: 2026-09-05 12:17 PT (last of two tasks)
- Status: success (2/2)
- Key numbers: 2000 warmup / 2000 draws; matched held-out likelihood / worst R-hat / lowest bulk ESS / divergences:
  - D≈10: 0.70972 / 1.005 / 756 / 15
  - D≈29: 0.70980 / 1.008 / 390 / 5

  The nearest any run came to the gates: R-hat passes at both, but D≈29's ESS is just under 400 and both runs have divergences.
- Feeds reports: r4 (also the NumPyro comparator for the Stan arm at D≈29)
- Notes: two parallel tasks.
