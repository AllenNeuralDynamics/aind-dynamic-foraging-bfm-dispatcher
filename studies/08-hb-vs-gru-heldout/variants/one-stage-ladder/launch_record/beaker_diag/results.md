# beaker_diag launch

- W&B group: `one-stage-ladder@20260903-230522`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling
- Beaker experiment: https://beaker.org/ex/01M1NGB9S8TAP8MH3R165T6RSC (`ai1/octo-hub-onprem-h200`, normal priority, 1 GPU / 128 GiB per task)
- Submitted: 2026-09-03 23:05 PT
- Settled: 2026-09-04 12:35 PT
- Status: success (1/1)
- Key numbers: D≈29 at the ladder's settings with extra NUTS fields logged; matched held-out likelihood 0.70984, worst R-hat 1.045, lowest bulk ESS 103, 0 divergences. These are identical to the ladder's D≈29 task, so the fit reproduces at the same seed.
- Feeds reports: none directly. Its run is in `analysis/convergence.json`, but r4 does not list the group.
- Notes: trigger and outcome are in `../beaker_diag.json`. The probe ruled out tree-depth truncation, adaptation and heavy tails, which motivated the draws probe.
