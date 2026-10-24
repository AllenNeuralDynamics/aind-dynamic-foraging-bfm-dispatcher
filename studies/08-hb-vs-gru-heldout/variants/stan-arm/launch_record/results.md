# stan-arm launch `hb-stan-d29-d100`

Covers the one Stan group with a finished run. The four later groups
(`stan-arm@20260905-115323`, `-121427`, `-174213`, `-180320`) have no finished run; each
one's record says how it ended.

- W&B group: `stan-arm@20260905-020928`
- W&B project: https://wandb.ai/AIND-disRNN/mice_data_scaling (sweep https://wandb.ai/AIND-disRNN/mice_data_scaling/sweeps/4i17yn5r)
- HPC: SLURM array 25579912, `aind` partition, 8 CPUs / 128G / 72 h per task (record: `hpc_stan_arm.json`)
- Submitted: 2026-09-05 02:09 PT (tasks started 02:11 PT)
- Settled: 2026-09-05 16:56 PT (D≈29 task completed; the D≈99 task was cancelled at 11:52 PT)
- Status: partial (1/2): D≈29 finished, D≈99 cancelled at 600/16000 iterations
- Key numbers: D≈29 at 2000 warmup / 2000 draws / 4 chains: 53001 s fit, 23 divergences. On the five population means, R-hat is 1.0030 and lowest bulk ESS 1220, recomputed from the saved draws. Against NumPyro (`beaker_draws2000`, D≈29), that is 1.36x ESS per draw, 2.3x wall time and 0.59x ESS per second.
- Feeds reports: r4
- Notes: W&B logged R-hat 1.0049 / ESS 7007 for this run. Those values came from a chain-axis bug in the trainer's diagnostics and are wrong. The Stan fit has no held-out scoring, so it is not an HB rung.
