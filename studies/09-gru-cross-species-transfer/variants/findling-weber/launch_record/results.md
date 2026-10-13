# Findling (human) Weber-imprecision result

Primary author-code-parity fit:

- W&B: [run sls99so5](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/sls99so5)
- SLURM job: `25582235` (`COMPLETED`, exit `0:0`)
- Fit/evaluation particles: `2` / `256`
- Held-out trials: `11,706`
- Normalized likelihood: `0.6738432502`

Particle sensitivity:

- W&B: [run dgxa85fc](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/dgxa85fc)
- SLURM job: `25582241` (`COMPLETED`, exit `0:0`)
- Fit/evaluation particles: `64` / `256`
- Held-out trials: `11,706`
- Normalized likelihood: `0.6876640792`

Both CPU-only jobs ran on Allen HPC and feed Result 1. Raising only the fitting
particle count recovers `0.0138208290` likelihood and nearly reaches common Q
(`0.6907667918`), showing that the released two-particle objective contributes
material fitting noise. It does not explain the lower GRU score (`0.6547256936`).
