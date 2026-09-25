# Miller (rat) RHG result

- W&B: [run 1rfk72im](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/1rfk72im)
- SLURM job: `25582234` (`COMPLETED`, exit `0:0`)
- Held-out trials: `515,238`
- Normalized likelihood: `0.6157150537`
- Mean log likelihood: `-0.4849709976` nats/trial
- Brier score: `0.1579586107`
- Accuracy: `0.7705468153`

The CPU-only job ran on Allen HPC and feeds Result 1. Its ordered held-out
trial-key hash exactly matches the common-Q result. Under this matched-half
screen, RHG (`0.615715`) exceeds common Q (`0.609445`) and GRU D=614
(`0.600067`).
