# Beron (mouse) RFLR result

- W&B: [run kx7rc3xy](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/kx7rc3xy)
- SLURM job: `25582233` (`COMPLETED`, exit `0:0`)
- Held-out trials: `188,926`
- Normalized likelihood: `0.8223582427`
- Mean log likelihood: `-0.1955791605` nats/trial
- Brier score: `0.0550263105`
- Accuracy: `0.9282735039`

The CPU-only job ran on Allen HPC and feeds Result 1. The RFLR equations match
the pinned author implementation; bounded differential evolution replaces the
original unconstrained SGD for the matched-half subject fits.
