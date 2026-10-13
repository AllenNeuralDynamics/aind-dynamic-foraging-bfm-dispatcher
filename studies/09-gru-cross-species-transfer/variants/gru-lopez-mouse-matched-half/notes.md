# López-Yépez mouse matched-half GRU transfer

Adapt only each external subject embedding for 500 steps at learning rate 0.001
with the source GRU core frozen. Score the immutable held-out trials for
D={10,30,100,300,614} and source seeds 0–2.

Run `smoke.yaml` first (D=614, seed 0). Launch the full 15-cell `sweep.yaml`
only after that GPU smoke passes. This variant is GPU-only on Beaker; no
CPU-only work belongs in this experiment.
