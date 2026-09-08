# frozen-random-core-d614

The only scientific change from Study 01 v2 is
`model.training.freeze_gru_core=true`. H=128, D=614, E=4, the 20260603 source
snapshot, session-conditioning schedule, training cap, early-stop gate, source
checkpoint selection, and 500-step held-out embedding conditioning are matched.

Execution is staged without duplicating a scientific cell:

1. `sweep-seed0.yaml` runs seed 0 as the end-to-end GPU smoke.
2. After validating frozen-core provenance and the complete held-out table,
   `sweep-seeds1-2.yaml` runs the remaining paired seeds.

Both launches use Beaker hub GPU infrastructure. No CPU-only task belongs in
this experiment.
