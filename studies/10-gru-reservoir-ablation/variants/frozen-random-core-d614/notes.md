# frozen-random-core-d614

The only scientific change from Study 01 v2 is
`model.training.freeze_gru_core=true`. H=128, D=614, E=4, the 20260603 source
snapshot, session-conditioning schedule, training cap, early-stop gate, source
checkpoint selection, and 500-step held-out embedding conditioning are matched.

Execution uses two launches without duplicating a scientific cell:

1. `sweep-seed0.yaml` runs seed 0 as the end-to-end GPU smoke.
2. At the investigator's request, `sweep-seeds1-2.yaml` started the remaining
   paired seeds before seed 0 completed. The same audit gates apply to all
   three completed outputs.

Both launches use Beaker hub GPU infrastructure. No CPU-only task belongs in
this experiment.

Seed 0 submitted 2026-09-07 17:55 PT: Beaker
[`01M1Z86SE16551913JFAA8D189`](https://beaker.org/ex/01M1Z86SE16551913JFAA8D189),
group `frozen-random-core-d614@20260907-175533`.

Seeds 1 and 2 submitted 2026-09-07 18:41 PT: Beaker
[`01M1ZATF3VZS00RZ82JQTWWQSG`](https://beaker.org/ex/01M1ZATF3VZS00RZ82JQTWWQSG),
group `frozen-random-core-d614@20260907-184115`.
