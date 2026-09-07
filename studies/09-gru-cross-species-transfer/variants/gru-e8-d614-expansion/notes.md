# E=8, D=614 valid-cohort expansion

**Question.** Does the E=8 transfer result generalize from the five-cohort
diagnostic panel to every valid, non-quarantined Study 09 cohort?

The completed diagnostic results for Grossman (mouse), Lebedeva (mouse),
Miller (rat), Findling (human), and Eckstein (human) are reused. This expansion
adds the remaining valid cohorts in the Report 1 order:

- primary: Zid (human), Chen (mouse), and Beron (mouse);
- stress test: López-Yépez (mouse), Alsiö (rat), and Costa (macaque);
- descriptive: Tang (macaque).

Kwak (mouse) remains quarantined because its frozen manifest adapts on CNO
sessions and tests on DMSO sessions. E8 is not run on that invalid estimand.

Every cell uses the three immutable E=8, D=614, H=128 source checkpoints from
`source_runs_e8.json`. Only the external subject embedding is adapted, for 500
steps at learning rate 0.001; the GRU core remains frozen. The existing v1/v2
manifests and held-out trial keys are unchanged.

The 21 cells are split into a 9-task primary shard and a 12-task
stress/descriptive shard to remain below the resumable launcher's approximate
15-task payload limit. Both shards are GPU-only Beaker launches on hub H200
clusters using low-priority preemptible tasks with automatic resume.

Status: launch configuration prepared; submission pending capacity validation.
