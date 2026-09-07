# E=8, D=614 diagnostic transfer

**Question.** Does doubling the source GRU subject-embedding dimension from
E=4 to E=8 improve frozen-core transfer on cohorts where the current D=614
model ranges from strong to weak relative to common Q?

The grid crosses Grossman (mouse), Lebedeva (mouse), Miller (rat), Findling
(human), and Eckstein (human) with the three paired E=8 source seeds. Only the
external subject embedding is adapted, for 500 steps at learning rate 0.001;
the GRU core remains frozen. Each cell consumes the existing immutable v1/v2
manifest and scores the same held-out trial keys as the E=4 D=614 result.

The E=8 screen is one 15-task GPU-only Beaker launch from `sweep.yaml`. Each
task requests one GPU and may schedule on either S3-capable H200 hub. Do not
launch until `source_runs_e8.json` contains three completed, digest-pinned
source artifacts.

When the three paired current-code E=4 sources land, `sweep_e4_pair.yaml` runs
the identical 15 target cells from `source_runs_e4_pair.json`. That is the
primary E=4 comparator; the older Study 09 D=614 E=4 cells are retained only as
a software-drift reference.

Status: waiting for the three E=8 Study 01 source runs in issue #148.
