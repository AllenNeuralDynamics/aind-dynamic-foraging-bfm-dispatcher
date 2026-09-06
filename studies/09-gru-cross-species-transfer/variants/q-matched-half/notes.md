# Q-learning matched-half baseline

Fit one standard subject-level Q-learning model per external subject on the
same adaptation half used by the GRU, then score the same held-out half. The
model has one learning rate, one unchosen-value forgetting rate, a one-step
choice kernel, and softmax action selection.

CPU execution only: a three-task SLURM array on Allen HPC. Each compute task
downloads one pinned public release, verifies its checksum, regenerates the
canonical table and frozen split, then performs the fit. This variant must not
be submitted to Beaker.
