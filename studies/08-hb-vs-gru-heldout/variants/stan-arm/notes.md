# `stan-arm` — the same HB model, fit with pystan

| | |
|---|---|
| **W&B project** | `mice_data_scaling` — study 01's project, same as the NumPyro rungs |
| **Launch surface** | HPC (`sweep_hpc.yaml` + `code/launch_hpc.py --mode cpu`) |
| **Model config** | `code/config/model/hb_hattori_stan.yaml` → `model_trainers.hb_stan_trainer.HBStanTrainer` |
| **Status** | Finished, no further launches planned. The D≈29 fit completed (14.7 h, 8 CPUs); no D≈99 fit completed under any configuration tried. Each attempt and why it ended is in `launch_record/` |
| **Tracked by** | [dispatcher #129](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/issues/129) |

## What differs from `one-stage-ladder`

Only the sampler. Same model (`hattori2019_three_level`, ported to Stan in
`aind-dynamic-foraging-models/.../benchmarks/reference_stan/hb_three_level.stan`), same
loader, same cohort selection, same sampler settings (2000 warmup / 2000 draws / 4 chains),
same seed. The port applies the three parameterisation traps from `docs/design-hb-baseline.md`
rather than inheriting the reference's conventions: `aF` is a *retention* factor there
(`aF = 1 - forget_rate_unchosen`), its `bias` is `-bias_l`, and it branches the learning rate
on `sign(PE)` where we branch on reward — equivalent only while Q stays in `[0, 1]`.

## Why it exists

`benchmarks/RESULTS.md` prices the two frameworks **per subject**, where Stan wins on every
measured axis, then argues NumPyro wins **at cohort scale** from a row it labels
*"inferred, not measured"*. This variant measures that row.

A claim previously used to avoid the question has been withdrawn: the joint estimator is not
inexpressible in Stan — `RESULTS.md` says the opposite one line after the sentence quoted for
it. The defensible argument is narrower, about where the parallelism lives, and it needs this
measurement to stand on.

## Deviations

- **No held-out scoring, so no `heldout/*` metrics.** The comparison is about sampler
  efficiency on the joint fit; held-out adaptation is a separate NumPyro-side machine whose
  port would double the work and measure something this does not turn on. **An HB-Stan run
  is therefore not an HB rung and must not be plotted on the scaling curve.**
- **Only D≈29 has a NumPyro comparator.** The draws-2000 probe (Beaker
  `01M1PGEPRYGY542Y5JBQB17EED`) is the only NumPyro fit finished at these settings. D≈100 is
  run because it is the first rung whose lane count (~20k) is well past the ~1000-lane
  threshold where `RESULTS.md` expects JAX to start winning — i.e. the rung that could
  actually falsify the inferred claim. NumPyro has a finished D≈99 fit at warmup 2000 / 500
  draws, but its chains did not mix (study-08 report r4), so it is not a usable comparator,
  and no D≈99 run exists at these settings. No Stan D≈99 fit has completed under any
  configuration tried (see the launch records).
- **Threading was off in the first launch, and that was my error, not a pystan limit.**
  httpstan *does* compile with `STAN_THREADS` defined (`httpstan/models.py`), so `reduce_sum`
  threads across subjects within a chain — but Stan reads `STAN_NUM_THREADS` at runtime and
  documents one thread when it is unset. Launch `20260905-020928` did not set it, so it was
  *expected* to run one thread per chain on its 8-CPU allocation. That is inference: the effective
  count was not recorded. Two separate `ps` samples were taken during the run (four chain processes at
  roughly 200% CPU each; python processes listing 8 and 10 OS threads, which counts every thread, not only
  `reduce_sum` threads), and neither establishes the `reduce_sum` count. The thread
  count of the reported 14.7 h D≈29 fit is therefore unverified (see its launch record).
  `HBStanTrainer` now derives the count from
  `SLURM_CPUS_PER_TASK // num_chains`.
  **Outcome of trying threads:** the one clean test (exclusive 112-core node, D≈99, 4 chains
  x 16 threads) showed no measurable gain: under 160 iterations in 3 h, about the
  rate of the earlier run that left `STAN_NUM_THREADS` unset. The other threaded attempts are uninformative (one pair shared a node
  with a third job; one pair was cancelled at about 20 minutes). The completed D≈29 fit
  (14.7 h, 8 CPUs) is therefore the only completed real-cohort Stan timing, and is the
  configuration the study reports. See `launch_record/` for each attempt and why it ended.
  A `reduce_sum` reslice was also tried; its effect is unresolved (the records say why).
- **`aind-dynamic-foraging-models` is pinned to a branch on the HPC CPU env.** The `.stan`
  program ships in that package (next to the NumPyro model it ports) and is not in the
  released 0.14.0 the env carried, so the branch is installed `--no-deps` into
  `dynamic-foraging-bfm-cpu`. Recorded in the launch record; reverts by reinstalling the
  release.

## Prior context

The synthetic validation of the port ran first and passed: on a simulated cohort drawn from
the model's own prior, Stan and NumPyro agreed on all five population means to within
**0.078 posterior SD** (worst case). That is the correctness gate — a speed comparison
between two implementations of *different* posteriors would mean nothing. Its timing result
(Stan 2.65× faster) came from a 96-lane problem, roughly 10× **below** the threshold where
`RESULTS.md` expects JAX to start winning, so it says nothing about cohort scale. That is
what these two rungs are for.
