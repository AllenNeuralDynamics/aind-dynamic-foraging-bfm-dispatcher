---
name: hpc-launch
description: Launch and monitor dynamic-foraging-bfm (disRNN) training on the Allen on-premise SLURM HPC (AI1) — launch_hpc.py W&B sweep arrays, manual sbatch/Hydra multirun, dry-run/smoke-test patterns, GPU tiers, extend/restore and held-out re-scoring, squeue/sacct monitoring. Use whenever running jobs on the Allen cluster via srun/sbatch rather than Beaker.
---

# Launching on Allen on-prem HPC (SLURM)

Canonical detail: `code/hpc/README.md`. **If this skill and the README conflict, the
README wins.**

## Hard rules first

1. **Never run *any* Python on the login node** (where you run) — no exceptions.
   Training, sweeps, analysis, synthetic data generation, and "quick" smoke tests or
   single-worker checks all go through `sbatch`/`srun`/`salloc` onto a compute node,
   whether run interactively or shipped ad-hoc via `call_command`/SSH. The **only**
   Python allowed on the login node is a *submit-only launcher* (`launch_hpc.py`,
   `launch_beaker*.py`, `check_gpu_availability.py`) — it creates a sweep / submits /
   probes capacity and returns, never importing wrapper loader/model/training code or
   starting a `multiprocessing` Pool. (An unguarded spawn-Pool on the login node once
   recursively forked into a 260 MB / 65k-error process cascade.)
2. Invoke the launcher from the `dynamic-foraging-bfm-cpu` conda env
   (`/allen/aind/scratch/han.hou/miniforge3/envs/dynamic-foraging-bfm-cpu`). The SLURM script
   activates `dynamic-foraging-bfm-cpu` or `dynamic-foraging-bfm-gpu` on the compute node per `--mode`.
3. Sibling layout: the wrapper repo is expected at `../aind-dynamic-foraging-bfm-wrapper`
   (override with `--wrapper-root`), and study sweeps live in the studies repo at
   `../aind-dynamic-foraging-bfm-studies/studies/<study>/variants/<variant>/` (dispatcher #124).
   A hand-written SLURM script that runs study code takes it from `STUDIES_ROOT`, never from
   `DISPATCHER_ROOT/studies` (see study 09's `variants/*/*.slurm`).
4. One-time setup per user: `cp code/hpc/slurm/user.env.example code/hpc/slurm/user.env`
   and edit (`SBATCH_*` vars for sbatch, `CONDA_SH` for env activation).
5. Credentials on HPC live in files, not the env: W&B in `~/.netrc` (the SDK reads it;
   producers using the GraphQL route need `WANDB_API_KEY` exported — see the
   posthoc-reporting skill) and Beaker in `~/.beaker/config.yml`.

## Not everything belongs on HPC

Report generation runs **locally**, not through `sbatch` — a producer on the reproducible
path reads a committed file, so a compute node buys nothing. Only scripts needing the
`wandb` SDK (which cannot start in the sandbox), the `/allen` mount, or large artifact
streaming need a node; W&B reads over GraphQL run locally fine. The check and the
per-script split: the `posthoc-reporting` skill.

## Check available resources FIRST (mandatory for large jobs)

**Before launching any large job (> 4 GPUs / > 4 concurrent array tasks), check
schedulable capacity** and route to whichever backend has room. Hard rule (AGENTS.md §10).

```bash
python code/check_gpu_availability.py --hpc     # this partition (default: aind)
python code/check_gpu_availability.py           # HPC + Beaker, to load-balance
```

The script reports `CfgTRES.gres/gpu − AllocTRES.gres/gpu` on non-drained nodes, broken
down by GPU type (a100/h200/v100/l40s/…), because raw `sinfo` free counts include
`drain`/`down`/reserved nodes. Needs the Allen network (login node, or the sandbox with
VPN up). HPC is the natural overflow for large GPU grids: it often has genuinely idle
GPUs while every Beaker hub cluster is saturated or cordoned.

## Standard launch (W&B sweep + SLURM array)

From the dispatcher repo root:

```bash
conda activate dynamic-foraging-bfm-cpu
python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/<sweep>.yaml \
  --mode gpu \
  --label <short-label> \
  --note "why this run exists + what we want to learn"
```

For a sweep under `studies/<study>/variants/<variant>/`, `--label` and `--note` are
**required** and the group must be `<variant>@<launch_id>`; the launcher refuses otherwise,
before creating the sweep (study-conventions; studies ADR 0001).

`launch_hpc.py` creates the W&B sweep, injects studies + dispatcher + wrapper git lineage as
Hydra `+meta.*` overrides (`+meta.studies_commit` among them), computes `AGENT_COUNT` for grid sweeps, submits the SLURM array,
and submits a cleanup job that stops the sweep after the array drains
(`--no-autostop` to keep it open for more agents).

Useful variants:

```bash
# Inspect planned commands; nothing is created or submitted.
python code/launch_hpc.py --mode gpu --dry-run

# End-to-end smoke test: one tiny run on one CPU array task.
python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/synthetic_disrnn_smoke.yaml \
  --mode cpu --sbatch-extra=--array=0-0 --agent-count 1

# Bounded validation: one grid point of the default sweep.
python code/launch_hpc.py --mode cpu --sbatch-extra=--array=0-0 --agent-count 1

# Specific GPU tier (default gres is gpu:v100:1, array 0-5 — the 6-GPU aind-QOS cap).
python code/launch_hpc.py --mode gpu --gpu-type a100
```

## Manual SLURM routes (skip lineage injection + AGENT_COUNT)

```bash
wandb sweep code/hpc/sweeps/<sweep>.yaml
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper \
  code/hpc/slurm/wandb_sweep_gpu.slurm <SWEEP_ID>

# Deterministic Hydra multirun without a W&B sweep controller:
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper \
  code/hpc/slurm/hydra_multirun_gpu.slurm   # or _cpu.slurm
```

## Resuming, extending & re-scoring runs

HPC `aind` jobs are **not preempted** (queue / fair-share scheduling, not Beaker's
preemptible-priority tiers), so the "resume after preemption" scenario does not arise
here — that mechanism is Beaker-only. The other two work the same as on Beaker (full
detail: wrapper `../aind-dynamic-foraging-bfm-wrapper/code/TRAINING.md` §1.5):

- **Extend a finished run to a longer horizon.** Set
  `model.training.restore_from_run_id=<source W&B run name>` (or env
  `BFM_RESTORE_FROM_RUN_ID`) and a **larger** `n_steps`. `run_hpc.py` downloads the
  source run's `<mtype>-output-<run_id>` artifact (`mtype` ∈ {`disrnn`,`gru`}) into
  `outputs/` before training, so it resumes from the checkpoint and skips warmup.
  Trainer-agnostic. Prereq: the source run must have FINISHED (its `training-output`
  artifact is COMMITTED).
- **Re-score a finished run's held-out stage only — no re-training.** Runs the held-out
  fine-tune off the existing checkpoint and re-injects `heldout/*` into the original
  W&B run — used to backfill metrics added *after* a run trained. The committed
  reference implementations both live under the wrapper's `code/`: the Beaker port
  `code/resume_heldout_beaker.py` (`--run-id <wandb_run_id>`, inside a container reaching
  GCS + W&B) and the HPC original `code/resume_heldout.py` (`--model-dir <dir> --wandb-run-id
  <id>`, run on a compute node that can read the checkpoint tree + reach W&B).

## Recording an intervention (resubmit / rescue / probe / backfill)

`launch_hpc.py` writes the **first** launch's record (`hpc_sweep.json` + a copy of the
sweep in `--output-dir`, default `results/hpc_<launch_id>/`; dispatcher #176) — copy it
into the variant's `launch_record/` like a Beaker record. Every post-sweep action
(resubmitting dead array tasks, rescuing a wedged job, a diagnostic rerun, recovering a
metric post-hoc) must still be recorded by hand with
`studies/util/launch_record.py::write_intervention(..., platform="hpc")` or it is lost. **Schema, required fields, and the full example live in the
`study-conventions` skill** — it owns the contract; don't re-derive it here.

Two HPC-specific points:

- **Record the W&B sweep id, not just the SLURM id** (`job_refs=[{"type": "wandb_sweep",
  "id": sweep_id}]`). SLURM ids are recycled and drop out of `sacct` within days; the sweep
  id is the only handle that still resolves later. Add the array id alongside if you like,
  never instead.
- `platform="hpc"` — the validator reconciles on `wandb_group`, so that field is what ties
  the record back to the runs.

## Monitoring

```bash
squeue -u $USER
scontrol show job <jobid>
sacct -j <array_jobid> --format=JobID,State,ExitCode,Start,End
tail -f $HOME/logfile/job_<jobid>.out
scancel <jobid>
wandb sweep --stop <SWEEP_ID>   # only needed after --no-autostop
```

When reporting results, link the W&B sweep/run URL and stamp times in Seattle time
(`TZ=America/Los_Angeles date`) — AGENTS.md §7.
