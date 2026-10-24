# Allen HPC Launching

This directory owns the AI1 Allen HPC launch path for disRNN. The dispatcher
repo is the control plane for Code Ocean, Beaker, and Allen HPC; the wrapper
repo remains the training/runtime payload.

Expected local layout:

```bash
/path/to/parent/
  aind-dynamic-foraging-bfm-dispatcher/
  aind-dynamic-foraging-bfm-wrapper/
```

## Setup

Create the runtime environments from the wrapper repo:

```bash
cd /path/to/parent/aind-dynamic-foraging-bfm-wrapper

conda create -n dynamic-foraging-bfm-cpu python=3.12 -y
conda activate dynamic-foraging-bfm-cpu
pip install -e .

conda create -n dynamic-foraging-bfm-gpu python=3.12 -y
conda activate dynamic-foraging-bfm-gpu
pip install -e ".[gpu]"
```

The launcher is always invoked from the CPU environment. The SLURM script then
activates `dynamic-foraging-bfm-cpu` or `dynamic-foraging-bfm-gpu` on the compute node based on
`--mode cpu|gpu`.

Create the per-user SLURM env file in the dispatcher repo:

```bash
cd /path/to/parent/aind-dynamic-foraging-bfm-dispatcher
cp code/hpc/slurm/user.env.example code/hpc/slurm/user.env
# edit code/hpc/slurm/user.env
```

Optionally source it from your shell startup file so manual `sbatch` commands
pick up the same settings:

```bash
echo 'source /path/to/aind-dynamic-foraging-bfm-dispatcher/code/hpc/slurm/user.env' >> ~/.bashrc
```

`SBATCH_*` variables are read by `sbatch`. `CONDA_SH` is used by the SLURM
scripts to activate the runtime environment.

## W&B Sweep Launch

Run from the dispatcher repo root:

```bash
conda activate dynamic-foraging-bfm-cpu

python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/synthetic_num_sessions_disrnn.yaml \
  --mode cpu

python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/synthetic_num_sessions_disrnn.yaml \
  --mode gpu
```

Useful launch variants:

```bash
# Inspect planned commands without creating a sweep or submitting SLURM jobs.
python code/launch_hpc.py --mode gpu --dry-run

# End-to-end smoke test: one tiny run on one CPU array task.
python code/launch_hpc.py \
  --sweep-yaml code/hpc/sweeps/synthetic_disrnn_smoke.yaml \
  --mode cpu \
  --sbatch-extra=--array=0-0 \
  --agent-count 1

# Bounded validation of the synthetic num-sessions sweep: one grid point.
python code/launch_hpc.py --mode cpu --sbatch-extra=--array=0-0 --agent-count 1

# Request a specific GPU tier.
python code/launch_hpc.py --mode gpu --gpu-type a100

# Use a non-sibling wrapper checkout.
python code/launch_hpc.py --mode gpu --wrapper-root /path/to/aind-dynamic-foraging-bfm-wrapper
```

`code/launch_hpc.py` creates the W&B sweep, injects dispatcher, wrapper and studies
git lineage into every run as Hydra `+meta.*` overrides, computes `AGENT_COUNT` for
grid sweeps, submits the SLURM array, and submits a small cleanup job that marks
the W&B sweep finished after the array drains.

### Launch record and the dirty-checkout rule

Every launch writes `hpc_sweep.json` plus a copy of the sweep YAML to `--output-dir`
(default `results/hpc_<launch_id>/` under the dispatcher root). It carries the W&B
sweep id (the durable handle), the SLURM array and auto-stop job ids, and the two-root
provenance block from `code/provenance.py`: `studies_git_commit` (the checkout that
held the sweep -- the studies repo after the split, dispatcher #124), `dispatcher_git_commit`
(the checkout that ran this launcher), `wrapper_git_commit`, and the branch / dirty flags.
Copy the directory into the variant's `launch_record/` like a Beaker record;
`validate_provenance.py --wandb` reconciles its sweep id against W&B.

A sweep whose `studies/<study>/` directory has uncommitted changes is **refused**: no
commit could reproduce it. Commit first, or for a diagnostic run pass
`--diagnostic-dirty-ok "<reason>"`; the reason and the dirty paths are written into the
record. `--dry-run` prints the provenance block without creating anything.

## Manual SLURM Use

Manual use skips lineage injection and `AGENT_COUNT` auto-computation:

```bash
wandb sweep code/hpc/sweeps/synthetic_num_sessions_disrnn.yaml
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper \
  code/hpc/slurm/wandb_sweep_gpu.slurm <SWEEP_ID>
```

Hydra multirun scripts are also available when you want deterministic config
enumeration without W&B sweep orchestration:

```bash
sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper \
  code/hpc/slurm/hydra_multirun_cpu.slurm

sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper \
  code/hpc/slurm/hydra_multirun_gpu.slurm
```

## Monitoring

```bash
squeue -u $USER
scontrol show job <jobid>
sacct -j <array_jobid> --format=JobID,State,ExitCode,Start,End
tail -f $HOME/logfile/job_<jobid>.out
scancel <jobid>
```

If you launched with `--no-autostop`, stop the W&B sweep manually after the
agents finish:

```bash
wandb sweep --stop <SWEEP_ID>
```

## GPU Tier

The GPU SLURM scripts default to `--gres=gpu:v100:1` and `--array=0-5`, matching
the six-GPU per-user cap on the `aind` QOS. Override the tier per launch with
`--gpu-type` for W&B sweeps or `sbatch --gres=gpu:<type>:1` for manual Hydra
multiruns.
