#!/bin/bash
# SLURM array script that runs a W&B sweep agent (CPU) for an existing sweep.
#
# Prefer launching via code/launch_hpc.py, which creates the sweep,
# injects git/launch lineage into each run's config (as Hydra +meta.* overrides),
# computes AGENT_COUNT from the grid size, and submits this script for you:
#
#     python code/launch_hpc.py --mode cpu
#
# Manual use (skips lineage injection and auto AGENT_COUNT):
#     wandb sweep code/hpc/sweeps/synthetic_num_sessions_disrnn.yaml
#     sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper code/hpc/slurm/wandb_sweep_cpu.slurm <SWEEP_ID>
#
#SBATCH -J disrnn_wb_cpu
#SBATCH -p aind
#SBATCH -t 05:00:00
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --mail-type=FAIL,END,BEGIN

# Per-user settings (mail-user, output/error paths, CONDA_SH) come from
# code/hpc/slurm/user.env; source it before sbatch. See code/hpc/README.md "Per-user setup".
source "${CONDA_SH:?CONDA_SH not set; source code/hpc/slurm/user.env before sbatch}"
# Prefer the post-rename env (ADR-0007); fall back to the legacy name so a
# machine that has not created the new env yet still runs. Old envs stay in
# place until nothing references them.
conda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu

# AI1 compute nodes carry gcc but NOT g++, while the login node has both -- so anything that
# compiles C++ inside a job fails only once it lands on a node, and a compiler probe run on
# the login node reports the wrong answer. pystan/httpstan builds a C++ extension per model,
# which is what hits this. Harmless for every other CPU job: it only puts a compiler on PATH.
# Note `module load gcc` alone does not resolve; the version suffix is required.
if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
    module load gcc/14-rocky9 2>/dev/null || true
    command -v g++ >/dev/null && export CXX="$(command -v g++)" CC="$(command -v gcc)"
fi

WRAPPER_ROOT="${WRAPPER_ROOT:?WRAPPER_ROOT not set; launch via code/launch_hpc.py or pass --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper}"
cd "$WRAPPER_ROOT/code"
export PYTHONPATH="$WRAPPER_ROOT/code:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export BFM_OUTPUT_DIR="${BFM_OUTPUT_DIR:-$HOME/outputs/disrnn}"

SWEEP_ID="$1"
if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch --export=ALL,WRAPPER_ROOT=/path/to/aind-dynamic-foraging-bfm-wrapper code/hpc/slurm/wandb_sweep_cpu.slurm <SWEEP_ID>" 1>&2
    exit 1
fi

AGENT_COUNT="${AGENT_COUNT:-3}"
echo "[CPU] SWEEP_ID=$SWEEP_ID AGENT_COUNT=$AGENT_COUNT ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
wandb agent --count "$AGENT_COUNT" "$SWEEP_ID"
