#!/bin/bash
#SBATCH -J study09_costa_fdrl
#SBATCH -p aind
#SBATCH -t 24:00:00
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source "${CONDA_SH:?CONDA_SH must point to conda.sh}"
conda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu

DISPATCHER_ROOT="${DISPATCHER_ROOT:?DISPATCHER_ROOT is required}"
WRAPPER_ROOT="${WRAPPER_ROOT:?WRAPPER_ROOT is required}"
MODELS_ROOT="${MODELS_ROOT:?MODELS_ROOT is required}"
RUN_ROOT="${BFM_OUTPUT_DIR:?BFM_OUTPUT_DIR is required}/study09-costa-feedback-dependent-${SLURM_JOB_ID}"
DATA_ROOT="${EXTERNAL_DATA_ROOT:?EXTERNAL_DATA_ROOT is required}"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-costa-feedback-dependent@slurm-${SLURM_JOB_ID}}"
export BFM_META_VARIANT="costa-feedback-dependent"
export PYTHONPATH="${MODELS_ROOT}/src:${DISPATCHER_ROOT}/code:${WRAPPER_ROOT}/code:${PYTHONPATH:-}"
export BFM_WRAPPER_CODE="${WRAPPER_ROOT}/code"
export DISPATCHER_COMMIT
export WRAPPER_COMMIT
export FORAGING_MODELS_COMMIT
DISPATCHER_COMMIT="$(git -C "${DISPATCHER_ROOT}" rev-parse HEAD)"
WRAPPER_COMMIT="$(git -C "${WRAPPER_ROOT}" rev-parse HEAD)"
FORAGING_MODELS_COMMIT="$(git -C "${MODELS_ROOT}" rev-parse HEAD)"

python "${DISPATCHER_ROOT}/studies/09-gru-cross-species-transfer/run_costa_author_baseline.py" \
  --data-root "${DATA_ROOT}" \
  --output-root "${RUN_ROOT}" \
  --subject-workers 11
