#!/bin/bash
#SBATCH -J study09_q_expand
#SBATCH -p aind
#SBATCH -t 24:00:00
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --array=0-9%4
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source "${CONDA_SH:?CONDA_SH must point to conda.sh}"
conda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu

DISPATCHER_ROOT="${DISPATCHER_ROOT:?DISPATCHER_ROOT is required}"
WRAPPER_ROOT="${WRAPPER_ROOT:?WRAPPER_ROOT is required}"
MODELS_ROOT="${MODELS_ROOT:?MODELS_ROOT is required}"
DATA_ROOT="${EXTERNAL_DATA_ROOT:?EXTERNAL_DATA_ROOT is required}"
RUN_ROOT="${BFM_OUTPUT_DIR:?BFM_OUTPUT_DIR is required}/study09-q-expanded-${SLURM_ARRAY_JOB_ID}"
DATASETS=(lebedeva beron kwak miller findling tang alsio eckstein costa lopez_mouse)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-q-expanded-matched-half@slurm-${SLURM_ARRAY_JOB_ID}}"
export BFM_META_VARIANT="q-expanded-matched-half"
export PYTHONPATH="${MODELS_ROOT}/src:${DISPATCHER_ROOT}/code:${WRAPPER_ROOT}/code:${PYTHONPATH:-}"
export BFM_WRAPPER_CODE="${WRAPPER_ROOT}/code"
export DISPATCHER_COMMIT
export WRAPPER_COMMIT
export FORAGING_MODELS_COMMIT
DISPATCHER_COMMIT="$(git -C "${DISPATCHER_ROOT}" rev-parse HEAD)"
WRAPPER_COMMIT="$(git -C "${WRAPPER_ROOT}" rev-parse HEAD)"
FORAGING_MODELS_COMMIT="$(git -C "${MODELS_ROOT}" rev-parse HEAD)"

python "${DISPATCHER_ROOT}/studies/09-gru-cross-species-transfer/run_q_baseline.py" \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${RUN_ROOT}" \
  --subject-workers 12
