#!/bin/bash
#SBATCH -J study09_tang_rl
#SBATCH -p aind
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source "${CONDA_SH:?CONDA_SH must point to conda.sh}"
conda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu

DISPATCHER_ROOT="${DISPATCHER_ROOT:?DISPATCHER_ROOT is required}"
WRAPPER_ROOT="${WRAPPER_ROOT:?WRAPPER_ROOT is required}"
MODELS_ROOT="${MODELS_ROOT:?MODELS_ROOT is required}"
RUN_ROOT="${BFM_OUTPUT_DIR:?BFM_OUTPUT_DIR is required}/study09-tang-block-type-rl-${SLURM_JOB_ID}"
DATA_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/study09-external"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-tang-block-type-rl@slurm-${SLURM_JOB_ID}}"
export BFM_META_VARIANT="tang-block-type-rl"
export PYTHONPATH="${MODELS_ROOT}/src:${DISPATCHER_ROOT}/code:${WRAPPER_ROOT}/code:${PYTHONPATH:-}"
export BFM_WRAPPER_CODE="${WRAPPER_ROOT}/code"
export DISPATCHER_COMMIT
export WRAPPER_COMMIT
export FORAGING_MODELS_COMMIT
DISPATCHER_COMMIT="$(git -C "${DISPATCHER_ROOT}" rev-parse HEAD)"
WRAPPER_COMMIT="$(git -C "${WRAPPER_ROOT}" rev-parse HEAD)"
FORAGING_MODELS_COMMIT="$(git -C "${MODELS_ROOT}" rev-parse HEAD)"

python -m external_bandit_datasets.cli tang --cache-root "${DATA_ROOT}"
python "${DISPATCHER_ROOT}/studies/09-gru-cross-species-transfer/run_tang_author_baseline.py" \
  --data-root "${DATA_ROOT}/canonical" \
  --output-root "${RUN_ROOT}" \
  --subject-workers 2
