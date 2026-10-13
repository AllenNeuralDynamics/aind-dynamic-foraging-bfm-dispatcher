#!/bin/bash
#SBATCH -J study09_zid_models
#SBATCH -p aind
#SBATCH -t 12:00:00
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --mail-type=FAIL,END

set -euo pipefail

source "${CONDA_SH:?CONDA_SH must point to conda.sh}"
conda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu

DISPATCHER_ROOT="${DISPATCHER_ROOT:?DISPATCHER_ROOT is required}"
WRAPPER_ROOT="${WRAPPER_ROOT:?WRAPPER_ROOT is required}"
MODELS_ROOT="${MODELS_ROOT:?MODELS_ROOT is required}"
RUN_ROOT="${BFM_OUTPUT_DIR:?BFM_OUTPUT_DIR is required}/study09-zid-models-${SLURM_ARRAY_JOB_ID}"
DATA_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/study09-external"
BASELINES=(zid-traditional-rlck zid-history-kernel-foraging)
BASELINE="${BASELINES[$SLURM_ARRAY_TASK_ID]}"

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-zid-history-kernel@slurm-${SLURM_ARRAY_JOB_ID}}"
export BFM_META_VARIANT="zid-history-kernel"
export PYTHONPATH="${MODELS_ROOT}/src:${DISPATCHER_ROOT}/code:${WRAPPER_ROOT}/code:${PYTHONPATH:-}"
export BFM_WRAPPER_CODE="${WRAPPER_ROOT}/code"
export DISPATCHER_COMMIT
export WRAPPER_COMMIT
export FORAGING_MODELS_COMMIT
DISPATCHER_COMMIT="$(git -C "${DISPATCHER_ROOT}" rev-parse HEAD)"
WRAPPER_COMMIT="$(git -C "${WRAPPER_ROOT}" rev-parse HEAD)"
FORAGING_MODELS_COMMIT="$(git -C "${MODELS_ROOT}" rev-parse HEAD)"

python -m external_bandit_datasets.cli zid --cache-root "${DATA_ROOT}"
python "${DISPATCHER_ROOT}/studies/09-gru-cross-species-transfer/run_published_baseline.py" \
  --baseline "${BASELINE}" \
  --data-root "${DATA_ROOT}/canonical" \
  --output-root "${RUN_ROOT}" \
  --subject-workers 12
