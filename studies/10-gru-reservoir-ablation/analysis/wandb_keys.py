"""W&B input contract for Study 10.

Project: AIND-disRNN/gru_reservoir_ablation
Pinned exact-split D=614 runs and groups: ``freeze_results.RESERVOIR_RUNS`` and
``freeze_results.RESERVOIR_GROUP_BY_SEED``. Pinned scaling-curve cells:
``freeze_dcurve.CELLS``. The launcher-produced W&B config omits the top-level
Hydra seed for the exact rerun, so immutable submitted launch specs are the
seed-identity source of truth there.
Positive-control project: AIND-disRNN/mice_data_scaling
Pinned positive-control runs: ``freeze_results.SOURCE_RESULT_RUNS`` and the
D=614 source-model run IDs in ``reference/study01-trained-gru.json``.

Summary keys:
- heldout/final/eval_likelihood
- final/eval_likelihood
- training_steps_completed

Run-table artifact:
- heldout/per_subject_likelihood

Training-output files:
- initialization/before_training/params.json
- params.json
"""
