# Zid published-model results

Both accepted tasks scored the same `38,700` held-out trials.

| model | W&B | normalized likelihood | mean log likelihood | Brier | accuracy |
|---|---|---:|---:|---:|---:|
| Traditional RLCK | [7qk36y7z](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/7qk36y7z) | 0.6932158704 | -0.3664138269 | 0.1041391450 | 0.8440826873 |
| Author-selected HK2 foraging RL | [7xcefdqr](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/7xcefdqr) | 0.6830631867 | -0.3811679102 | 0.1053370849 | 0.8441602067 |

The traditional RLCK result is task `25580838_0`; the corrected HK2 result is task
`25580951_1`, which completed in `00:11:54` with exit `0:0`. These runs feed
`analysis/author_baseline_results.json` and Result 2.

## Superseded HK2 runs

- [xvtn1ewo](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/xvtn1ewo)
  (`25580838_1`, normalized likelihood `0.6942748459`) is superseded. A direct check against
  the authors' `model_ForagingFlex.m` showed that the first exploitation value must start at
  `1`, the first observed choice is not a switch, and the first two MATLAB-indexed history
  kernel entries remain `0`.
- [xap7wvxu](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer/runs/xap7wvxu)
  (`25580950_1`) was cancelled after `00:03:01` when CI exposed the remaining one-trial
  history-kernel offset. It produced no accepted result.

The full replacement chain and code-SHA change are recorded in
`hpc_resubmit_after_author_code_parity.json`.
