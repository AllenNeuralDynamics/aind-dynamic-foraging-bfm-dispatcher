# q-expanded-matched-half launch

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `q-expanded-matched-half@20260906-001656`
- **SLURM array:** `25581304` (`aind`, CPU-only, 24 CPU / 64 GiB per task, 4 concurrent)
- **Submitted:** 2026-09-06 00:16 PT
- **Pinned refs:** dispatcher `230fee7`, wrapper `9595dd3`, models `459c05c`
- **Status:** running

**Headline.** The common subject-level Q model is fitting the ten newly admitted cohorts on the same complete adaptation halves used by GRU and will score the identical immutable held-out trials.

**Feeds.** Results 2 after W&B artifact and exact trial-key reconciliation

**Notes.** The canonical data directory was staged to HPC scratch and verified with an rsync checksum dry run. This CPU-only work was not submitted to Beaker.
