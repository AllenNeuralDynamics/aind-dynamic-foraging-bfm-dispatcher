# q-expanded-matched-half launch

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `q-expanded-matched-half@20260906-001656`
- **SLURM array:** `25581304` (`aind`, CPU-only, 24 CPU / 64 GiB per task, 4 concurrent)
- **Submitted:** 2026-09-06 00:16 PT
- **Settled:** 2026-09-06 04:26 PT
- **Pinned refs:** dispatcher `230fee7`, wrapper `9595dd3`, models `459c05c`
- **Status:** success (10/10)

**Headline.** All ten common subject-level Q fits completed on the same adaptation halves used by GRU. Exact ordered held-out trial-key parity passed for every one of the 150 new GRU cells and its cohort's Q prediction file. Miller, the final and largest fit, scored 0.609445 normalized likelihood across 515,238 held-out trials.

**Feeds.** Results 2 and 3

**Notes.** The canonical data directory was staged to HPC scratch and verified with an rsync checksum dry run. This CPU-only work was not submitted to Beaker. Frozen result artifacts and per-trial prediction digests are recorded in `analysis/matched_half_results.json`.
