# Changelog

## 2026-10-23

- Created the study as the 3-way (L/R/ignore) counterpart of Study 12. Surveyed every
  `ignore_policy=include` run in the `AIND-disRNN` entity: the only 3-way GRUs are Study 02's
  `nxd-3way` grid (`mice_ignore_scaling`); the other hits are early-2026 disRNN test runs.
- Pinned C3 = `nxd-3way` D=614 / H128, 3 seeds (`make freeze-sources`), chosen as the
  config twin of Study 12's C (see README for the field-by-field check).
- Wrapper `gru_trial_states` generalised to 3-action heads (wrapper commit `8682fe7` on
  branch `study/13-gru-dynamics-embedding-ignore`, also applied to the shared checkout).
  Regression: re-extracting Study 12's C-s0 reproduces its stored table (same rows,
  columns, dtypes; floats within GPU-vs-CPU error, likelihood within 5e-7).
- A1 extraction for C3 seed 0 on CPU: 4,761,751 eval trials, all three logged likelihoods
  reproduced (diffs ≤ 4e-7), readout error 2e-6. The first attempt was OOM-killed while a
  second 26 GB job ran; the rerun alone peaked at 31 GB, so `extract_trial_states.py` now
  refuses to start below 40 GB available. The JSON's `wrapper_checkout.path` is the
  worktree the run used; its `head` (`8682fe7`) is the commit to check out.
