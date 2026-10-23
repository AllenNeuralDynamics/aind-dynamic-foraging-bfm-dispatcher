# Changelog

## 2026-10-23

- Created the study as the 3-way (L/R/ignore) counterpart of Study 12. Surveyed every
  `ignore_policy=include` run in the `AIND-disRNN` entity: the only 3-way GRUs are Study 02's
  `nxd-3way` grid (`mice_ignore_scaling`); the other hits are early-2026 disRNN test runs.
- Pinned C3 = `nxd-3way` D=614 / H128, 3 seeds (`make freeze-sources`), chosen as the
  config twin of Study 12's C (see README for the field-by-field check).
- Wrapper `gru_trial_states` generalised to 3-action heads (branch
  `study/13-gru-dynamics-embedding-ignore`); 2-way output unchanged.
- A1 extraction for C3 seed 0 on CPU.
