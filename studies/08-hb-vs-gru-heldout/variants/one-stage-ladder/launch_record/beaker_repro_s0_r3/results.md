# beaker_repro_s0_r3 launch

- W&B group: `one-stage-ladder@20260902-235410`
- W&B project: https://wandb.ai/AIND-disRNN/test (smoke project, not the study's)
- Beaker experiment: https://beaker.org/ex/01M1K0QZ6RCDEQTZNMTTZMWH0R (`ai1/octo-hub-onprem-h200`, normal priority, 1 GPU / 128 GiB per task)
- Submitted: 2026-09-02 23:54 PT
- Settled: 2026-09-03 00:24 PT
- Status: success (1/1)
- Key numbers: matched held-out likelihood 0.7090835670131135 (minimal smoke sampler). All three `beaker_repro_s0_r*` launches give this value exactly, so a fixed seed reproduces the fit.
- Feeds reports: none (smoke-sampler path check)
- Notes: one of three identical launches at seed 0. The value was read from W&B on 2026-10-23 and is not frozen in the study's inputs.
