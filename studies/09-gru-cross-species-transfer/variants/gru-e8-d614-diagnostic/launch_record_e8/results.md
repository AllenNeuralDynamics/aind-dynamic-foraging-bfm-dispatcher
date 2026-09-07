# Launch record — E=8 diagnostic transfer

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `gru-e8-d614-diagnostic@20260907-071400`
- **Beaker experiment:** [`01M1Y3NHNHGZBXYYZQSW6JD9T2`](https://beaker.org/ex/01M1Y3NHNHGZBXYYZQSW6JD9T2)
- **Clusters:** `ai1/octo-hub-onprem-h200`, `ai1/octo-hub-aws-h200`
- **Grid:** five diagnostic cohorts × three E=8 source seeds (15 GPU tasks)
- **Pinned refs:** dispatcher `37ac351`, wrapper `9595dd3`, models `faa0f5a`
- **Status:** running (6/15 tasks started immediately; 9/15 queued at launch)

Only the external subject embedding is adapted for 500 steps at learning rate
0.001. The E=8 source core is frozen, and each target uses the existing v1/v2
split manifest and immutable held-out trials.

The resumable launcher rendered and pinned the submitted spec. Its SDK submission
path could not read `BEAKER_TOKEN` in this local session, so the exact rendered
spec was submitted through the authenticated Beaker CLI; the rendered YAML is
retained here.
