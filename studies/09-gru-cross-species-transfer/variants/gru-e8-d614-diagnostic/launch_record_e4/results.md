# Launch record — paired E=4 diagnostic transfer

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `gru-e8-d614-diagnostic@20260907-093646`
- **Beaker experiment:** [`01M1YBQERTCR0HF6Y749X7M5HT`](https://beaker.org/ex/01M1YBQERTCR0HF6Y749X7M5HT)
- **Clusters:** `ai1/octo-hub-onprem-h200`, `ai1/octo-hub-aws-h200`
- **Grid:** five diagnostic cohorts × three paired E=4 source seeds (15 GPU tasks)
- **Pinned refs:** dispatcher `8746e60`, wrapper `9595dd3`, models `faa0f5a`
- **Status:** running

Only the external subject embedding is adapted for 500 steps at learning rate
0.001. The E=4 source core is frozen, and each target uses the same v1/v2 split
manifest and held-out trials as the completed E=8 screen.

The resumable launcher rendered and pinned the submitted spec. Its SDK submission
path is unavailable without `BEAKER_TOKEN` in this local session, so the exact
rendered spec was submitted through the authenticated Beaker CLI; the rendered
YAML is retained here.
