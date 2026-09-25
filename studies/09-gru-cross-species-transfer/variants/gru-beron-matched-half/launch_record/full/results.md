# Launch record — full GRU matrix

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `gru-beron-matched-half@20260905-232924`
- **Beaker experiment:** [`01M1TPMFZM1GYD60REY6JHTQSR`](https://beaker.org/ex/01M1TPMFZM1GYD60REY6JHTQSR)
- **Cluster:** `ai1/octo-hub-onprem-h200`
- **Grid:** `D={10,30,100,300,614}` × three source seeds (15 GPU tasks)
- **Pinned refs:** dispatcher `0008314`, wrapper `9595dd3`, models `459c05c`
- **Status:** success (15/15 clean finished GPU runs; frozen artifact validation passed)

**Headline.** Beron Stage-A embedding-only transfer completed on the complete admitted cohort. Only the external subject embedding was adapted for 500 steps at learning rate 0.001; the source GRU core remained frozen.

**Feeds.** Results 2 and 3 after Beaker/W&B reconciliation and frozen-file validation
