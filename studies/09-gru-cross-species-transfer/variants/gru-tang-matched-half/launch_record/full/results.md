# Launch record — full GRU matrix

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `gru-tang-matched-half@20260905-232924`
- **Beaker experiment:** [`01M1TPMY71W6PEGRPTRHMYQQDM`](https://beaker.org/ex/01M1TPMY71W6PEGRPTRHMYQQDM)
- **Cluster:** `ai1/octo-hub-aws-h200`
- **Grid:** `D={10,30,100,300,614}` × three source seeds (15 GPU tasks)
- **Pinned refs:** dispatcher `0008314`, wrapper `9595dd3`, models `459c05c`
- **Status:** running

**Headline.** Tang Stage-A embedding-only transfer is running on the complete admitted cohort. Only the external subject embedding is adapted for 500 steps at learning rate 0.001; the source GRU core is frozen.

**Feeds.** Results 2 and 3 after Beaker/W&B reconciliation and frozen-file validation
