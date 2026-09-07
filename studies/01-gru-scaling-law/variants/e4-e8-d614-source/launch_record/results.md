# Launch record — e4-e8-d614-source

- **W&B project:** [`mice_data_scaling`](https://wandb.ai/AIND-disRNN/mice_data_scaling)
- **W&B group:** `e4-e8-d614-source@20260906-195409`
- **Beaker experiment:** [`01M1WWK9HP269XQS440Y29631F`](https://beaker.org/ex/01M1WWK9HP269XQS440Y29631F)
- **Runtime refs:** dispatcher `ab399a50a736add59fcc189d0ac14e2dc6687279`; wrapper `9595dd371ab87de49c281d8ca4bb6ae8af7c32e4`; foraging models `faa0f5ad063e375765aa9c31c7d3fee5eca78ecf`
- **Resources:** six independent tasks on the on-prem H200 hub; one GPU, 12 CPU, and 90 GiB RAM requested per task
- **Status:** complete — 3 E=4 seeds and 3 E=8 seeds succeeded; all six W&B runs finished with immutable checkpoints, held-out per-subject tables, result datasets, and final train/eval likelihoods

**Headline result.** Mean held-out likelihood was 0.72891 for E=4 and 0.72920
for E=8 (+0.00029). The subject-paired E8−E4 difference across 149 held-out mice
had median +0.00016 (two-sided Wilcoxon p=0.000872); E8 PCs 5–8 carried 18.7%
of source-embedding covariance.

**Feeds.** Study 01 E=4 versus E=8 held-out AIND comparison and frozen source
artifacts for the gated Study 09 external-transfer screen.
