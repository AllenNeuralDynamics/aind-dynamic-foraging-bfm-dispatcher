# Launch record — e8-d614-smoke

- **W&B project:** [`mice_data_scaling`](https://wandb.ai/AIND-disRNN/mice_data_scaling)
- **W&B group:** `e8-d614-smoke@20260906-192034`
- **W&B run:** [`e8-d614-smoke-20260906-192034-36d9a55f`](https://wandb.ai/AIND-disRNN/mice_data_scaling/runs/e8-d614-smoke-20260906-192034-36d9a55f)
- **Beaker experiment:** [`01M1WTNQSMRKSKCWD1Q6CXRPHT`](https://beaker.org/ex/01M1WTNQSMRKSKCWD1Q6CXRPHT)
- **Beaker result dataset:** `01M1WTNQSZQ0BHAPCZY1BTXVXC`
- **Runtime refs:** dispatcher `117d40fcf22b6df35375535c1904c468f332fc82`; wrapper `9595dd371ab87de49c281d8ca4bb6ae8af7c32e4`; foraging models `faa0f5ad063e375765aa9c31c7d3fee5eca78ecf`
- **Resources:** one on-prem H200 GPU; request 12 CPU and 90 GiB RAM
- **Status:** success — Beaker exit 0 and W&B `finished`

**Validation.** The full D=614 loader trained an H=128 GRU with an E=8 subject
embedding, saved a checkpoint/result artifact, adapted held-out subject embeddings,
and logged `heldout/final/eval_likelihood=0.6252979`.

**Feeds.** Infrastructure gate only. The 1,000-step likelihood is not a scientific
result and must not enter the E=4 versus E=8 comparison.
