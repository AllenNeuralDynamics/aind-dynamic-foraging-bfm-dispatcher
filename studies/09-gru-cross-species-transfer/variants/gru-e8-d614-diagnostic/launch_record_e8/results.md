# Launch record — E=8 diagnostic transfer

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `gru-e8-d614-diagnostic@20260907-071400`
- **Beaker experiment:** [`01M1Y3NHNHGZBXYYZQSW6JD9T2`](https://beaker.org/ex/01M1Y3NHNHGZBXYYZQSW6JD9T2)
- **Clusters:** `ai1/octo-hub-onprem-h200`, `ai1/octo-hub-aws-h200`
- **Grid:** five diagnostic cohorts × three E=8 source seeds (15 GPU tasks)
- **Pinned refs:** dispatcher `37ac351`, wrapper `9595dd3`, models `faa0f5a`
- **Status:** success (15/15 Beaker tasks exit 0; 15/15 W&B runs finished
  with final train/eval likelihoods)

Only the external subject embedding is adapted for 500 steps at learning rate
0.001. The E=8 source core is frozen, and each target uses the existing v1/v2
split manifest and immutable held-out trials.

The resumable launcher rendered and pinned the submitted spec. Its SDK submission
path could not read `BEAKER_TOKEN` in this local session, so the exact rendered
spec was submitted through the authenticated Beaker CLI; the rendered YAML is
retained here.

**E=8 cohort means (held-out likelihood):** Grossman (mouse) `0.74521`,
Lebedeva (mouse) `0.76670`, Miller (rat) `0.60371`, Findling (human)
`0.66505`, and Eckstein (human) `0.63093`. These are not yet an E=8 versus
E=4 conclusion; the paired current-code E=4 transfer is still gated on its
remaining source seed.
