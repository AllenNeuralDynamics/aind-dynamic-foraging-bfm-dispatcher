# Launch record — Hattori (mouse) mature E4 full matrix

- **W&B project:** [`gru_cross_species_transfer`](https://wandb.ai/AIND-disRNN/gru_cross_species_transfer)
- **W&B group:** `gru-hattori-matched-half@20260907-200329`
- **Beaker experiment:** [`01M1ZFJ3BPJSZNJ403DPD3FS6C`](https://beaker.org/ex/01M1ZFJ3BPJSZNJ403DPD3FS6C)
- **Grid:** `D={10,30,100,300,614}` × source seeds `{0,1,2}` (15 GPU tasks)
- **Target cohort:** Hattori (mouse), each subject's session 15 onward
- **Status:** success (15/15 GPU tasks exited 0)

**Headline.** Three-seed mean normalized held-out likelihood increased from
`0.560258` at `D=10` to `0.566074` at `D=614`. The intermediate means were
`0.560620` (`D=30`), `0.563715` (`D=100`), and `0.565660` (`D=300`). Every
cell scored the identical 68,972 held-out trials from seven subjects.

**Feeds.** Mature Hattori E4 GRU-versus-D, subject-level comparisons, and
embedding-space analysis.
