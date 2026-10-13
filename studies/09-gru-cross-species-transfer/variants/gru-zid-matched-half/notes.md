# GRU Zid matched-half transfer

Fit only each new subject embedding on trials 0--149, replay the full session,
and score trials 150--299. There is no session-count K for this within-session
split. The frozen H=128 GRU core comes from the 15 Study 01 source cells
(`D={10,30,100,300,614}` and seeds 0--2).

GPU execution only: 15 one-GPU Beaker tasks. Source checkpoints and their W&B
artifact digests are pinned in `../../source_runs.json`.
