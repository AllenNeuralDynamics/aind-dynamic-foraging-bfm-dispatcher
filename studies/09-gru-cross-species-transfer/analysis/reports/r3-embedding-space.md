---
title: "Result 3: transferred subject embedding space"
status: live
issue: 126
data:
  - ../embedding_space_results.json
figures:
  - ../fig_embedding_space_pca.png
  - ../fig_embedding_space_distance.png
wandb_groups:
  - v2-sc-active@20260622-144622
  - gru-grossman-matched-half@20260905-022602
  - gru-chen-matched-half@20260905-024731
  - gru-zid-matched-half@20260905-025752
  - gru-lebedeva-matched-half@20260905-232924
  - gru-beron-matched-half@20260905-232924
  - gru-kwak-matched-half@20260905-232924
  - gru-miller-matched-half@20260905-232924
  - gru-findling-matched-half@20260905-232924
  - gru-tang-matched-half@20260905-232924
  - gru-alsio-matched-half@20260905-232925
  - gru-eckstein-matched-half@20260905-232924
  - gru-costa-matched-half@20260905-232924
  - gru-lopez-mouse-matched-half@20260905-232924
reproduce: make r3
---

# Result 3: where transferred subjects land in the GRU embedding space

This result asks where unseen held-out AIND mice and every admitted external
two-arm-bandit cohort land in the subject-embedding manifold learned from the
614 source mice.

<!-- BEGIN result-3 -->
## Result

![Subject embeddings in source-fitted PCA space](../fig_embedding_space_pca.png)

The comparison is deliberately anchored on **held-out AIND mice**, not on the source-training mice. The 149 held-out AIND mice and all external subjects were unseen while the GRU core was trained; each entered at the source-embedding mean and received the same 500-step, learning-rate-0.001 embedding-only adaptation. The 614 source-training mice define the coordinate system and reference distribution.

Each seed has its own independently learned embedding coordinates, so PCA was fit on that seed's source-training mice and no raw coordinates were pooled across seeds. The first three PCs contain 93.4%--98.0% of source variance across seeds. The star is the source mean and therefore the initialization point for every adapted subject; the dashed ellipse is the Gaussian 95% contour of the source distribution in each displayed 2D projection.

![Full-dimensional distance from the source distribution](../fig_embedding_space_distance.png)

The second figure checks the same question in the full four-dimensional space rather than relying on a 2D projection. Distances use each seed's source mean and covariance; the dashed line is that seed's empirical source 95th percentile. External median distance exceeds the held-out-AIND median for Grossman mouse (3/3 seeds), Chen mouse (3/3 seeds), Zid human (3/3 seeds). This is descriptive evidence of how far each transferred cohort must move in the learned subject space, not a test of a pure species effect.

| seed | population | n | median distance | centroid distance | outside source 95% |
|---:|---|---:|---:|---:|---:|
| 0 | AIND held-out mice | 149 | 1.67 | 0.07 | 5.4% |
| 0 | Grossman mouse | 48 | 2.56 | 0.61 | 14.6% |
| 0 | Chen mouse | 32 | 10.51 | 8.76 | 100.0% |
| 0 | Zid human | 258 | 11.38 | 7.84 | 97.3% |
| 1 | AIND held-out mice | 149 | 1.67 | 0.14 | 5.4% |
| 1 | Grossman mouse | 48 | 2.45 | 0.60 | 22.9% |
| 1 | Chen mouse | 32 | 6.56 | 5.54 | 100.0% |
| 1 | Zid human | 258 | 9.27 | 5.68 | 97.3% |
| 2 | AIND held-out mice | 149 | 1.62 | 0.09 | 4.0% |
| 2 | Grossman mouse | 48 | 2.31 | 0.53 | 20.8% |
| 2 | Chen mouse | 32 | 8.45 | 7.07 | 100.0% |
| 2 | Zid human | 258 | 10.74 | 7.75 | 98.1% |

## Scientific interpretation

- **Matched internal control:** held-out AIND mice are the clean reference for transfer because, like external subjects, they were absent from source-core training and only their embeddings were adapted.
- **Main result:** held-out AIND is calibrated to the source distribution (4.0%--5.4% outside the source 95th percentile), Grossman is moderately shifted (14.6%--22.9%), and Chen (100%) and Zid (97.3%--98.1%) are strongly displaced in every seed.
- **Task is a better first explanation than species:** Chen mice and Zid humans share the restless random-walk task and both move far from the AIND manifold, while Grossman mice perform a blockwise task closer to AIND dynamic foraging and remain much nearer. This repeated cross-seed geometry argues against reading the Zid separation as simply mouse versus human.
- **What proximity means:** overlap with held-out AIND says the frozen core can represent the target behavior using subject coordinates similar to those used for new in-distribution mice. Larger distance says adaptation found a more out-of-distribution coordinate; it does not by itself mean worse prediction.
- **What this cannot identify:** dataset, task schedule, species, recording duration, and adaptation-data volume change together. Consequently, external separation cannot be assigned to species alone. Grossman and Chen are especially useful mouse controls for judging whether task structure, rather than species, drives the displacement.
- **Seed discipline:** agreement of the qualitative ordering across independently trained spaces is stronger evidence than any absolute PC direction. PC axes and embedding coordinates have no cross-seed identity.

## Reproduce

The committed JSON contains every 4D embedding plus SHA-256 digests of the downloaded tables, subject maps, and adaptation summaries. Regenerate both figures and this report offline with:

```bash
make r3
```
<!-- END result-3 -->
