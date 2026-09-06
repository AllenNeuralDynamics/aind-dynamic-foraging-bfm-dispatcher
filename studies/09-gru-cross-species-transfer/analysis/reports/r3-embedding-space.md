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
  - gru-kwak-matched-half@20260906-071413
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

![All transferred subjects in source-fitted PCA space](../fig_embedding_space_pca.png)

The primary comparison is **held-out AIND mice versus external subjects**. All of
these subjects were unseen during GRU-core training, initialized at the source
embedding mean, and adapted for the same 500 steps at learning rate 0.001 while
the core remained frozen. The 614 source-training mice define the coordinate
system but are not treated as the transfer control.

Kwak uses the corrected canonical choice orientation (`0=left, 1=right`),
converted from the release's `0=right, 1=left` encoding. Its embedding points
therefore come from the corrected D=614 reruns rather than the superseded
2026-09-05 launch.

PCA is fit independently to each seed's source-training mice; raw coordinates
are never pooled across seeds. The first three PCs explain
93.4%–98.0% of source variance. The
star is the common initialization point and the dashed ellipse is the Gaussian
95% source contour in each displayed projection.

![Full-dimensional distance from the source distribution](../fig_embedding_space_distance.png)

The Mahalanobis analysis uses all four embedding dimensions and each seed's
source covariance. External median distance exceeds the held-out-AIND median in
Grossman mouse (3/3 seeds); Chen mouse (3/3 seeds); Zid human (3/3 seeds); Lebedeva mouse (3/3 seeds); Beron mouse (3/3 seeds); Kwak mouse (3/3 seeds); Miller rat (3/3 seeds); Findling human (3/3 seeds); Tang macaque (3/3 seeds); Alsiö rat (3/3 seeds); Eckstein human (3/3 seeds); Costa macaque (3/3 seeds); López-Yépez mouse (3/3 seeds).

| seed | population | species | n | median distance | centroid distance | outside source 95% |
|---:|---|---|---:|---:|---:|---:|
| 0 | AIND held-out mice | mouse | 149 | 1.67 | 0.07 | 5.4% |
| 0 | Grossman mouse | mouse | 48 | 2.56 | 0.61 | 14.6% |
| 0 | Chen mouse | mouse | 32 | 10.51 | 8.76 | 100.0% |
| 0 | Zid human | human | 258 | 11.38 | 7.84 | 97.3% |
| 0 | Lebedeva mouse | mouse | 10 | 2.54 | 1.88 | 0.0% |
| 0 | Beron mouse | mouse | 6 | 2.82 | 2.64 | 0.0% |
| 0 | Kwak mouse | mouse | 39 | 4.33 | 4.80 | 66.7% |
| 0 | Miller rat | rat | 20 | 6.62 | 6.83 | 95.0% |
| 0 | Findling human | human | 22 | 10.04 | 9.72 | 100.0% |
| 0 | Tang macaque | macaque | 2 | 2.59 | 2.57 | 0.0% |
| 0 | Alsiö rat | rat | 95 | 10.09 | 9.42 | 100.0% |
| 0 | Eckstein human | human | 306 | 12.64 | 12.16 | 100.0% |
| 0 | Costa macaque | macaque | 11 | 5.58 | 6.08 | 100.0% |
| 0 | López-Yépez mouse | mouse | 8 | 17.24 | 15.34 | 87.5% |
| 1 | AIND held-out mice | mouse | 149 | 1.67 | 0.14 | 5.4% |
| 1 | Grossman mouse | mouse | 48 | 2.45 | 0.60 | 22.9% |
| 1 | Chen mouse | mouse | 32 | 6.56 | 5.54 | 100.0% |
| 1 | Zid human | human | 258 | 9.27 | 5.68 | 97.3% |
| 1 | Lebedeva mouse | mouse | 10 | 2.41 | 1.69 | 0.0% |
| 1 | Beron mouse | mouse | 6 | 2.50 | 2.36 | 16.7% |
| 1 | Kwak mouse | mouse | 39 | 3.40 | 3.02 | 53.8% |
| 1 | Miller rat | rat | 20 | 4.36 | 4.28 | 75.0% |
| 1 | Findling human | human | 22 | 6.15 | 5.60 | 100.0% |
| 1 | Tang macaque | macaque | 2 | 2.35 | 2.33 | 0.0% |
| 1 | Alsiö rat | rat | 95 | 7.43 | 6.53 | 100.0% |
| 1 | Eckstein human | human | 306 | 9.23 | 8.22 | 100.0% |
| 1 | Costa macaque | macaque | 11 | 6.51 | 6.59 | 100.0% |
| 1 | López-Yépez mouse | mouse | 8 | 10.33 | 9.30 | 87.5% |
| 2 | AIND held-out mice | mouse | 149 | 1.62 | 0.09 | 4.0% |
| 2 | Grossman mouse | mouse | 48 | 2.31 | 0.53 | 20.8% |
| 2 | Chen mouse | mouse | 32 | 8.45 | 7.07 | 100.0% |
| 2 | Zid human | human | 258 | 10.74 | 7.75 | 98.1% |
| 2 | Lebedeva mouse | mouse | 10 | 3.39 | 2.90 | 50.0% |
| 2 | Beron mouse | mouse | 6 | 2.59 | 2.46 | 0.0% |
| 2 | Kwak mouse | mouse | 39 | 4.07 | 4.04 | 69.2% |
| 2 | Miller rat | rat | 20 | 6.05 | 6.61 | 95.0% |
| 2 | Findling human | human | 22 | 8.00 | 8.29 | 100.0% |
| 2 | Tang macaque | macaque | 2 | 3.61 | 3.54 | 50.0% |
| 2 | Alsiö rat | rat | 95 | 9.12 | 8.28 | 100.0% |
| 2 | Eckstein human | human | 306 | 10.72 | 10.27 | 100.0% |
| 2 | Costa macaque | macaque | 11 | 10.44 | 10.19 | 100.0% |
| 2 | López-Yépez mouse | mouse | 8 | 10.06 | 8.97 | 87.5% |

## Cross-cohort read

Average median distance across the three independently trained spaces, nearest
to farthest, is:

1. **Grossman mouse** — 2.44
2. **Beron mouse** — 2.64
3. **Lebedeva mouse** — 2.78
4. **Tang macaque** — 2.85
5. **Kwak mouse** — 3.93
6. **Miller rat** — 5.67
7. **Costa macaque** — 7.51
8. **Findling human** — 8.06
9. **Chen mouse** — 8.51
10. **Alsiö rat** — 8.88
11. **Zid human** — 10.46
12. **Eckstein human** — 10.87
13. **López-Yépez mouse** — 12.54

This ordering is descriptive. Species, task schedule, reward contingencies,
recording duration, and adaptation-data volume change together across these
datasets, so distance cannot be interpreted as a pure species effect. Proximity
means the frozen core can express a target near the coordinates used by new
in-distribution mice; distance does not by itself imply poor prediction.

## Reproduce

The committed JSON contains every 4D embedding plus SHA-256 digests of the
downloaded embedding tables, subject maps, and adaptation summaries. Regenerate
both figures and this report offline with:

```bash
make r3
```
<!-- END result-3 -->
