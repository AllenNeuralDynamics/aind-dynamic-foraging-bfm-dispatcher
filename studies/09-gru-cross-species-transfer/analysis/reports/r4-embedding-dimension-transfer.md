---
id: r4
slug: embedding-dimension-transfer
status: live
authors: [han, codex]
wandb_groups:
  - gru-e8-d614-diagnostic@20260907-093646
  - gru-e8-d614-diagnostic@20260907-071400
  - gru-e8-d614-expansion@20260907-132415
  - gru-e8-d614-expansion@20260907-132625
inputs:
  script: analysis/report_embedding_dimension.py
  freezers:
    - analysis/freeze_embedding_dimension.py
    - analysis/freeze_embedding_dimension_expansion.py
  data:
    - analysis/embedding_dimension_results.json
    - analysis/embedding_dimension_expansion_results.json
    - analysis/matched_half_results.json
    - source_runs_e4_pair.json
    - source_runs_e8.json
  figures:
    - analysis/fig_embedding_dimension_transfer.png
    - analysis/fig_subject_embedding_dimension_delta.png
reproduce: make -C studies/09-gru-cross-species-transfer r4
---

# Result 4 — does a wider subject embedding improve external transfer?

This report compares E=4 and E=8 source GRUs at D=614 and H=128 across all 12
valid external cohorts. Both dimensions use the same three source seeds, AIND
source snapshot, training recipe, external adaptation observations, 500-step
adaptation at learning rate 0.001, and immutable held-out trials. Five cohorts
have strict paired current-code E4/E8 reruns; the remaining seven compare the
current E8 result with the frozen historical E4 screen. The source core stays
frozen; only each external subject embedding is adapted.

<!-- BEGIN result-4 -->
![Paired seed-level E=4 and E=8 transfer](../fig_embedding_dimension_transfer.png)

Each thin line joins the same source seed. The black diamond-line is the three-seed mean.

![Subject-paired E=8 minus E=4 likelihood](../fig_subject_embedding_dimension_delta.png)

Dots are subjects; the short bar is the median and the hollow diamond is the mean. P-values are two-sided paired Wilcoxon signed-rank tests against zero.

| cohort | comparison basis | subjects | held-out trials | E=4 mean ± SD | E=8 mean ± SD | subject median Δ | subject mean Δ | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Grossman (mouse) | paired current-code E4/E8 | 48 | 101,877 | 0.74538 ± 0.00018 | 0.74521 ± 0.00024 | +0.00013 | -0.00050 | 0.726 |
| Chen (mouse) | historical E4 versus current E8 | 32 | 35,644 | 0.59275 ± 0.00140 | 0.59514 ± 0.00162 | +0.00273 | +0.00208 | 0.0184 |
| Zid (human) | historical E4 versus current E8 | 258 | 38,700 | 0.71363 ± 0.00192 | 0.70997 ± 0.00266 | +0.00019 | -0.00308 | 0.644 |
| Lebedeva (mouse) | paired current-code E4/E8 | 10 | 63,993 | 0.76631 ± 0.00045 | 0.76670 ± 0.00019 | +0.00040 | +0.00036 | 0.16 |
| Beron (mouse) | historical E4 versus current E8 | 6 | 188,926 | 0.82296 ± 0.00056 | 0.82574 ± 0.00096 | +0.00314 | +0.00279 | 0.0312 |
| Miller (rat) | paired current-code E4/E8 | 20 | 515,238 | 0.60007 ± 0.00291 | 0.60371 ± 0.00251 | +0.00358 | +0.00415 | 1.91e-06 |
| Findling (human) | paired current-code E4/E8 | 22 | 11,706 | 0.65453 ± 0.00441 | 0.66505 ± 0.00210 | +0.01222 | +0.01089 | 2.38e-06 |
| Eckstein (human) | paired current-code E4/E8 | 306 | 20,248 | 0.62554 ± 0.00908 | 0.63093 ± 0.00288 | +0.00720 | +0.00461 | 1.32e-05 |
| Alsiö (rat) | historical E4 versus current E8 | 95 | 225,751 | 0.52710 ± 0.00098 | 0.52843 ± 0.00057 | +0.00112 | +0.00136 | 7.97e-16 |
| Costa (macaque) | historical E4 versus current E8 | 11 | 162,960 | 0.58820 ± 0.00291 | 0.60040 ± 0.00146 | +0.00763 | +0.01234 | 0.000977 |
| López-Yépez (mouse) | historical E4 versus current E8 | 8 | 77,662 | 0.57554 ± 0.01322 | 0.59038 ± 0.00247 | +0.01275 | +0.01440 | 0.00781 |
| Tang (macaque) | historical E4 versus current E8 | 2 | 7,728 | 0.55416 ± 0.00036 | 0.55711 ± 0.00194 | +0.00295 | +0.00295 | 0.5 |

E8 has higher mean held-out likelihood in 10/12 cohorts (Chen (mouse), Lebedeva (mouse), Beron (mouse), Miller (rat), Findling (human), Eckstein (human), Alsiö (rat), Costa (macaque), López-Yépez (mouse), Tang (macaque)) and lower mean likelihood in 2/12 (Grossman (mouse), Zid (human)). Subject-level effect sizes and Wilcoxon tests are reported above; direction alone is not treated as evidence.

The strict current-code comparison covers Grossman (mouse), Lebedeva (mouse), Miller (rat), Findling (human), Eckstein (human). For Chen (mouse), Zid (human), Beron (mouse), Alsiö (rat), Costa (macaque), López-Yépez (mouse), Tang (macaque), E8 is compared with the frozen historical E4 runs on exactly the same held-out trial keys. Those historical E4 runs were previously shown to agree with the current-code E4 reruns to negligible numerical tolerance, but this remains a weaker comparison than a paired rerun.
<!-- END result-4 -->
