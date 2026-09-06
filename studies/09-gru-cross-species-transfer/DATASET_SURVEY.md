# Study 09 dataset survey and Stage-A admission record

This is the frozen cohort-level admission record for the Study 09 expansion.
An admitted cohort must retain its complete analytic sample and satisfy exactly one
existing split contract:

- **v1:** every subject has at least two real sessions; chronological odd-positioned
  sessions adapt and even-positioned sessions test.
- **v2:** every subject has exactly one sufficiently long session; its complete first
  half adapts and its second half tests.

No schema v3, pseudo-session construction, or subject removal was used to make a
cohort fit. The machine-readable exact-count and digest audit is
[`analysis/dataset_suite_validation.json`](analysis/dataset_suite_validation.json).

## Admitted cohorts

| Cohort | Species and task | Source and terms | Subjects | Sessions | Valid trials | Held-out trials | Split compatibility | Author-selected model | Reproduction difficulty | Stage-A status |
|---|---|---|---:|---:|---:|---:|---|---|---|---|
| Grossman, Bari & Cohen | Mouse; blockwise independent probabilities | [Dryad v4](https://doi.org/10.5061/dryad.cz8w9gj4s), CC0 | 48 | 754 | 210,159 | 101,877 | v1 | uncertainty-dependent meta-learning RL | Moderate; already implemented | Completed; reuse frozen GRU/Q/author results |
| Chen et al. | Mouse; restless independent random walks | [Dryad v5](https://doi.org/10.5061/dryad.z612jm6c0), CC0 | 32 | 256 | 70,778 | 35,644 | v1 | four-parameter RL plus choice kernel (RLCK) | Low; already implemented | Completed; reuse frozen GRU/Q/author results |
| Zid et al. Experiment 1 | Human; schedule-matched restless random walks | [Figshare v5](https://doi.org/10.6084/m9.figshare.32193990.v5), MIT | 258 | 258 | 77,400 | 38,700 | v2 | history-kernel-2 foraging RL; RLCK comparator | Low; already implemented | Completed; reuse frozen GRU/Q/author results |
| Lebedeva et al. | Mouse; 80/20 probabilistic reversals | [Figshare v8](https://doi.org/10.6084/m9.figshare.31231741.v8), CC BY 4.0 | 10 | 254 | 132,494 | 63,993 | v1 | perseveration/reward-learning (PR) model; simplest of five tied top models | Moderate; equations are complete, runnable model code was not in the behavior archive | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Beron et al. | Mouse; nonstationary probabilistic two-arm bandit | [Harvard Dataverse v1.1](https://doi.org/10.7910/DVN/7E0NM5), CC0 | 6 | 525 | 378,351 | 188,926 | v1 | recursively formulated logistic regression (RFLR); equivalent RL/sticky-HMM views | Moderate; published equations and code links require a Stage-B audit | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Kwak & Jung | Mouse; dynamic two-arm bandit under D1/D2 manipulation | [Dryad](https://doi.org/10.5061/dryad.4c80mn5), CC0 | 39 | 780 | 121,100 | 60,045 | v1 | subject/condition-level RL with learning rate and value-dependent softmax choice | Low to moderate; simple equations, original fit code not found | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Miller, Botvinick & Brody | Rat; large dynamic two-arm bandit | [Figshare v2](https://doi.org/10.6084/m9.figshare.20449356.v2), CC BY 4.0 | 20 | 1,857 | 1,040,731 | 515,238 | v1 | three-timescale mixture: reward seeking, perseveration, and gambler's fallacy | Moderate to high; preprint specification, no maintained fit package found | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Findling et al. | Human; 85/15 reversals under varying volatility | [GitHub commit `ee68853`](https://github.com/csmfindling/Volnoise/tree/ee688535b569a8af8c0531350ec06f34cb989f8e), MIT | 22 | 132 | 23,275 | 11,706 | v1 | Weber-variability Bayesian inference model | High; hierarchical inference family despite available research code | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Tang, Bartolo & Averbeck | Macaque; blockwise action/object values | [Mendeley Data v1](https://doi.org/10.17632/m4f38w49fb.1), CC BY 4.0 | 2 | 8 | 15,375 | 7,728 | v1 | no author-selected behavioral prediction model identified | Not applicable unless a new scientific baseline is chosen | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Alsiö et al. cohorts II–V | Rat; visual/spatial discrimination and reversal | [Cambridge Apollo](https://doi.org/10.17863/CAM.80290), CC BY 4.0 | 95 | 2,334 | 457,007 | 225,751 | v1 | separate positive/negative learning-rate RL was selected for the separate PRL cohort | High for direct comparison: selected model and admitted cohort do not align | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Eckstein et al. | Human; stochastic reversal across development | [OSF `7wuh4`](https://osf.io/7wuh4/) (no explicit OSF data license) | 306 | 306 | 40,229 | 20,248 | v2 | co-winners: hierarchical RL and Bayesian-inference models | High; two hierarchical winners and paper reports an analytic n=291 | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Costa et al. | Macaque; stochastic stimulus reversal | [Zenodo record 20086410](https://doi.org/10.5281/zenodo.20086410), CC BY 4.0 | 11 | 245 | 329,840 | 162,960 | v1 | feedback-dependent RL with separate positive/negative learning rates | Moderate; equations are published, original fitting code not found | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| López-Yépez et al. mouse | Mouse; baited variable-interval dynamic matching | [Figshare v1](https://doi.org/10.6084/m9.figshare.14540283.v1), CC BY 4.0 | 8 | 218 | 147,726 | 77,662 | v1 | double-trace RL with fast and slow choice traces | Moderate; equations are published, MATLAB implementation was not released with the data | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |

The Eckstein release contains 306 valid one-session participant files, while the
paper reports 291 analytic participants. Because no machine-readable exclusion list
was identified, Stage A retains all 306 public files rather than silently recreating an
unknown filter. This discrepancy remains visible in the report.

Kwak sessions are paired across treatment folders by the release's source indices.
The archive does not publish cross-treatment calendar timestamps, so chronological
ordering beyond that source-index convention cannot be independently verified.
The release metadata defines its raw choice bit as `0=right, 1=left`, opposite the
canonical `0=left, 1=right` convention. The adapter therefore stores
`animal_response = 1 - source_choice` while retaining `source_choice` for provenance;
reward-condition columns already appear in canonical left/right order.

Alsiö cohorts II–V form a complete multi-session cohort and are admitted together.
The separate cohort VI probabilistic-reversal release contains dose/reversal labels but
no recoverable chronological real-session identity, so it is not mixed into this cohort.

## Skipped cohorts

| Cohort | Raw-release audit | Status | Exact reason |
|---|---|---|---|
| López-Yépez et al. human | 19 subjects, 26 sessions, 4,335 valid binary choices; per-subject session counts range 1–4 | `skipped: requires unsupported split structure` | Mixed structure: 15 subjects have one session, two have two, one has three, and one has four. The complete cohort is neither v1 nor v2. |
| Shin et al. rat | 383 released session matrices from 27 rats | `skipped: requires unsupported split structure` | The public release does not map each session matrix to a rat identity, so subject-level v1 adaptation/test membership cannot be reconstructed. |
| Alsiö et al. cohort VI PRL | Complete PRL trial records, but dose/reversal labels do not define chronological sessions | `skipped: requires unsupported split structure` | Preserving real session state would require a new grouping rule or pseudo-sessions. |
| Hattori et al. | Pinned Dryad derivative located; original Zenodo archive is about 6.5 GB | `skipped: data access blocked` | The smaller pinned Dryad behavior archive returned authorization/WAF failures during the audit. The available `Hattori2019` model remains a useful Stage-B positive-control candidate if data access is restored. |
| Samejima et al. | No stable public trial-level choice/reward release located | `skipped: no public trial-level data` | Aggregate paper results are insufficient for an immutable held-out trial split. |

## Stage-B author-model feasibility gate

No new author-selected model is implemented or launched in Stage A. Existing Grossman,
Chen, Zid, and Hattori-family code may be displayed where an already frozen result exists.

| Cohort | Candidate author family | Available code | Important missing or non-matching details | Estimated effort | Reproduction confidence |
|---|---|---|---|---|---|
| Grossman | uncertainty meta-learning RL | Existing Study 09 implementation | Original hierarchical Stan fit and parameter-order constraint are not reproduced | Existing | Moderate |
| Chen | RLCK | Existing Study 09 implementation | Matched-half fitting differs from the paper's full-data selection protocol | Existing | High for equations; moderate for paper-level reproduction |
| Zid | history-kernel-2 foraging RL | Existing Study 09 implementation | Matched-half fitting differs from the paper's full-data AIC cohort | Existing | High for equations; moderate for paper-level reproduction |
| Lebedeva | PR model | No runnable code in the downloaded behavior archive | Need exact cross-validation/fitting parity and parameter bounds | 3–5 days | Moderate |
| Beron | RFLR | Paper-associated analysis code appears available, not yet pinned | Need choose among mathematically related views and reproduce condition handling | 3–5 days | Moderate |
| Kwak | simple RL | No maintained fit package found | Source-index chronology caveat and treatment pooling must match the paper | 2–3 days | Moderate |
| Miller | three-timescale cognitive mixture | Dataset and manuscript, no maintained fit package found | Full likelihood/state reset and fit-bound details need reconstruction | 5–8 days | Low to moderate |
| Findling | Weber-variability inference | Research code pinned with the data | Hierarchical model and latent-noise inference are substantially more complex | 8–15 days | Moderate |
| Tang | none identified | Neural-analysis release | No selected behavioral baseline to reproduce | Not applicable | High confidence that Stage B needs a new scientific choice |
| Alsiö | separate positive/negative-rate RL | Paper equations; no pinned package | Selected model was evaluated on excluded PRL cohort, not admitted II–V cohort | 3–5 days, but low scientific value | Low for a direct author-aligned claim |
| Eckstein | hierarchical RL plus Bayesian inference | OSF research code | Two co-winners, developmental hierarchy, and 291-versus-306 cohort mismatch | 10–15 days | Moderate |
| Costa | feedback-dependent RL | Published equations | Original optimization and phase-specific fitting details need reconstruction | 3–5 days | Moderate |
| López mouse | double-trace RL | Published MATLAB equations | Original data fit code and exact parameter bounds are not in the release | 4–7 days | Moderate |
| Hattori | `Hattori2019` Q family / Bayesian candidates | Existing model family in `aind-dynamic-foraging-models` | Dataset access is currently blocked; paper/model identity must be pinned before claiming parity | 1–3 days after data access | High for existing equations; low for paper-level parity until audited |

Stage-B selection will occur only after the expanded GRU-versus-common-Q results are
reviewed, using scientific value, effect size, sample/species coverage, code clarity,
effort, and reproduction confidence.
