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
| Grossman (mouse) | Mouse; blockwise independent probabilities | [Dryad v4](https://doi.org/10.5061/dryad.cz8w9gj4s), CC0 | 48 | 754 | 210,159 | 101,877 | v1 | uncertainty-dependent meta-learning RL | Moderate; already implemented | Completed; reuse frozen GRU/Q/author results |
| Chen (mouse) | Mouse; restless independent random walks | [Dryad v5](https://doi.org/10.5061/dryad.z612jm6c0), CC0 | 32 | 256 | 70,778 | 35,644 | v1 | four-parameter RL plus choice kernel (RLCK) | Low; already implemented | Completed; reuse frozen GRU/Q/author results |
| Zid (human) | Human; schedule-matched restless random walks | [Figshare v5](https://doi.org/10.6084/m9.figshare.32193990.v5), MIT | 258 | 258 | 77,400 | 38,700 | v2 | history-kernel-2 foraging RL; RLCK comparator | Low; already implemented | Completed; reuse frozen GRU/Q/author results |
| Lebedeva (mouse) | Mouse; 80/20 probabilistic reversals | [Figshare v8](https://doi.org/10.6084/m9.figshare.31231741.v8), CC BY 4.0 | 10 | 254 | 132,494 | 63,993 | v1 | perseveration/reward-learning (PR) model; simplest of five tied top models | Moderate; paper and pinned author code disagree on one initial state, which is disclosed | Author PR reproduction implemented; matched-half HPC fit completed |
| Beron (mouse) | Mouse; nonstationary probabilistic two-arm bandit | [Harvard Dataverse v1.1](https://doi.org/10.7910/DVN/7E0NM5), CC0 | 6 | 525 | 378,351 | 188,926 | v1 | recursively formulated logistic regression (RFLR); equivalent RL/sticky-HMM views | Moderate; equations and pinned author code are available | Author RFLR reproduction implemented; matched-half HPC fit completed |
| Kwak (mouse) | Mouse; dynamic two-arm bandit under D1/D2 manipulation | [Dryad](https://doi.org/10.5061/dryad.4c80mn5), CC0 | 39 | 780 | 121,100 | 60,045 | v1 | subject/condition-level RL with learning rate and value-dependent softmax choice | Low to moderate; simple equations, original fit code not found | Quarantined; current split adapts on CNO and tests on DMSO. Rerun DMSO/control only with odd sessions adapting and even sessions testing |
| Miller (rat) | Rat; large dynamic two-arm bandit | [Figshare v2](https://doi.org/10.6084/m9.figshare.20449356.v2), CC BY 4.0 | 20 | 1,857 | 1,040,731 | 515,238 | v1 | reward-seeking, habit, and gambler's-fallacy (RHG) model | Moderate to high; exact executable equations were recovered from a later symbolic-model paper | Author RHG reproduction complete; matched-half likelihood 0.615715 on identical held-out trials |
| Findling (human) | Human; 85/15 reversals under varying volatility | [GitHub commit `ee68853`](https://github.com/csmfindling/Volnoise/tree/ee688535b569a8af8c0531350ec06f34cb989f8e), MIT | 22 | 132 | 23,275 | 11,706 | v1 | Weber-imprecision Bayesian inference model | High; stochastic latent-state particle filter with an unusually low-particle released fit | Author filter and exact Sobol grid reproduced; primary two-particle and 64-particle sensitivity fits completed |
| Tang (macaque) | Macaque; blockwise action/object values | [Mendeley Data v1](https://doi.org/10.17632/m4f38w49fb.1), CC BY 4.0 | 2 | 8 | 15,375 | 7,728 | v1 | no author-selected behavioral prediction model identified | Not applicable unless a new scientific baseline is chosen | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Alsiö (rat) | Rat; visual/spatial discrimination and reversal | [Cambridge Apollo](https://doi.org/10.17863/CAM.80290), CC BY 4.0 | 95 | 2,334 | 457,007 | 225,751 | v1 | separate positive/negative learning-rate RL was selected for the separate PRL cohort | High for direct comparison: selected model and admitted cohort do not align | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Eckstein (human) | Human; stochastic reversal across development | [OSF `7wuh4`](https://osf.io/7wuh4/) (no explicit OSF data license) | 306 | 306 | 40,229 | 20,248 | v2 | co-winners: counterfactual RL and Bayesian-inference models, originally fit hierarchically | High; two co-winners, hierarchical original fit, and paper reports an analytic n=291 | Both author co-winners implemented; matched-half HPC fits completed |
| Costa (macaque) | Macaque; stochastic stimulus reversal | [Zenodo record 20086410](https://doi.org/10.5281/zenodo.20086410), CC BY 4.0 | 11 | 245 | 329,840 | 162,960 | v1 | feedback-dependent RL with separate positive/negative learning rates | Moderate; equations are published, original fitting code not found | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| López-Yépez (mouse) | Mouse; baited variable-interval dynamic matching | [Figshare v1](https://doi.org/10.6084/m9.figshare.14540283.v1), CC BY 4.0 | 8 | 218 | 147,726 | 77,662 | v1 | double-trace RL with fast and slow choice traces | Moderate; equations are published, MATLAB implementation was not released with the data | Admitted; GRU and common Q completed and frozen; exact trial-key parity passed |
| Hattori (mouse) | Mouse; baited probabilistic reversals during longitudinal OFC imaging | [Zenodo v3](https://doi.org/10.5281/zenodo.10969434) (no explicit Zenodo data license) | 7 | 390 | 192,272 | 94,425 | v1 | asymmetric rewarded/unrewarded Q-learning with unchosen-value forgetting | Low; the exact equations match the existing `Hattori2019` family | Admitted; checksum, exact-count, deterministic-manifest, and wrapper-loader audits passed; matched fits pending |

The Eckstein (human) release contains 306 valid one-session participant files, while the
paper reports 291 analytic participants. Because no machine-readable exclusion list
was identified, Stage A retains all 306 public files rather than silently recreating an
unknown filter. This discrepancy remains visible in the report.

Kwak (mouse) sessions are paired across treatment folders by the release's source
indices. The current generic lexical split places CNO sessions in adaptation and DMSO
sessions in testing, so all frozen Kwak (mouse) scores and embeddings are quarantined.
The replacement cohort will retain DMSO/control sessions only and split chronological
odd DMSO sessions for adaptation versus even DMSO sessions for testing. The archive
does not publish cross-treatment calendar timestamps, but that ordering is no longer
needed for the within-control estimand.
The release metadata defines its raw choice bit as `0=right, 1=left`, opposite the
canonical `0=left, 1=right` convention. The adapter therefore stores
`animal_response = 1 - source_choice` while retaining `source_choice` for provenance;
reward-condition columns already appear in canonical left/right order.

Alsiö (rat) cohorts II–V form a complete multi-session cohort and are admitted together.
The separate cohort VI probabilistic-reversal release contains dose/reversal labels but
no recoverable chronological real-session identity, so it is not mixed into this cohort.

Hattori (mouse) uses the complete seven-mouse untreated longitudinal imaging cohort.
The intervention and paAIP2 cohorts are separate causal experiments and are not mixed
into this within-condition transfer estimand. Source actions `1=right, 2=left` map to
canonical `0=left, 1=right`; alarm (`3`) and miss (`4`) trials have no binary choice and
are excluded. All 390 calendar-dated imaging sessions remain distinct and chronological.

## Skipped cohorts

| Cohort | Raw-release audit | Status | Exact reason |
|---|---|---|---|
| López-Yépez (human) | 19 subjects, 26 sessions, 4,335 valid binary choices; per-subject session counts range 1–4 | `skipped: requires unsupported split structure` | Mixed structure: 15 subjects have one session, two have two, one has three, and one has four. The complete cohort is neither v1 nor v2. |
| Shin (rat) | 383 released session matrices from 27 rats | `skipped: requires unsupported split structure` | The public release does not map each session matrix to a rat identity, so subject-level v1 adaptation/test membership cannot be reconstructed. |
| Alsiö (rat) cohort VI PRL | Complete PRL trial records, but dose/reversal labels do not define chronological sessions | `skipped: requires unsupported split structure` | Preserving real session state would require a new grouping rule or pseudo-sessions. |
| Samejima (macaque) | No stable public trial-level choice/reward release located | `skipped: no public trial-level data` | Aggregate paper results are insufficient for an immutable held-out trial split. |

## Stage-B author-model feasibility gate

The primary-set models selected after the Stage-A screen are now implemented.
All are fitted to the same adaptation half and scored on the same held-out trials as
GRU and common Q. This is a matched predictive benchmark, not a claim to recreate each
paper's original population-level model-selection analysis.

| Cohort | Candidate author family | Available code | Important missing or non-matching details | Estimated effort | Reproduction confidence |
|---|---|---|---|---|---|
| Grossman (mouse) | uncertainty meta-learning RL | Existing Study 09 implementation | Original hierarchical Stan fit and parameter-order constraint are not reproduced | Existing | Moderate |
| Chen (mouse) | RLCK | Existing Study 09 implementation | Matched-half fitting differs from the paper's full-data selection protocol | Existing | High for equations; moderate for paper-level reproduction |
| Zid (human) | history-kernel-2 foraging RL | Existing Study 09 implementation | Matched-half fitting differs from the paper's full-data AIC cohort | Existing | High for equations; moderate for paper-level reproduction |
| Lebedeva (mouse) | PR model | Pinned MATLAB author code plus paper equations | Paper says both states start at zero; code starts reward state at five. We follow the paper and disclose finite fitting bounds | Implemented | Moderate |
| Beron (mouse) | RFLR | Pinned paper-associated Python code | Original unconstrained SGD is replaced by bounded differential evolution on the matched half | Implemented | High for equations; moderate for fit-procedure parity |
| Kwak (mouse) | simple RL | No maintained fit package found | DMSO/control-only odd/even session split must replace the invalid cross-treatment result | 2–3 days | Moderate |
| Miller (rat) | RHG cognitive mixture | Exact executable specification in Castro et al. Appendix E.2 | The original paper does not provide maintained fitting code; finite optimization bounds are disclosed | Implemented | Moderate to high for equations; moderate for paper-level fit parity |
| Findling (human) | Weber-imprecision inference | Pinned research code and exact released Sobol grid | Released fit uses two stochastic particles and no seed; our seed is fixed and held-out prediction uses 256 particles | Implemented | High for code-path reproduction; moderate for stochastic fit stability |
| Tang (macaque) | none identified | Neural-analysis release | No selected behavioral baseline to reproduce | Not applicable | High confidence that Stage B needs a new scientific choice |
| Alsiö (rat) | separate positive/negative-rate RL | Paper equations; no pinned package | Selected model was evaluated on excluded PRL cohort, not admitted II–V cohort | 3–5 days, but low scientific value | Low for a direct author-aligned claim |
| Eckstein (human) | counterfactual RL plus Bayesian inference | Pinned OSF research code | Individual matched-prefix MLE replaces the paper's hierarchical population fit; both co-winners and the 291-versus-306 mismatch remain visible | Implemented | High for equations; moderate for paper-level fit parity |
| Costa (macaque) | feedback-dependent RL | Published equations | Original optimization and phase-specific fitting details need reconstruction | 3–5 days | Moderate |
| López-Yépez (mouse) | double-trace RL | Published MATLAB equations | Original data fit code and exact parameter bounds are not in the release | 4–7 days | Moderate |
| Hattori (mouse) | asymmetric rewarded/unrewarded Q-learning with unchosen-value forgetting | Existing `Hattori2019` family in `aind-dynamic-foraging-models` | The matched-half subject fits do not reproduce the paper's L2 penalty selected by tenfold cross-validation | Implemented; fit pending | High for equations; moderate for paper-level fitting parity |

The primary set has passed this gate. Any additional author model remains gated on
scientific value, effect size, sample/species coverage, code clarity, effort, and
reproduction confidence; none is added automatically.
