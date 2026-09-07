# Paired E=4 versus E=8 source cores at D=614

**Scientific question.** Is the four-dimensional subject embedding a capacity
bottleneck for held-out AIND-mouse generalization and downstream external transfer?

The six cells cross `E={4,8}` with source seed `{0,1,2}`. Both dimensions train from
scratch on the same `20260603` snapshot, full source cohort (`subject_ratio=1.0`),
H=128 GRU, scalar session conditioning, 150,000-step budget, optimizer, curriculum,
early-stopping gate, and held-out-mouse protocol. The newly rerun E=4 arm isolates
embedding dimension from software/image drift; the historical Study 01 E=4 group is
a consistency reference rather than the primary comparator.

Study 01 will report paired held-out AIND performance and the source-embedding
covariance spectrum. Study 09 will consume the immutable E=4 and E=8 artifacts for
the external matched-half screen without retraining either core.

Status: planned under #148; launch only after the E=8 smoke completes.
