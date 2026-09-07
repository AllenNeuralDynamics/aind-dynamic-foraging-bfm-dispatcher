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

Status: running under #148. All six one-GPU tasks started on the on-prem H200
hub in Beaker experiment
[`01M1WWK9HP269XQS440Y29631F`](https://beaker.org/ex/01M1WWK9HP269XQS440Y29631F),
W&B group `e4-e8-d614-source@20260906-195409`.
