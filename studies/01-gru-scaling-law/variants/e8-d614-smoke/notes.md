# E=8, D=614 source-core smoke

**Question.** Does the full 614-source-mouse GRU pipeline accept an eight-dimensional
subject embedding and complete held-out embedding adaptation in a one-GPU bundle?

This is a compressed, non-scientific plumbing run: 1,000 source-training steps and
10 held-out adaptation steps, seed 0 only. It changes only the subject-embedding
dimension relative to the established H=128 Study 01 GRU architecture. A successful
run licenses the paired `e4-e8-d614-source` launch; its likelihood must not enter a
scientific report.

Status: running under #148. W&B group `e8-d614-smoke@20260906-192034`;
Beaker experiment [`01M1WTNQSMRKSKCWD1Q6CXRPHT`](https://beaker.org/ex/01M1WTNQSMRKSKCWD1Q6CXRPHT).
