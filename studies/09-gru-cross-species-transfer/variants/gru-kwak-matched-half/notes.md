# Kwak matched-half GRU transfer

Adapt only each external subject embedding for 500 steps at learning rate 0.001
with the source GRU core frozen. Score the immutable held-out trials for
D={10,30,100,300,614} and source seeds 0–2.

Run `smoke.yaml` first (D=614, seed 0). Launch the full 15-cell `sweep.yaml`
only after that GPU smoke passes. This variant is GPU-only on Beaker; no
CPU-only work belongs in this experiment.

The original 2026-09-05 launch used an incorrect direct copy of the released
choice bit. Kwak defines `0=right, 1=left`, opposite the canonical
`0=left, 1=right` contract. Results from that launch are superseded by the
choice-corrected rerun using the immutable bundle recorded in
`analysis/kwak_choicefix_bundle.json`; the split manifest itself is unchanged.
