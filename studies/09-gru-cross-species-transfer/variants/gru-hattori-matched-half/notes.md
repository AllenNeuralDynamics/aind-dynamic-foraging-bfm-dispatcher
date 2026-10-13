# Hattori (mouse) matched-half GRU transfer

Retain every mouse in the untreated imaging cohort, then keep only each mouse's
15th probabilistic-reversal session onward, following the paper's late/expert
threshold. Adapt only each external mouse embedding on chronological odd mature
sessions, then score chronological even mature sessions with the source GRU
core frozen.

Run one E8 D=614 seed-0 GPU smoke first. After it passes, run the 15-cell E4
`D={10,30,100,300,614}` by three-seed matrix and the three-seed E8 D=614 panel.
All GRU work is GPU-only Beaker work.
