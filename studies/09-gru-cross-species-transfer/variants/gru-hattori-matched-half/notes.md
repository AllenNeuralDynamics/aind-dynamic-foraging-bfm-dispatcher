# Hattori (mouse) matched-half GRU transfer

Adapt only each external mouse embedding on chronological odd sessions from the
complete untreated imaging cohort, then score chronological even sessions with
the source GRU core frozen.

Run one E8 D=614 seed-0 GPU smoke first. After it passes, run the 15-cell E4
`D={10,30,100,300,614}` by three-seed matrix and the three-seed E8 D=614 panel.
All GRU work is GPU-only Beaker work.
