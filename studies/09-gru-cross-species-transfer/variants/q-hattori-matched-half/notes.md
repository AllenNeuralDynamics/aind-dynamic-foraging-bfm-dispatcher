# Hattori (mouse) common-Q baseline

Fit the common subject-level Q model to each mouse's chronological odd sessions
and score the identical even sessions used by GRU and the author-aligned model.

This is a CPU-only Allen HPC job. The checksum-verified canonical v1 table is
staged once to HPC scratch; the 6.5 GB neural archive is not downloaded by each
fit task.
