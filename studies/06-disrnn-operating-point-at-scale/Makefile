# studies/06-disrnn-operating-point-at-scale/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per docs/posthoc-analysis.md
# "Makefile convention". Invoke from this directory; producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# r1 regenerates OFFLINE from committed study 03 + 05 data (analysis/data/d100_selection.csv +
# ../05.../analysis/grid.csv) -- no WANDB_API_KEY needed, zero new compute.
# r2 regenerates OFFLINE from the committed mult-d-grid (analysis/grid.csv); `pull` re-pulls it
# from W&B (needs WANDB_API_KEY) -- run before r2 to get the freshest numbers while the grid is
# still in flight. r2 overlays this study's live grid against study 05's fixed-penalty curve and
# study 01's GRU curve.
# r3 regenerates OFFLINE from the committed mult-d-grid (analysis/grid.csv) PLUS wave 2's
# committed grid_wave2.csv; `pull_wave2` re-pulls the latter from W&B (needs WANDB_API_KEY;
# wandb.Api() is blocked in the Claude Science sandbox -- see pull_wave2_grid.py's docstring
# for the GraphQL fallback used from there).

PY ?= python
.PHONY: all r1 r2 r3 pull pull_wave2

all: r1 r2 r3

# r1 = penalty selection (Kevin's in-sample-vs-held-out plot), zero new compute.
r1:
	$(PY) analysis/model_selection.py

# r2 = mult-d-grid scaling surface, live while the grid runs. `make pull` first for fresh data.
r2:
	$(PY) analysis/scaling_report.py

# r3 = wave-2 penalty-edge extension (does lighter-than-mult-d-grid's-floor keep helping?).
r3:
	$(PY) analysis/r3_penalty_edge_extension.py

# Refresh analysis/grid.csv from W&B, filtered to this study's group (needs WANDB_API_KEY).
# Not part of `all` — the committed CSV is the source of truth so reports regenerate offline.
pull:
	$(PY) analysis/pull_grid.py

# Refresh analysis/grid_wave2.csv from W&B, filtered to wave 2's group (needs WANDB_API_KEY).
pull_wave2:
	$(PY) analysis/pull_wave2_grid.py
