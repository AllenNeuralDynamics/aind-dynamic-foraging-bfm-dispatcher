# studies/10-disrnn-near-gru-ceiling/Makefile
#
# One PHONY target per report; `all` regenerates everything. Per docs/posthoc-analysis.md
# "Makefile convention". Invoke from this directory; producers are cwd-independent via
# HERE = Path(__file__).parent.
#
# r1 regenerates OFFLINE from the committed analysis/grid.csv + analysis/reference_points.csv;
# `pull` re-pulls both from W&B (needs WANDB_API_KEY).

PY ?= python
.PHONY: all r1 pull

all: r1

# r1 = staged gap-closure attribution across the 3-stage ablation.
r1:
	$(PY) analysis/attribution_report.py

# Refresh analysis/grid.csv + analysis/reference_points.csv from W&B (needs WANDB_API_KEY).
pull:
	$(PY) analysis/pull_grid.py
