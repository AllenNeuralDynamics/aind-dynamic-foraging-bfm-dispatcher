PYTHON ?= python
WRAPPER ?= ../../../aind-dynamic-foraging-bfm-wrapper/code
MODEL ?= C3-s0
RUN = PYTHONPATH=$(WRAPPER) $(PYTHON)

.PHONY: freeze-sources fetch extract

# Reads W&B over GraphQL; needs WANDB_API_KEY (or ~/.netrc). Rewrites source_runs.json.
freeze-sources:
	$(PYTHON) analysis/freeze_source_runs.py

# Stage A (weights + data + JAX; GPU box, or CPU with JAX_PLATFORMS=cpu). MODEL=<key> from source_runs.json.
# extract peaks at ~31 GB RAM per model and refuses to start below 40 GB available:
# run one model at a time, never with -j or alongside another heavy job.
fetch:
	$(RUN) analysis/fetch_models.py $(MODEL)

extract: fetch
	$(RUN) analysis/extract_trial_states.py $(MODEL)
