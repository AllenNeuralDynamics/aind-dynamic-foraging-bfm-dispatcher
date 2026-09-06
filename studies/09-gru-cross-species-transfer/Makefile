PYTHON ?= python
CACHE_ROOT ?= data-cache
PYTHONPATH := ../../code

.PHONY: all suite rebuild validate smoke-loaders freeze-matched examples freeze-embeddings r2 r3

all: r2 r3

suite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m external_bandit_datasets.cli all --cache-root $(CACHE_ROOT)

rebuild:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m external_bandit_datasets.cli all --cache-root $(CACHE_ROOT) --no-download

validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) analysis/validate_dataset_suite.py \
		--canonical-root $(CACHE_ROOT)/canonical \
		--output analysis/dataset_suite_validation.json

smoke-loaders:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) analysis/smoke_external_loaders.py \
		--canonical-root $(CACHE_ROOT)/canonical \
		--wrapper-code $${BFM_WRAPPER_CODE:?BFM_WRAPPER_CODE is required}

freeze-matched:
	$(PYTHON) analysis/freeze_matched_half.py

examples:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) analysis/freeze_example_behavior_sessions.py --raw-root $(CACHE_ROOT)/raw

freeze-embeddings:
	$(PYTHON) analysis/freeze_embedding_space.py

r2:
	$(PYTHON) analysis/report_author_baselines.py

r3:
	$(PYTHON) analysis/report_embedding_space.py
