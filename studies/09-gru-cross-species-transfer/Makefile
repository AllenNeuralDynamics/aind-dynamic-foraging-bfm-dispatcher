PYTHON ?= python
CACHE_ROOT ?= data-cache
PYTHONPATH := ../../code

.PHONY: all suite rebuild examples r2

all: r2

suite:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m external_bandit_datasets.cli all --cache-root $(CACHE_ROOT)

rebuild:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m external_bandit_datasets.cli all --cache-root $(CACHE_ROOT) --no-download

examples:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) analysis/freeze_example_behavior_sessions.py --raw-root $(CACHE_ROOT)/raw

r2:
	$(PYTHON) analysis/report_author_baselines.py
