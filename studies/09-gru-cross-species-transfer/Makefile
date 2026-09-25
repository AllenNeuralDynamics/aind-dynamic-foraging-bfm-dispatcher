PYTHON ?= python
CACHE_ROOT ?= data-cache
SOURCE_DIMENSION ?=
PYTHONPATH := ../../code

.PHONY: all suite rebuild validate smoke-loaders freeze-matched freeze-authors examples freeze-embeddings freeze-generalization freeze-task-design freeze-source-embedding-manifests freeze-embedding-dimension freeze-embedding-dimension-expansion r1 r2 r3 r4

all: r1 r2 r3 r4

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

freeze-authors:
	$(PYTHON) analysis/freeze_author_baselines.py

examples:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) analysis/freeze_example_behavior_sessions.py --raw-root $(CACHE_ROOT)/raw

freeze-embeddings:
	$(PYTHON) analysis/freeze_embedding_space.py --dimension 4
	$(PYTHON) analysis/freeze_embedding_space.py --dimension 8

freeze-generalization:
	$(PYTHON) analysis/freeze_generalization_drivers.py --dimension 4
	$(PYTHON) analysis/freeze_generalization_drivers.py --dimension 8

freeze-task-design:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) analysis/freeze_task_design_features.py --dimension 4 --raw-root $(CACHE_ROOT)/raw
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) analysis/freeze_task_design_features.py --dimension 8 --raw-root $(CACHE_ROOT)/raw

freeze-source-embedding-manifests:
	$(PYTHON) analysis/freeze_source_embedding_manifests.py $(if $(SOURCE_DIMENSION),--dimension $(SOURCE_DIMENSION))

freeze-embedding-dimension:
	$(PYTHON) analysis/freeze_embedding_dimension.py

freeze-embedding-dimension-expansion:
	$(PYTHON) analysis/freeze_embedding_dimension_expansion.py

r1:
	$(PYTHON) analysis/report_author_baselines.py

r2:
	$(PYTHON) analysis/report_embedding_space.py

r3:
	$(PYTHON) analysis/report_generalization_drivers.py

r4:
	$(PYTHON) analysis/report_embedding_dimension.py
