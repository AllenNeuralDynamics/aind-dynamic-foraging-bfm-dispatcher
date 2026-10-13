PYTHON ?= python

.PHONY: all freeze freeze-curve freeze-reference validate r1
all: r1

# Run on an HPC compute node after every pinned W&B group settles.
freeze:
	$(PYTHON) analysis/freeze_results.py

freeze-curve:
	$(PYTHON) analysis/freeze_dcurve.py

freeze-reference:
	$(PYTHON) analysis/freeze_reference.py

validate:
	$(PYTHON) ../util/validate_provenance.py . --variant frozen-random-core-d614 --strict
	$(PYTHON) ../util/validate_provenance.py . --variant frozen-random-core-h128-dcurve --strict

# Pure offline formatter; run locally from the committed frozen JSON.
r1:
	$(PYTHON) analysis/report_reservoir.py
