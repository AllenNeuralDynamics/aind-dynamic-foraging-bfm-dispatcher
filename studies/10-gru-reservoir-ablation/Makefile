PYTHON ?= python

.PHONY: all freeze freeze-curve r1
all: r1

# Run on an HPC compute node after every pinned W&B group settles.
freeze:
	$(PYTHON) analysis/freeze_results.py

freeze-curve:
	$(PYTHON) analysis/freeze_dcurve.py

# Pure offline formatter; run locally from the committed frozen JSON.
r1:
	$(PYTHON) analysis/report_reservoir.py
