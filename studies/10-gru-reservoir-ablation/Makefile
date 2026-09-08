PYTHON ?= python

.PHONY: all freeze r1
all: r1

# Run on an HPC compute node after every pinned W&B group settles.
freeze:
	$(PYTHON) analysis/freeze_results.py

# Pure offline formatter; run locally from the committed frozen JSON.
r1:
	$(PYTHON) analysis/report_reservoir.py
