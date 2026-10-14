# studies/08-hb-vs-gru-heldout/Makefile
#
# One PHONY target per report; `all` regenerates every report from the FROZEN inputs, offline, with no
# W&B, Beaker or HPC access. Run from this directory. Producers are cwd-independent (HERE = Path(__file__).parent).
#
# Freezing is separate from reporting, because it needs credentials and network access:
#   freeze-runs       analysis/extract_hb_runs.py      W&B (WANDB_API_KEY) -> hb_runs.json, hb_runs.csv
#   freeze-posteriors analysis/summarize_hb_posteriors.py   needs the saved posteriors from Beaker result datasets;
#                     see its docstring for arguments -> hb_posterior_summary.json
#   freeze-subjects   analysis/freeze_hb_subjects.py   same inputs plus the per-subject tables -> hb_per_subject_heldout.csv,
#                     hb_subject_posteriors.csv, hb_subjects_manifest.json
#   ml_params/        copied once from /allen with checksums (ml_params/MANIFEST.json); `make verify-ml` re-checks them.
# The frozen files are committed, so `make all` works on a clean checkout.

.PHONY: all r1 r2 r3 r4 verify-ml freeze-runs recompute-stan

all: r1 r2 r3 r4

# r1: three-arm held-out comparison. Reads hb_runs.json, hb_posterior_summary.json and study 01's committed GRU / RL files.
r1:
	python analysis/three_arm.py
	python analysis/update_reports.py --report r1

# r2: per-mouse paired comparison. Needs ml_params/ and hb_per_subject_heldout.csv.
r2: verify-ml
	python analysis/per_mouse.py
	python analysis/update_reports.py --report r2

# r3: shrinkage by parameter. Needs ml_params/ and hb_subject_posteriors.csv.
r3: verify-ml
	python analysis/shrinkage.py
	python analysis/update_reports.py --report r3

# r4: convergence and Stan-vs-NumPyro. Reads hb_posterior_summary.json, hb_runs.json and stan_d29_diagnostics.json.
r4:
	python analysis/convergence.py
	python analysis/update_reports.py --report r4

verify-ml:
	python analysis/ml_params/verify_ml_params.py

# Needs WANDB_API_KEY. Re-running against unchanged W&B state reproduces every per-run digest.
freeze-runs:
	python analysis/extract_hb_runs.py

# Recompute the Stan fit's R-hat / ESS from its saved draws (analysis/stan_inputs/, committed). Needs the ArviZ
# environment (environment.lock), not the reporting one.
recompute-stan:
	python analysis/recompute_stan_diagnostics.py
