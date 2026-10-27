---
name: codebase-map
description: Orient in the aind-dynamic-foraging-bfm-dispatcher (formerly aind-disrnn-dispatcher) codebase — the repo architecture (dispatcher = control plane, wrapper = training payload, studies = sweeps/records/analysis), Hydra config layout, where launchers/configs/docs/studies live, and which skill or doc to read next. Use when starting work in this repo, answering "where does X live", or deciding how to run/launch anything.
---

# Codebase map — aind-dynamic-foraging-bfm-dispatcher

## Big picture (dispatcher, wrapper, studies)

- **This repo (dispatcher)** is the *control plane* for the dynamic-foraging-bfm MLOps stack.
  It composes Hydra configs into job specs and submits them to one of three backends:
  Code Ocean, Beaker (AI Hub), or Allen on-prem SLURM HPC.
- **`aind-dynamic-foraging-bfm-wrapper`** (expected as a sibling checkout at
  `../aind-dynamic-foraging-bfm-wrapper`) is the *compute/runtime payload*: training code, the
  Beaker image, `run_hpc`. Job containers refresh the wrapper, dispatcher, and
  `aind-dynamic-foraging-models` sources at startup, so code edits need **no image
  rebuild** (pin via `WRAPPER_REF`, `DISPATCHER_REF`, and `FORAGING_MODELS_REF`).
- **`aind-dynamic-foraging-bfm-studies`** (sibling checkout at
  `../aind-dynamic-foraging-bfm-studies`) holds every study: sweeps, launch records, analysis,
  reports. Split out of this repo at tag `studies-split-v1` (#124). Launches run from the
  dispatcher and point at a sweep there; no job runtime reads it.
- W&B (`entity: AIND-disRNN`) is the experiment tracker across all backends.
- **Claude Science layer** (AGENTS.md §13): the agent's persistent brain runs on the
  user's Mac; GitHub is the source of truth, tracked by the Mac authoring clone
  (`~/Scripts/aind-dynamic-foraging-bfm-dispatcher`) and a pull-only HPC runtime checkout
  (`/home/han.hou/code/...`). Load balancing: CPU jobs → HPC SLURM, GPU jobs →
  Beaker. Full scheme + credentials: `references/claude-science-workflow.md`.

## Repo layout

| Path | What it is |
|---|---|
| `code/config/` | Hydra config groups (`data=mice\|synthetic`, `model=disrnn\|baseline_rl\|...`) — composed here, consumed by the wrapper's `run_capsule.py` |
| `code/launch_beaker_resumable.py` | Preferred Beaker launcher: grid sweep → one resumable preemptible task per grid point |
| `code/launch_beaker.py` | Native-route Beaker launcher: `wandb sweep` + `wandb agent` replicas |
| `code/beaker/` | Beaker sweep/experiment YAMLs + `README.md` (control-plane detail) |
| `code/launch_hpc.py` | Allen on-prem SLURM launcher (W&B sweep + sbatch array) |
| `code/hpc/` | SLURM scripts, sweep YAMLs, `user.env`, + `README.md` |
| `code/launch_CO_wrapper.py` | Code Ocean route |
| `code/check_gpu_availability.py` | Schedulable-GPU probe (Beaker + HPC) — mandatory before large launches (AGENTS.md §10) |
| `code/beaker_client.py` | Sandbox-safe Beaker/W&B client helpers used by the launchers |
| `../aind-dynamic-foraging-bfm-studies/studies/<study>/` | **Sibling repo** (split out of this one, #124): one folder per scientific question (`NN-{model}-{purpose}`); variants + analysis (see study-conventions skill). The launchers here take its sweep paths and record its commit as `studies_git_commit` |
| `../aind-dynamic-foraging-bfm-studies/studies/util/` | Shared helpers: `_meta.py` (provenance block), `figure_style.py` + `figure_audit.py` (paper-style SVG figures, checked by `scripts/check_figures.py`; `plot_style.py` is the legacy style of studies 01–11), `launch_record.py` (intervention records), `validate_provenance.py` (wrap-up reconciler), `watch_runs.py` (poll Beaker+W&B, append timestamped snapshots so transitions between check-ins aren't lost) |
| `docs/` | Design records + pointer stubs (index below) |
| `AGENTS.md` | Behavioral guardrails — the terse rules; this pack and the code-adjacent docs hold the detail |

## Where knowledge lives (one canonical home per topic)

- **AGENTS.md** — always-loaded terse guardrails (dispatcher, wrapper and studies repos each carry one).
- **This skills pack** — canonical for cross-cutting *operational* knowledge:
  launching (beaker-launch, hpc-launch), study conventions (study-conventions),
  reporting (posthoc-reporting), git/provenance (git-session-isolation), the Claude
  Science workflow (`references/claude-science-workflow.md`). The former `docs/*.md`
  playbooks were absorbed into these skills and are now pointer stubs — update the
  skill, not the stub.
- **`docs/` design records** — canonical for *forward-looking modeling decisions*, which
  no skill covers. Read the relevant one before touching that modeling direction:
  - `docs/design-hb-baseline.md` — settled decision record for the hierarchical-Bayesian
    cognitive-model baseline (implementation state lives in dispatcher issue #72, not the
    note). Relevant to `studies/08-hb-vs-gru-heldout/`.
  - `docs/design-hierarchical-vi-foundation-model.md` — design/plan (not yet implemented)
    for a hierarchical mixed-effects foundation model via amortized VI.
  - `docs/repo-split-plan.md` — why the dispatcher/wrapper split is shaped as it is; read
    before moving code across the two repos.
- **Code-adjacent living docs** — canonical for code-coupled reference; skills
  defer to them:
  - `../aind-dynamic-foraging-bfm-wrapper/code/TRAINING.md` — **§1.5 "Run lifecycle & key
    switches" first**: the four run phases, the `_step` warmup offset, the **two
    different held-out switches**, checkpoints/resumability/extendability.
    **Read before interpreting any run's logs or metrics** (distilled in the
    wrapper-runtime skill).
  - `../aind-dynamic-foraging-bfm-wrapper/code/POST_TRAINING_ANALYSIS.md` — the analysis
    codebase + `run_analysis.py` CLI.
  - `../aind-dynamic-foraging-bfm-wrapper/beaker/README.md` — image build plane +
    GPU-efficiency benchmarks (why L40S beats H200 here; batch/length-bucketing
    levers).
  - `code/beaker/README.md` — Beaker flow, cluster + **image** tables, memory
    pitfalls, resumable-run mechanics.
  - `code/hpc/README.md` — SLURM setup, launch variants, monitoring.

## Which skill next — and why the rules are not on this page

**This map answers "where is X" and "which skill next"; it deliberately does not state
the rules themselves.** Each rule has one owner, and a copy here is a second copy that
rots while the owner is corrected. Since this is the first skill loaded, a stale copy
here is the first thing every agent reads — so the rules stay with their owners.

| To do / know | Load |
|---|---|
| Submit to Beaker: cluster allowlist, capacity, sizing, priority, resumable launches | **beaker-launch** |
| Launch on Allen on-prem SLURM; the login-node prohibition; `dynamic-foraging-bfm-cpu` | **hpc-launch** |
| Interpret logs/metrics, held-out numbers, checkpoints/resume; training + analysis code | **wrapper-runtime** |
| Create a study/variant, name W&B groups, record an intervention | **study-conventions** |
| Write or regenerate reports, figures, curated JSON | **posthoc-reporting** |
| Commit / branch / push, provenance SHAs, concurrent sessions on a shared checkout | **git-session-isolation** |
| File an issue, set its project-board fields, link a PR | **issue-tracking** |
| Commit style, merge policy, Seattle-time logs, verify-with-data | `AGENTS.md` (always loaded) |

No exceptions to that, including for the rules it would be most tempting to repeat (the
login-node prohibition, the capacity check). `AGENTS.md` is always loaded and states both;
a copy here would add nothing except a second thing to keep correct.
