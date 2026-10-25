---
name: posthoc-reporting
description: Post-hoc analysis and reporting conventions for studies — report files with YAML frontmatter and BEGIN/END result markers, JSON _meta blocks, launch_record results.md stubs, Makefile targets, cache .gitignore policy, one-producer-per-artifact, and running report generation locally rather than on HPC. Use when writing/regenerating analysis reports, figures, or JSON summaries from finished W&B groups.
---

# Post-hoc analysis & reporting

This skill is the source of truth for analysis/reporting conventions (the one-line
rule lives in `AGENTS.md` §12). Full artifact contracts: `references/` here. When
in doubt, mirror a normalized study (e.g. `studies/01-gru-scaling-law/`) rather
than inventing layout.

Studies live in **`aind-dynamic-foraging-bfm-studies`**, a sibling checkout of the
dispatcher (split out at tag `studies-split-v1`, dispatcher #124). Every `studies/...` path
below is relative to that repo's root; the launchers stay in the dispatcher.

**Field definitions live in the studies repo** — report frontmatter
([`schemas/report-frontmatter.v1.schema.json`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/main/schemas/report-frontmatter.v1.schema.json)),
analysis `_meta` ([`schemas/meta.v2.schema.json`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/main/schemas/meta.v2.schema.json)) and the
vocabulary ([`CONTEXT.md`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/main/CONTEXT.md)). This skill keeps the rules and the why. Run
`make validate` at the studies root before a PR; CI's required `provenance` check runs the
same checker and fails on any new finding.

## Principles

1. **Reports are code** — committed, regenerable, diffable; changed only via their
   producer script (manual prose lives outside markers).
2. **One producer per artifact** — every figure/JSON is written by exactly one script.
3. **Caches are `.gitignore`'d** — anything re-derivable from W&B (per-subject pulls,
   `analysis/_cache_*.json`). Curated summary JSON and reports/figures are committed.
4. **Provenance lives inside the artifact** — JSON `_meta`, report frontmatter.
5. **Idempotent regeneration** — `make r<n>` twice produces identical files
   (see the caveats below: `_meta` timestamps and cross-machine figure bytes).

## Pin the W&B groups — never discover them live (hard rule)

A producer MUST read runs from a **declared allowlist** of W&B groups
(`WANDB_GROUPS = [...]` at module level, queried per group, as in
`studies/01-gru-scaling-law/analysis/nxd_scaling.py` / `freeze_nxd.py`). It must NOT query the whole
project and keep whatever matches its grid.

Why: a project-wide query silently absorbs **any** new run that happens to fit the
cell definition, so a report's numbers change when someone launches something
unrelated. This actually happened — `02-gru-scaling-law-ignore/analysis/scaling.py`
globbed the project, and a 200-step validation smoke run moved its published H16
mean from 0.71635 to 0.70348 (SEM 5.3e-5 → 1.3e-2) with no code change. Pinned
producers (studies 01/03/04) ignored the same run and reproduced exactly.

`_meta.wandb_groups` must be a subset of the declared allowlist — never a set
discovered from whatever the query returned, which makes a contaminated pull look
self-consistent.

## Freeze the numbers, key them by an immutable ID — never a live handle (hard rule)

A committed figure/table MUST be built from **frozen values in a committed file**
(CSV/JSON), read by an **offline** producer. The producer must NOT fetch from W&B,
cloud storage, or any live API on the reproducible path — W&B is a live view, not a
frozen store (runs get deleted, artifacts re-versioned, `:latest` re-pointed, tags
edited). A query is reproducible only until someone touches the project.

The run identifier is the provenance **key**, not the data. Record it *in the frozen
file* so each row is auditable and deterministically regenerable — but the committed
file, not the run, is the source of truth. Two columns per row:

- `wandb_run_id` — the run whose config (n / hidden / embed / seed …) fully identifies
  the cell. Human-readable, but a run alone is not enough: it can be deleted or its
  output artifact re-versioned.
- an **immutable digest** of the exact input used — the artifact `version_id` /
  content digest that produced the number. This is the authoritative key; `:latest`,
  a sweep-membership query, or a W&B group are NOT immutable and must never be the key.

**Never key a value by an ambiguous or mutable handle.** Three that have burned us:
- **Filename prefixes that look like IDs but aren't.** A producer hardcoded paths like
  `.../v833ecb41_subject_embeddings.pkl`; the same logical run was re-saved many times
  and the prefix->run mapping drifted, so re-running loaded *different data* — the figure
  "totally changed" with no code edit. Fix: read by immutable `version_id`, resolved via
  lineage, not by on-disk name.
- **Content that omits the swept axis.** The stage-1 embedding pickle records
  `n_subjects` + `embed_dim` but NOT `hidden_size`; the grid ran hidden {16,64}, so
  `(n, embed)` was not a unique key and pooling "all copies of a cell" silently mixed
  two widths — which manufactured a fake run-to-run "spread" (~0 for embed-4 where
  recovery saturates; up to 0.76 for embed-2 below the identifiability threshold). If
  content must be the key, ASSERT the discriminating config is present before trusting it.
- **A mean over ambiguous copies wearing one run's ID.** Averaging those mixed-width
  pickles and stamping the single canonical `run_id` on the mean implies run-specific
  precision that doesn't exist. Freeze ONE deterministic value per cell, keyed to the
  one immutable digest that produced it — don't average and relabel.

**State the absence of replicates.** Single seed per cell => no run-to-run variance =>
no error bars. Say so in the caption/Methods; never let a reader infer a band that was
never measured.

To refresh numbers: re-run extraction on an authenticated node and **overwrite the
committed file** — the git diff is the audit trail. The offline producer never changes.

## What "reproducible" actually means here (verified 2026-07-11)

- **Data (curated JSON/CSV) is exactly reproducible** — a re-run changes nothing but
  `_meta`. This is the thing to check: `git diff` the JSON, ignore `_meta`, expect zero.
- **`_meta.produced_at_pt` (and `studies_git_sha` / `studies_git_dirty`) change every
  run**, so regeneration always dirties the tree. A CI check cannot be a bare
  `git diff --exit-code` — it must ignore `_meta`.
- **Figure PNGs are bit-identical only on the same machine.** Across machines the
  bytes (and ~6% of pixels — text/AA rendering) differ, because the *plotting* env
  is not pinned: `environment.lock` pins the wrapper commit that produced the
  *metrics*, not matplotlib/freetype. A PNG diff after regenerating on a different
  box is expected and is **not** evidence that results changed — confirm via the JSON.

## Run report generation LOCALLY, not on HPC

**A report producer runs in the local sandbox by default. Reaching for `sbatch` is the
error.** This follows from the freeze rule above rather than being a separate policy: a
producer on the reproducible path reads a committed CSV/JSON and touches no live service,
so it has nothing to gain from a compute node — and an `sbatch` round-trip adds queue wait,
a second checkout to keep in sync, and a log file to go fetch, in exchange for nothing.

The discriminator is **not** "does it read W&B" — it is whether the script needs something
the sandbox genuinely lacks. Three things do:

```bash
# scripts that need a node; everything else runs locally
grep -lE 'wandb\.Api\(|^\s*import wandb|from wandb|/allen/|BEAKER_TOKEN' analysis/*.py
```

- **The `wandb` Python SDK** — `wandb.Api()` cannot start in the sandbox (it spawns a
  `wandb-core` helper the sandbox blocks). Study 01's `freeze_per_subject.py`,
  `freeze_rl_baseline.py` (also `/allen`), `freeze_nxd.py`, `freeze_generative.py`;
  `mature_fewshot_curve.py`.
- **The `/allen` mount** — HPC-only (`freeze_rl_baseline.py`'s fallback).
- **Large artifact streaming** — `fetch_gru_history_patterns.py` (`BEAKER_TOKEN`, ~3.1 GB).

**Since dispatcher #177 every study's `make all` is offline, and CI enforces it**
(`scripts/check_regeneration.py` in the studies repo, no credentials). The live reads above
live only in `freeze-*` Makefile targets that write a committed frozen JSON — one row per kept
run, keyed by W&B run id plus the artifact digest (or the `/allen` file's sha256) where a
table was read — and the producers read only that file. A new producer that needs W&B
follows the same split: freeze script + offline producer.

Everything else is local, **including W&B reads over GraphQL.** `api.wandb.ai` + `requests`
works fine in the sandbox and only needs `WANDB_API_KEY` exported — that is what
`references/wandb-graphql-sandbox.md` is for. Study-02's `scaling.py` pulls W&B this way and
runs locally; study-01's `freeze_nxd.py` pulls the same kind of data through the SDK and
cannot.
**Same data, different route, different answer** — so read the import, don't assume from the
fact that a producer mentions W&B.

Pure formatters are the clear-cut local case: they read the frozen file and rewrite report
markers or redraw a figure (`update_final_report_nxd.py`, `update_r8.py`,
`pswitch_history_patterns.py`, `embedding_param_decode.py`; study-02's `update_reports.py`).

A refresh spans both classes — study-01's `make freeze` (node) rewrites the frozen JSONs,
then `make r7` (local) runs `nxd_scaling.py`, `rl_baseline.py` and
`update_final_report_nxd.py` offline — so the rule is per script, not per target. Run the extraction where it has to run, commit the frozen
file, and do the report step locally off that commit. That split is also what lets someone
without cluster access reproduce the report half.

## Layout (per study)

```
studies/<study>/
  Makefile                 # one PHONY target per report; `all` regenerates everything
  environment.lock         # pins the wrapper commit that PRODUCED the metrics
  CHANGELOG.md
  analysis/
    <producer>.py          # one script == one report block or one JSON producer
    <producer>.json        # carries _meta (via studies/util/_meta.py, pass study_root=)
    wandb_keys.py          # comment-only contract naming the exact W&B summary keys read
    fig_<slug>.png         # style via studies/util/plot_style.py
    reports/
      INDEX.md             # one row per report
      r<n>-<slug>.md
    provenance/            # backfill history, metric caveats, etc.
```

Shared helpers live in `studies/util/` (`_meta.py`, `plot_style.py`) — reuse them,
never copy per-study. Commit derived outputs (JSON/CSV/PNG); gitignore only caches.

## Start a report from the scaffolder

```bash
make new-report STUDY=<study> SLUG=heldout-curve TITLE="Held-out vs D" [AUTHOR=han]
```

From the studies repo root. It takes the next free `r<n>` (counting Makefile targets too, so
a deleted draft's number is never reused) and writes `analysis/reports/r<n>-<slug>.md`
(draft frontmatter, a `result-1` block), a producer stub `analysis/r<n>_<slug_>.py` (the
slug with `-` turned into `_`, e.g. `r3_heldout_curve.py`; it writes `r3_heldout_curve.json`
and already calls `build_meta(..., code=[__file__])` + `write_artifact()`), the `r<n>`
Makefile target, and the INDEX row. Fill in the TODOs; add `r<n>` to `all` once the producer runs offline.
Never copy an old report: its frontmatter may predate the schema.

## Contracts in brief (full templates: `references/report-contracts.md`)

- **Report** `r<n>-<slug>.md`: frontmatter per the schema — `status` is
  `draft | live | superseded | retracted`; `wandb_groups` are exact names, never globs;
  `inputs` are study-relative paths that exist, and commands or other studies' reports go
  in `upstream`; a report that amends an earlier one declares `corrects:` (the corrected
  report is never edited); `reproduce` is `make -C studies/<study> r<n>`. Script-owned
  `<!-- BEGIN result-N -->` / `<!-- END result-N -->` regions — anything inside is
  overwritten on regen; producers match marker pairs only.
- **JSON** (`_meta` v2): write it with
  `build_meta(produced_by, WANDB_GROUPS, study_root=STUDY, code=[__file__], inputs=[...])`
  and `write_artifact(path, obj, outputs=[fig, "<report>.md#result-N", ...])` from
  `studies/util/_meta.py`. That records digests of the producer and the `studies/util`
  modules it imports (`_meta.py` itself is excluded: it shapes only `_meta`), of every
  committed input (a gitignored cache is never an input), and of every output — so the
  checker catches a stale figure, a hand-edited result block, or an input changed since
  the run. A freeze passes `kind="freeze"` and the external `sources` it read. Reproducible
  means *committed code + input bytes*; the whole-checkout `studies_git_dirty` flag is
  informational. Without `code=`, `build_meta` returns the v1 shape unchanged — leave
  frozen v1 outputs as they are and migrate a producer when you next touch it.
- **`launch_record/results.md`**: written once after the group settles —
  W&B group + project URL, Beaker exp id + resources, timestamps, status, headline
  metric, which reports it feeds.

## Multi-agent collaboration rules

- Never hand-edit inside a `live` report's markers — rerun its producer instead.
- One PR = one report or one producer change; don't silently shift another report's numbers.
- New analysis output ⇒ declare it: add to a report's `inputs:`, add `_meta`, add
  cache patterns to `.gitignore`.
- Never squash-merge (AGENTS.md §9); state assumptions in the PR body and cite the
  W&B groups that justify a changed claim.
- Reports use Seattle time and clickable W&B links (AGENTS.md §7).

## References (read on demand)

- `references/report-contracts.md` — full templates: report frontmatter + marker
  example, `_meta` JSON, `results.md` stub, Makefile convention, `.gitignore`
  policy, enforcement checklist.
- `references/wandb-graphql-sandbox.md` — pulling W&B data from the Claude Science
  sandbox where `wandb.Api()` can't run (GraphQL + `requests` recipe, deleteRun
  gotcha).
