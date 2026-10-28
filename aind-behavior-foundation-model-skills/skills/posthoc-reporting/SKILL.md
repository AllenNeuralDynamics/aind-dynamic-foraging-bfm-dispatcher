---
name: posthoc-reporting
description: Post-hoc analysis and reporting conventions for studies — the fixed report skeleton (Question, Answer findings, Methods, Results, Caveats), frontmatter claims tied to JSON pointers, uncertainty and estimand naming, the claims check before a report goes live, report files with YAML frontmatter and BEGIN/END result markers, JSON _meta blocks, launch_record results.md stubs, Makefile targets, cache .gitignore policy, one-producer-per-artifact, and running report generation locally rather than on HPC. Use when writing, editing or regenerating analysis reports, figures, or JSON summaries from finished W&B groups.
---

# Post-hoc analysis & reporting

This skill is the source of truth for analysis/reporting conventions (the one-line
rule lives in `AGENTS.md` §12). Full artifact contracts: `references/` here. Start a
report from the scaffolder (`make new-report`); never mirror an old report, whose
layout may predate these rules.

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
6. **Every claim points at a number** — a report's findings live in its frontmatter,
   each tied to a value in its JSON; prose explains them and adds none.
7. **A fresh reader checks the text** — before a report goes live, a new agent that
   did not write it compares every claim with the numbers.

## Report content (hard rules)

Why: a survey of all 46 reports (2026-10-28) found the script-written result blocks
matched their JSON, but the hand-written text did not hold still. Prose went stale after
a block was regenerated (13 r2 kept the pre-rescue "12% fail k̂"). Claims swung on
unchanged numbers (13 r1's account of the HB gain reversed about five times in 40 hours).
A producer printed a fixed conclusion on an `n/a` comparison (01 r7). Prose quoted
numbers with no source. The verdict lived in three places that disagreed. There were five
title styles, two templates and seven ways to write uncertainty.

These rules apply to reports with frontmatter `schema_version: 2`. A version-1 report
keeps the version-1 rules; when you next change it, bump it to 2 and bring it up to these
rules. Never add a finding to `schemas/provenance-baseline.json` instead. Machine checks
for version 2 arrive with studies#62; until then, follow them by hand.

### Skeleton

Every report has these sections, in this order. `make new-report` writes them; the
template is in `references/report-contracts.md`.

1. `# r<n> — <the finding in a few words>`. The title states the result, not the topic.
2. `## Question`: one sentence, and why it matters.
3. `## Answer`: numbered findings `F1`, `F2`, …, one sentence each. Order them as an
   argument: the headline first, then what supports it, then what limits it. Never in
   the order the analysis was done. About five at most; more findings need a new report.
4. `## Methods`: a script-written facts block (mice, sessions, split, seeds, model
   versions, interval type) read from the frozen inputs, then prose on the approach. An
   approach that the study's reports share lives once in the study README; the report
   says "as in the study methods, except …".
5. `## Results`: one `### F<k>` subsection per finding, each with its own `result-N`
   block. Figure first, then table, then prose.
6. `## What it means`: the interpretation.
7. `## Caveats`: what this result does not show, and the other explanations not ruled out.
8. `## Next`: the next step, or "none".
9. `## Provenance`: W&B groups, inputs, reproduce command.

### Claims and the verdict

- Each finding is a frontmatter claim with a JSON Pointer (RFC 6901, a path to one value)
  to the number that supports it:
  `claims: [{id: F1, text: "...", pointer: "analysis/r1_fit.json#/delta_ll/mean"}]`.
- **The frontmatter claims are the only home of the verdict.** INDEX.md rows and the
  README `## Verdict` are generated from them by `make`; never edit those by hand.
- Prose explains the claims and adds no new ones. Cite a finding by its ID ("F2 shows")
  instead of retyping its number. A number you must write in prose comes from the
  report's JSON; a warn-only lint flags any number that matches nothing there.
- **A producer never prints a fixed conclusion sentence.** A verdict word is a JSON field
  that the code chooses from the numbers, and the claim points at it — as study 10 does:
  `"non-inferior" if stats["noninferior"] else "not non-inferior"`. Then an `n/a`
  comparison shows as `n/a`.

### Numbers, uncertainty and names

- Every number carries n and an uncertainty: a 95% CI (a credible interval for a Bayesian
  fit) or the SEM. Name which one, and what n counts (mice, sessions, seeds). With one
  seed, write "single seed, no error bar".
- Name the estimand in full — the exact quantity, including how it is aggregated — with
  the `CONTEXT.md` glossary name, in table headers, claims and the JSON key or metadata:
  "held-out likelihood, trial-pooled" is not "held-out likelihood, per-subject median"
  (13 r1 and 13 r2 used one label for both).
- Explain a specialised term in one sentence where it first appears, and link its
  glossary entry.
- A number from another study is a named version from the `cross_study.py` registry,
  printed with its name: "0.7282 (`gru_ceiling.v2_h128`)". A non-default version needs one
  line of why. Compare two numbers only if their mice, metric and split match, or say
  that they do not.

### Changing a report

- After a result block or its inputs change, re-read every hand-written sentence in the
  report, and every sibling report that says "same as r<n>". Then bump `prose_reviewed:`.
  The checker warns when a block changed but `prose_reviewed` did not.
- A `live` report's title and claims change only if the numbers changed, or a CHANGELOG
  entry gives the reason and Han reviews it. Rewording counts as a change.
- A draft carries `draft_reason:` (one line, e.g. "k̂ rescue pending").
- **Draft → live** needs all seven: (1) the question in one sentence; (2) Answer findings
  that match the claims; (3) at least one figure; (4) n and uncertainty on every number;
  (5) caveats; (6) a next step; (7) terms explained. Then a claims check, then Han's
  explicit go-ahead.

### Claims check

Run it before draft → live, and on any change to a live report's claims.

- Launch a **new** agent with the Agent tool (general-purpose), read-only, with
  `references/claims-check-rubric.md`. **Never `subagent_type: "fork"`:** a fork inherits
  the writing conversation, and the point is a reader who did not write the report.
- Give it files only: the report, its claims with pointers resolved, the JSON it cites,
  its figures, upstream source JSONs, the diff against the last live version, and the
  study CHANGELOG.
- Paste its summary in the PR. Answer every finding like a review thread: fix it, or say
  why not.
- Its output describes the report. It never feeds the frontmatter, INDEX or README.
- The **domain review** is a different, advisory reviewer (scientific blind spots,
  scores, ideas). Keep the two apart; mixing the jobs makes the claims check miss more.

## Never retype another study's number (hard rule)

A comparison against another study (the GRU ceiling, the best RL baseline, another study's
curve) reads the number from that study's **committed JSON**; the shared references have
readers in the studies repo's `studies/util/cross_study.py`. Pass the **source JSON** (the
helper's matching `*_SOURCE` path, not the helper module) in `build_meta(..., inputs=[...])`,
so a change to the number makes the artifact stale, and name the source in the report's `upstream`:
`{study, report}` for a result, `{study, wandb_group}` for raw data. The checker then warns
when that source is corrected, superseded or retracted (`report-upstream-stale`), or when the
declaration is missing (`report-upstream-undeclared`). Under the baseline ratchet that
warning fails CI, so a correction PR also regenerates or re-points the reports that depend on
it. A retyped constant (`GRU = {614: 0.7268}`) is invisible to every check; it is how two GRU
comparators (H128 0.7268, H256 0.7290) came to be quoted interchangeably until study 11
untangled them.

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
- **Figures (SVG) regenerate byte-identically on one plotting stack**: `figure_style`
  fixes the SVG ids, drops the date and vendors its font. On another matplotlib version
  the bytes can still differ, and the legacy PNGs of studies 01–11 differ across machines
  (~6% of pixels): `environment.lock` pins the wrapper commit that produced the *metrics*,
  not matplotlib. An image diff after regenerating elsewhere is **not** evidence that
  results changed. CI ignores image bytes; confirm via the JSON.

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
    fig_<slug>.svg         # paper style, written by studies/util/figure_style.save_figure
    slides/                # optional slide versions (style="slide"); reports embed the paper ones
    reports/
      INDEX.md             # one row per report
      r<n>-<slug>.md
    provenance/            # backfill history, metric caveats, etc.
```

Shared helpers live in `studies/util/` (`_meta.py`, `figure_style.py`) — reuse them,
never copy per-study. Commit derived outputs (JSON/CSV/SVG); gitignore only caches.

## Figures: paper-style SVG (hard rule)

Every figure is drawn with `studies/util/figure_style.py` and saved only through
`fs.save_figure(fig, HERE / "fig_<slug>.svg")`, which writes SVG (never PNG/PDF) and refuses
a file that would break in Illustrator. `make validate` and CI run `scripts/check_figures.py`
on every study except the legacy 01–11 (printed `EXEMPT` until redrawn).

- `fs.apply_paper_style()`: Nature style, 5–7 pt Arial, 8 pt bold panel letters
  (`fs.panel_label`), Okabe–Ito colours.
- Draw at print size and **size by content, not by column**: ≈15 mm per bar or violin,
  35–45 mm per scatter panel, + 18 mm per y axis (`fs.mm(fs.fit_width(groups=4), 55)`).
  At most 183 mm wide; any height (a prototype grid can be tall).
- No math text (`$...$`, default log ticks): it splits into one letter per text box. Use
  Unicode (σ, ρ, ², −, ≈), `fs.plain_log_axis(ax)` and `fs.num(x, "+.3f")` for a true minus.
- Slide versions: `fs.apply_slide_style()` and `save_figure(..., style="slide")`, in an
  `analysis/slides/` directory.
- Leave `plot_style.py` byte-for-byte alone: `build_meta` digests every imported
  `studies/util` module, so editing it marks the legacy producers' artifacts stale.

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
Add `tags: {model: [...], question: [...]}` for what the report does (values from the studies
repo's `schemas/tags.yaml`; a model only if this report trains or fits it). The atlas filters a
study by its own tags plus its reports', so a report doing a generative run or a methods check
is found even when the study's tags do not name it. A producer that writes the whole report
must write the tags too, or regeneration drops them.
Never copy an old report: its frontmatter may predate the schema.

## Contracts in brief (full templates: `references/report-contracts.md`)

- **Report** `r<n>-<slug>.md`: frontmatter per the schema — `status` is
  `draft | live | superseded | retracted`; `wandb_groups` are exact names, never globs;
  `inputs` are study-relative paths that exist, and commands or other studies' reports go
  in `upstream`; a report that amends an earlier one declares `corrects:` (the corrected
  report is never edited); `reproduce` is `make -C studies/<study> r<n>`. Script-owned
  `<!-- BEGIN result-N -->` / `<!-- END result-N -->` regions — anything inside is
  overwritten on regen; producers match marker pairs only. `schema_version: 2` adds
  `claims`, `draft_reason` (drafts) and `prose_reviewed` (see "Report content").
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

- `references/report-contracts.md` — full templates: the version-2 report skeleton and
  frontmatter, `_meta` JSON, `results.md` stub, Makefile convention, `.gitignore`
  policy, enforcement checklist.
- `references/claims-check-rubric.md` — what the claims check reviewer reads, checks and
  returns; give it to the new agent as its instructions.
- `references/wandb-graphql-sandbox.md` — pulling W&B data from the Claude Science
  sandbox where `wandb.Api()` can't run (GraphQL + `requests` recipe, deleteRun
  gotcha).
