---
aliases:
  - repo split plan
  - split studies
  - aind-dynamic-foraging-bfm-studies
tags:
  - planning
  - migration
  - one-shot
status: approved
---

# Repo-split plan: extract `studies/` into `aind-dynamic-foraging-bfm-studies`

> **Status:** approved 2026-07-16, not yet executed. Re-reviewed 2026-09-03:
> the architectural decision still stands, but execution is blocked on the
> pre-split hardening and branch freeze below. Originally written 2026-06-30 after the
> `posthoc-analysis` standard-structure migration (ending at commit `122abfd`,
> later merged to `main`); revised 2026-07-16 when the split was approved and
> amended 2026-09-03 after a fresh repository audit.

## TL;DR

Split `aind-dynamic-foraging-bfm-dispatcher` at the framework/application seam:

- **`aind-dynamic-foraging-bfm-dispatcher`** (this repo, stays) — launchers (`code/`),
  framework docs (`docs/`), Docker env, CO capsule metadata, root `AGENTS.md`.
- **`aind-dynamic-foraging-bfm-studies`** (new sibling repo) — everything currently under
  `studies/`, keeping the `studies/` path prefix (see Decision below), one
  study per subfolder, each self-contained per `docs/study-organization.md`.

All eight studies (`01`–`08`) move in one pass. Preserve per-file history via
`git filter-repo`.

Do **not** execute the older numbered steps verbatim. The 2026-09-03 amendment
adds four gates that must land first: dual-repository provenance, a
self-contained studies `AGENTS.md`, explicit CI ownership, and a freeze of all
branches that touch `studies/`. Build and verify the extracted repository
locally before its first push; never publish a temporarily broken `main`.

## Decision (2026-07-16): split now

The 2026-06-30 discussion below leaned "not yet," with the explicit revisit
trigger *"when there is a second study."* Re-measured 2026-07-16, every
load-bearing fact has flipped:

- **Five studies exist** (`01-gru-scaling-law` … `05-disrnn-scaling-law`),
  each with its own `Makefile`, `analysis/`, and reports.
- **Studies dominate the repo.** Last 3 months: 255 non-merge commits touched
  `studies/` vs ~175 file-touches under `code/`. Tracked `studies/` content is
  ~43 MB vs ~3 MB for everything the runtime needs. Framework changes are
  drowned out, and every study commit advances `main`, so pinned
  `DISPATCHER_REF` SHAs churn for reasons meaningless to the runtime. Beaker
  jobs re-clone the dispatcher at startup (`entrypoint.sh`), so the split also
  keeps that clone permanently light.
- **Coupling is thin.** Of the 255 study commits, only 11 (~4%) also touched
  `code/` or `environment/` — cross-repo PRs will be rare. Verified again:
  zero Python imports from `studies/**` onto `code/**`; the only references
  are shell-path launch commands in READMEs/notes and comment lines in sweep
  YAMLs. (Study 04's analysis scripts `sys.path`-insert the *wrapper* sibling
  clone — cross-repo side-by-side consumption is already the working
  convention.)
- **Packaging `code/` is NOT a prerequisite** (reverses the 2026-06-30
  sequencing note below, restoring the original Non-goals bullet). The real
  interface is a CLI that takes the `sweep.yaml` path as an argument;
  `launch_hpc.py` and `launch_beaker_resumable.py` only parse the
  `studies/<study>/variants/<variant>` components out of that path to derive
  the W&B group. A sibling clone
  (`python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_....py studies/...`) works
  today with no code change. Packaging remains a good follow-up, not a
  blocker.
- **Path prefix (was Open Question Q1): keep `studies/` — resolved, and not
  just for lower churn.** The launchers' study/variant derivation searches for
  the literal `studies` path component (`launch_hpc.py:218`,
  `launch_beaker_resumable.py:135`); a flat layout silently breaks W&B group
  naming unless both launchers are patched. Prefixed it is; flatten later only
  together with a launcher change.

Dated observation (2026-07-16): there were then no open PRs and no local
study commits ahead of `origin/main`. This is historical context only; the
execution-time branch audit below is authoritative.

### Re-verified 2026-09-01 (deltas since the 2026-07-16 decision)

The decision to split stands. Three of its inputs have moved; the counts above
are left as the dated observations they were.

- **Eight studies now exist**, not five (`01-gru-scaling-law` …
  `08-hb-vs-gru-heldout`). Seven carry a `Makefile` (`01`–`07`);
  `08-hb-vs-gru-heldout` has only `README.md`, `figures/` and `variants/`, so it
  is not yet normalized to the posthoc-reporting layout and will not satisfy the
  `make all` gate in the verification checklist below.
- **Prereq 1 no longer held.** Dispatcher PR #73 contained study work that
  would have been dropped if it were absent from dispatcher `main` at filter
  time. Wrapper PR #65 mattered only because the studies depended on its
  coupled change/pinned revision; filtering dispatcher history cannot itself
  carry or drop wrapper commits. These PR numbers are dated observations, not
  execution-time status.
- **The target repo name is gated on the rename.** #74 renames the project
  identity `disrnn` -> `dynamic-foraging-bfm` and is sequenced *before* this split, so the
  new repo is created as **`aind-dynamic-foraging-bfm-studies`** and the cross-repo shell
  path below becomes `../aind-dynamic-foraging-bfm-dispatcher/code/...`. Rename first:
  this plan writes that dispatcher path into every study README, `notes.md` and
  sweep-YAML comment, so splitting first means sweeping the same files twice —
  the second time in a new repo where the rename's CI guardrail does not yet
  exist. The rename has since landed (ADR-0007), so the repo names below are
  the real ones.

### Re-reviewed 2026-09-03 (execution-hardening amendment)

The split remains the right seam, and the evidence is stronger: the current
tree has 670 tracked files under `studies/` (~22 MB); over the preceding three
months, 368 non-merge commits touched `studies/`, 104 touched `code/`, and only
13 touched both. There are still no Python imports from `studies/**` into the
dispatcher launchers. The active cross-repo surface is paths, configuration,
and provenance rather than Python APIs.

The runbook nevertheless needed correction before execution:

- **Provenance changes meaning after extraction.** `studies/util/_meta.py`
  currently runs `git rev-parse HEAD` and writes that value as
  `dispatcher_git_sha`. In the new repo the same call returns the *studies*
  SHA, silently mislabelling it as dispatcher provenance. The Beaker launcher
  likewise records the dispatcher SHA but not the SHA of the external repo
  containing the sweep. Dual-repo provenance is therefore a prerequisite,
  specified below.
- **A sibling `AGENTS.md` link is not inheritance.** Agents working from a
  studies-only clone are not guaranteed to load the dispatcher's file. The
  new repo must carry its own concise, enforceable safety, provenance, issue,
  and PR-merge rules, with links to the dispatcher for detailed playbooks.
- **CI and the skills pack now exist and know about studies.** Dispatcher CI
  stays with the launcher tests. The new repo gets study validation CI. The
  bundled `aind-behavior-foundation-model-skills` pack stays in dispatcher but
  its codebase map, study/reporting paths, and launch examples must understand
  the sibling studies checkout.
- **Do not rewrite historical records.** Active READMEs, scripts, templates,
  and comments move to sibling-dispatcher paths; immutable
  `launch_record/**` files retain the paths and refs that were true when the
  launch occurred.
- **The target repo is not yet present.** An authenticated `git ls-remote`
  check on 2026-09-03 returned “repository not found.” Repository creation and
  visibility are explicit user-controlled steps; do not hardcode `--public`.
- **The source is not frozen.** As of 2026-09-03, remote branches for studies
  07 and 08 contain study commits not on `main`. Re-check live PRs and branches
  immediately before extraction; branch names and counts in this paragraph are
  observations, not a durable allowlist.

## Historical discussion (2026-06-30; non-operative)

> The “not yet” recommendation and “package first” sequencing below were
> superseded on 2026-07-16. Preserve this section as decision history, but do
> not use it as an execution checklist. The analysis-contract findings remain
> relevant.

**Lean at the time: not yet.** The split is a defensible long-term direction but
premature today, and separating the *analysis* layer — not the whole `studies/`
tree — is the sharper cut. Reasoning, from a 2-people-plus-AI project's point of
view (don't over-engineer):

- **Only one study exists.** Standing up a second repo to organize a single
  folder pays multi-repo overhead (side-by-side clones, cross-repo PRs,
  container entrypoint changes, a 3-way version matrix for provenance) with no
  offsetting benefit yet. Revisit when there is a *second study* or a *second
  contributor who only touches studies* — either gives the seam a concrete job.
- **Package before you split.** The launch-side coupling
  (`studies/*/variants/` → `code/` launchers) is a *shell path* (`python
  ../aind-dynamic-foraging-bfm-dispatcher/code/launch_*.py`), not a real interface. That is
  fragile in containers and unversioned. The clean boundary only exists once
  `code/` is a pip-installable package the studies repo depends on with a pinned
  version — exactly how the study already consumes the wrapper via
  `environment.lock`. Splitting first leaves you in the fragile relative-path
  phase indefinitely. So: **package `code/` is a prerequisite, not a non-goal**
  (supersedes the "Non-goals" bullet below for sequencing purposes).

### The analysis layer is the most-separable piece — and it is already split

The "analysis" concept is *already* bisected across the two existing repos, along
a natural axis, with **W&B as the boundary**:

- **Producer (per-run, in the wrapper):**
  `aind-dynamic-foraging-bfm-wrapper/code/post_training_analysis/` (`generative_analysis.py`,
  `heldout_finetuning.py`, `likelihood_*`, `baseline_rl_analysis.py`,
  `embedding_space_analysis.py`), invoked in the capsule via
  `run_analysis.py <subcommand>` (generative / from-histories /
  likelihood-comparison / likelihood-advantage / embedding / baseline-rl /
  finetune). Needs the trained model + data + JAX/GPU. Writes quantitative
  summaries into each run's **W&B summary + logged artifacts**.
- **Consumer (cross-run, in the study):** `studies/data-scaling-law/analysis/`
  reads those summary keys/artifacts back via `wandb.Api()`, aggregates across
  cells (D × seed × subject), fits curves, renders `reports/r*.md`. It does
  **not** import `post_training_analysis`; verified it has **zero references to
  `code/`** either. Its only inbound contract is a handful of W&B key strings.

So the study `analysis/` is the single most extractable component (its input is
the cloud, keyed by group name). But it is also the most *semantically* bound to
its study — it **is** the study's answer — so pulling it into its own repo fights
the "a study is self-contained" principle in `study-organization.md`. Net: leave
it in place; the fragility to fix is the *contract*, not the location.

### The wrapper↔study analysis contract (measured, then hardened in place)

Measured 2026-06-30: the consumer hardcodes **7 distinct W&B summary keys**; the
key *vocabulary* has been stable in wrapper history (churn is in analysis
internals, not key names). The real hazard was the **failure mode**: reads used
`summary.get(key)` then `if None: continue`, so a renamed/dropped key **silently
drops runs** — a report would shrink and its numbers shift with no error.

Right-sized fixes applied (no shared schema package, no CI validator, no
machine-readable manifest — those exceed the benefit at this scale):

1. **`analysis/wandb_keys.py`** — single source of truth for the 7 keys
   (builders + constants) plus a `require()` helper. The whole wrapper-contract
   surface is now one greppable file to review when bumping the wrapper pin.
2. **Loud-on-schema-break guards** — `generative_match.py` and `nxd_scaling.py`
   still skip individual partial runs, but now `raise KeyError` if *all* cells
   lack the required key (the rename signal), pointing at `wandb_keys.py`.
3. **`_meta.wrapper_git_sha`** — every analysis JSON now stamps the wrapper
   commit (read from `environment.lock`) that produced the keys, alongside
   `dispatcher_git_sha`. Closes the producer-side provenance gap.

`analysis/watch_nxd_d30.py` (a defunct one-off watcher that only presence-checks
readiness, no report corruption risk) was intentionally left untouched.

Because that contract is small (7 keys) and stable, it is genuinely fine to keep
the analysis layer where it is; these guards make the W&B boundary safe to cross
whether or not analysis ever becomes its own repo.

## Prerequisites (hard gates; do these before extracting)

1. **Track the migration.** Confirm an existing repo-split issue or file one on
   the dispatcher, assign an owner, put it on project 184, and make its “Done
   when” checklist match this runbook. Migration PRs use `Refs #<issue>`;
   close the issue only after the final smoke verification.
2. **Freeze all study work, not only open PRs.** Enumerate live PRs and every
   remote branch whose diff against `origin/main` touches `studies/**`. Merge or
   explicitly abandon each branch; do not assume a remote branch without a PR
   is disposable. Abandonment requires the owner's confirmation and a manifest
   entry with the branch tip SHA and reason, followed by remote-branch deletion.
   Never merge a PR without explicit user confirmation, and never squash-merge.
   -> verify: no surviving remote branch contains a study commit absent from
   `origin/main`; every deliberately abandoned tip is recorded in `MIGRATION.md`.
3. **Extract the current remote truth.** Fetch, require a clean worktree, check
   out the exact `origin/main` SHA in the scratch clone, and record that full
   SHA as `source_dispatcher_commit`. Do not extract from a stale local `main`.
   -> verify: `git rev-parse HEAD` equals
   `git ls-remote origin refs/heads/main | cut -f1`.
4. **Land dual-repository provenance before extraction.** The dispatcher
   launchers and study helpers must implement the contract below, with tests.
   This commit must be on `main` so `git filter-repo` carries the studies half
   into the new repository.
5. **Allocate CI, skills, and agent rules.** Decide the exact files that stay in
   dispatcher versus seed the studies repo. Update the skills pack to understand
   both roots. Draft a self-contained studies `AGENTS.md`; do not rely on a
   sibling file being automatically loaded.
6. **Confirm no import-time coupling** from `studies/**` onto `code/**`.
   -> verify: `rg -n "from (launchers|code|dispatcher)|import (launchers|code|dispatcher)" studies/`
   is empty apart from comments or prose.
7. **Resolve the study-08 verification gap.** Either add its normalized
   `Makefile` before the freeze or record an explicit, temporary CI exception
   with an owner and follow-up issue. Absence must not make the all-studies loop
   fail ambiguously.
8. **Choose repository visibility explicitly.** Match the source repository by
   default. Creating a public repository requires explicit user confirmation;
   never embed `--public` as an assumed migration step.

## Target state

### `aind-dynamic-foraging-bfm-dispatcher/` (framework, unchanged root)

```text
aind-dynamic-foraging-bfm-dispatcher/
├── AGENTS.md                    # framework behaviour rules
├── README.md                    # pointer to aind-dynamic-foraging-bfm-studies in "Studies" section
├── docs/                        # posthoc-analysis, study-organization, beaker-playbook
├── code/                        # launchers (launch_beaker*.py, launch_hpc.py, ...)
│   ├── beaker/                  # shared beaker templates
│   ├── config/                  # shared configs consumed by launchers at runtime
│   ├── hpc/                     # shared SLURM helpers
│   └── util.py
├── environment/                 # Dockerfile
├── .codeocean/                  # CO capsule metadata
└── (no studies/)
```

### `aind-dynamic-foraging-bfm-studies/` (new sibling repo)

```text
aind-dynamic-foraging-bfm-studies/
├── AGENTS.md                    # self-contained critical rules + canonical links
├── README.md                    # study index + side-by-side clone instructions
├── .github/workflows/ci.yml     # offline regeneration/provenance checks
├── requirements-dev.txt         # dependencies required by studies CI
├── studies/                     # prefix KEPT — see resolved Q1
│   ├── 01-gru-scaling-law/
│   │   ├── analysis/
│   │   ├── variants/
│   │   ├── Makefile
│   │   ├── environment.lock
│   │   ├── CHANGELOG.md
│   │   └── README.md
│   ├── 02-gru-scaling-law-ignore/
│   ├── 03-disrnn-beta-scan/
│   ├── 04-gru-vs-disrnn-embedding-recovery/
│   ├── 05-disrnn-scaling-law/
│   ├── 06-disrnn-operating-point-at-scale/
│   ├── 07-gru-timing-inputs/
│   ├── 08-hb-vs-gru-heldout/
│   └── util/
└── .gitignore                   # ignore W&B pulls/caches; keep curated outputs
```

Path prefix trade-off (resolved 2026-07-16, see Decision above and Q1):
- **Flat** (`01-gru-scaling-law/` at root): tighter for a repo whose sole
  purpose is studies — but **breaks the launchers' W&B group derivation**,
  which searches for the literal `studies` path component.
- **Prefixed** (`studies/01-gru-scaling-law/`): zero-diff paths for docs,
  Makefiles, launchers, and tools that hardcode the `studies/` prefix; easier
  `filter-repo`. **Chosen.** Flatten later only together with a launcher
  change.

## What moves, what stays

| Path | Action |
|---|---|
| `studies/**` (tracked files only) | -> `aind-dynamic-foraging-bfm-studies/studies/**` |
| `code/**` | stays in dispatcher |
| `docs/**` | stays in dispatcher; studies repo links back |
| `environment/**` | stays in dispatcher |
| `.codeocean/**` | stays in dispatcher |
| `.github/workflows/ci.yml` | dispatcher launcher CI stays; studies repo gets a new study-validation workflow |
| `aind-behavior-foundation-model-skills/**` | stays in dispatcher; update paths and two-repo guidance in the same migration |
| `AGENTS.md` | stays; studies repo gets a **self-contained** critical-rule subset plus canonical links |
| `README.md`, `LICENSE`, `CODE_OF_CONDUCT.md` | copy-forward (fresh, not filter-repo'd) into studies repo |
| `.gitignore` | copy-forward the studies-relevant rules; drop dispatcher-only rules |
| `artifacts/` | untracked already; regenerated locally in either repo |

## Cross-repo runtime dependency

After the split, study launchers still invoke the dispatcher's launchers by
shell. The conventional layout the two READMEs assume is **side-by-side clones**:

```text
~/code/
├── aind-dynamic-foraging-bfm-dispatcher/
└── aind-dynamic-foraging-bfm-studies/
```

Study docs are updated so that:

```text
# before (single repo, from repo root)
python code/launch_beaker_resumable.py ...

# after (from studies repo root, dispatcher cloned as sibling)
python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_beaker_resumable.py ...
```

Ditto for content references (`code/config/model/gru_scaling.yaml` becomes
`../aind-dynamic-foraging-bfm-dispatcher/code/config/model/gru_scaling.yaml`).

Files to update in the studies repo after extraction: **regenerate the list at
execution time, excluding immutable launch records** —

```bash
rg -l 'code/launch|code/config|code/hpc|code/beaker' studies/ \
  --glob '!**/launch_record/**'
```

The 2026-09-03 audit found 32 matching files when historical launch records are
included, up from ~20 in the 2026-07-16 count. Rewrite only active READMEs,
`notes.md`, scripts, live sweep/experiment templates, and current error/help
strings. Do **not** rewrite `launch_record/**`: an old path in a frozen record is
part of the record of what actually ran. After the active rewrite, inspect all
remaining matches and classify them as immutable history or a missed live
reference; do not suppress the check wholesale.

Long-term (out of scope for this split): promote the launchers to a proper
Python package with console-script entrypoints so the shell path drops out
entirely. Track as follow-up.

## Cross-repo provenance contract (must land before extraction)

A split is successful only if a launch can still answer “which study definition
and which runtime control plane produced this run?” without inferring from a
mutable branch or sibling checkout. Use these exact field names:

| Surface | Required fields | Shape and source |
|---|---|---|
| Beaker/HPC launch record | `studies_git_commit`, `studies_git_branch`, `studies_git_dirty` | full 40-character SHA plus informational branch and boolean dirty state, resolved from the Git root containing the supplied sweep |
| Beaker/HPC launch record | `dispatcher_git_commit`, `dispatcher_git_branch`, `dispatcher_git_dirty` | full SHA, branch and dirty state from the launcher checkout |
| Beaker/HPC launch record | `wrapper_git_commit`, `foraging_models_git_commit` | full SHAs resolved by the launcher from the submitted runtime refs |
| W&B `meta.*` | `meta.studies_commit`, `meta.dispatcher_commit`, `meta.wrapper_commit`, `meta.foraging_models_commit` | the same four immutable full SHAs; branches are optional display metadata, never identity |
| Analysis JSON `_meta` | `studies_git_sha`, `studies_git_dirty` | analysis-producer checkout HEAD and boolean dirty state |
| Analysis JSON `_meta` | `source_dispatcher_git_shas`, `source_wrapper_git_shas`, `source_foraging_models_git_shas` | sorted unique arrays resolved from the source launch records; use `[]` plus an explicit provenance finding when unavailable, never a guessed scalar |

Additional rules:

- A dirty studies checkout is rejected for a scientific launch unless an
  explicit diagnostic-only override is recorded in the launch record. Analysis
  may run dirty during development, but `_meta.studies_git_dirty` must say so.
- After extraction, `studies/util/_meta.py` must never write its own repository
  HEAD under `dispatcher_git_sha`. Keep already-committed legacy outputs
  unchanged; new or intentionally regenerated outputs use the new schema.
- A study-wide `.dispatcher_pin` may be a convenient default for a new launch,
  but it is not historical provenance because variants can use different
  dispatcher commits. The per-launch resolved refs are authoritative.
- Add unit tests that place the sweep in a second temporary Git repository and
  assert that studies and dispatcher SHAs are distinct and correct. Test both
  Beaker and HPC record paths and the W&B metadata injection.

The no-submit smoke test and the optional real launch below inspect these exact
fields rather than merely running `--help`.

## Migration mechanics

### Step 0 — Land the pre-split hardening

On a dispatcher branch linked to the tracking issue:

1. implement the exact dual-repository provenance schema above in the Beaker
   and HPC launchers plus `studies/util/_meta.py`;
2. add tests using a sweep in a separate temporary Git repository;
3. update the bundled skills so they can locate a sibling studies repo while
   remaining correct before the split; and
4. define the new repo's root files and CI in this plan, but create them only in
   the scratch extract during Step 3—do not stage them under dispatcher paths.

Open a PR with `Refs #<issue>`. Do not merge it automatically: request Copilot
review, resolve every thread, then stop for explicit user confirmation. Merge
with a merge commit, never squash.

### Step 1 — Freeze branches and snapshot remote truth

Immediately before extraction, fetch all refs and enumerate both same-repo and
fork PR heads. Exclude `origin/HEAD` and `origin/main` from the branch loop:

```bash
git fetch origin --prune
gh pr list --state open --limit 500 \
  --json number,headRefName,headRefOid,isCrossRepository,files
git for-each-ref --format='%(refname:short) %(objectname)' refs/remotes/origin/ |
while read -r ref sha; do
  case "$ref" in origin/HEAD|origin/main) continue ;; esac
  git log --oneline origin/main.."$sha" -- studies/
done
git ls-remote origin refs/heads/main
```

For cross-repository PRs, inspect `headRefOid` through the GitHub API or fetch
the PR ref explicitly; do not assume it appears under `refs/remotes/origin/`.
Merge or deliberately abandon every study-touching branch as specified in the
prerequisites, then repeat the audit.

Record the final full SHA and the AGENTS revision in a temporary operator note;
Step 3 promotes them into tracked `MIGRATION.md`:

```bash
SOURCE_DISPATCHER_COMMIT=<full-origin-main-sha>
AGENTS_SOURCE_COMMIT=$(git log -1 --format=%H origin/main -- AGENTS.md)
```

No study commits land in dispatcher between this snapshot and the removal PR;
urgent study work waits or targets the extracted repository.

### Step 2 — Extract `studies/` with full history

Use a fresh temporary directory and a pinned, isolated `git-filter-repo`
installation. Select and record the reviewed tool version at execution time:

set -euo pipefail
MIGRATION_ROOT=$(mktemp -d)
FILTER_REPO_VERSION=<reviewed-version>
python3 -m venv "$MIGRATION_ROOT/venv"
"$MIGRATION_ROOT/venv/bin/pip" install "git-filter-repo==$FILTER_REPO_VERSION"
git clone --no-local https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher.git "$MIGRATION_ROOT/extract"
cd "$MIGRATION_ROOT/extract"
git checkout -B main "$SOURCE_DISPATCHER_COMMIT"
git ls-tree -r --full-tree "$SOURCE_DISPATCHER_COMMIT" studies/ > "$MIGRATION_ROOT/source-tree.tsv"
find studies -type f -path '*/launch_record/*' -exec shasum -a 256 {} + | sort > "$MIGRATION_ROOT/source-launch-records.sha256"
"$MIGRATION_ROOT/venv/bin/git-filter-repo" --path studies/
git ls-tree -r --full-tree HEAD studies/ > "$MIGRATION_ROOT/extracted-tree.tsv"
diff -u "$MIGRATION_ROOT/source-tree.tsv" "$MIGRATION_ROOT/extracted-tree.tsv"
```

The tree manifests must match before any assembly edits: modes, blob IDs, and
paths prove every tracked study file crossed the boundary. `git filter-repo`
removes the source remote intentionally; do not add the target remote yet.

Also verify `git log --follow` for representative files from multiple studies,
including study 01's `analysis/rl_baseline.py` back through `1e30716`.

### Step 3 — Assemble a complete studies repository locally

Before any push, add:

- `MIGRATION.md` containing the source repo URL, `SOURCE_DISPATCHER_COMMIT`,
  `AGENTS_SOURCE_COMMIT`, pinned filter-repo version, extraction time, operator,
  abandoned branch tip SHAs/reasons (if any), the one initial direct-push
  exception, and the reserved annotated tag name `studies-split-v1`;
- `migration/source-tree.tsv` and
  `migration/source-launch-records.sha256` copied from Step 2;
- a self-contained `AGENTS.md` using the amended template below;
- a README with all eight studies and side-by-side clone instructions;
- `LICENSE`, `CODE_OF_CONDUCT.md`, and a studies-only `.gitignore`;
- `requirements-dev.txt` and `.github/workflows/ci.yml`; and
- provenance fixtures or tests that belong with the studies side.

Rewrite only active cross-repo references. Use searches that cover the current
and retired repo names, absolute paths, and generic launcher/config paths:

```bash
rg -n 'aind-(disrnn|dynamic-foraging-bfm)-dispatcher|python[[:space:]]+code/|(^|[[:space:]`])code/(launch|config|hpc|beaker)' studies/ \
  --glob '!**/launch_record/**'
```

Run the same search including `launch_record/**`; classify every remaining
match in `MIGRATION.md` as immutable history or intentional. Preserve all
launch records byte-for-byte. Keep shared `code/config/` and `code/beaker/`
templates in dispatcher; exact study-specific sweep/experiment manifests stay
under their variants. Resolve study 08's Makefile exception.

Commit the assembled state with source SHAs in both `MIGRATION.md` and the
commit message. Focused path/CI commits may follow. Once all content and checks
are complete, call the current HEAD the **assembly commit**. Add its full SHA to
`MIGRATION.md` in one final local metadata commit; this is not self-referential
because the recorded SHA names its parent. The first published `main` contains
both commits and must be internally usable. The annotated tag created in Step 5
identifies the exact initial published tip.

### Step 4 — Verify locally before repository creation

Run the full checklist below. Generate a machine-readable assembly diff:
compare the extracted tree manifest with the assembled tree, then record every
new or changed path in `migration/assembly-changes.tsv` with one of these
reasons: `root-seed`, `active-path-rewrite`, `provenance-schema`,
`study08-normalization`, or `ci`. Fail on any unclassified path.

For artifacts, provide a comparison script that removes only the explicit
metadata allowlist (`produced_at_pt` and the documented provenance-schema
transition), then compares JSON structurally, CSV cells exactly, report
`BEGIN/END` blocks exactly, and image checksums exactly. Numeric or narrative
differences are not migration noise.

Also run external-sweep tests, render a tiny Beaker spec with `--no-submit`,
and exercise the HPC metadata helper without submission. Inspect the exact
schema table fields and confirm the studies/dispatcher SHAs differ correctly.
Do not create the GitHub repository while any required check is red.

### Step 5 — Create the GitHub repository and publish the verified state

The user chooses visibility explicitly. Prefer matching the source repository;
public visibility requires explicit confirmation.

```bash
gh repo create AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies --<confirmed-visibility>
git remote add origin git@github.com:AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies.git
git push -u origin main
```

The initial push is the one documented exception to “all tracked changes
through a PR”: an empty repository has no protected base branch. Immediately
tag the exact published tip, then enable dispatcher-equivalent protection:

```bash
git tag -a studies-split-v1 -m "Initial studies split from $SOURCE_DISPATCHER_COMMIT"
git push origin studies-split-v1
```

`MIGRATION.md` names the operator, assembly commit, and reserved tag name; the
annotated tag supplies the non-self-referential identity of the published tip.
Every subsequent change uses a PR. Verify visibility, tag target, branch
protection, studies CI, and a fresh clone before accepting new work.

### Step 6 — Remove `studies/` from dispatcher through a PR

Only after the studies repository is published and verified, fetch current
remote truth and prove that `studies/**` has not changed since extraction. If
this diff is non-empty, stop and restart at Step 1 with a new source boundary:

```bash
set -euo pipefail
git fetch origin --prune
git merge-base --is-ancestor "$SOURCE_DISPATCHER_COMMIT" origin/main
test "$(git rev-list --count "$SOURCE_DISPATCHER_COMMIT"..origin/main -- studies/)" -eq 0
git diff --exit-code "$SOURCE_DISPATCHER_COMMIT"..origin/main -- studies/
git switch -c chore/remove-studies-after-split origin/main
git rm -r studies/
```

Update dispatcher README/AGENTS pointers and finalize the skills-pack changes
so codebase-map distinguishes dispatcher from studies. Keep launcher CI in
dispatcher and run it after deletion.

Open a PR with `Refs #<issue>`, the `studies-split-v1` tag and assembly commit, and an explanation
that dispatcher history is not rewritten. **Stop before merge and ask the user
for confirmation.** If approved, merge with a merge commit; never squash.

### Step 7 — Update operational checkouts and verify launch behavior

Clone studies alongside dispatcher wherever launches are initiated. Do not add
it to Code Ocean or Beaker runtimes without evidence that an entrypoint reads
study files after submission.

The required migration gate is the no-submit Beaker render plus HPC metadata
test from Step 4. A minimal real launch is stronger but requires separate user
authorization. If authorized, confirm the W&B run and committed launch record
contain every exact provenance field. If authorization is declined or deferred,
record that fact, create an owned follow-up issue for the live check, and allow
the migration issue to close after all non-live acceptance criteria pass.

Reconcile every issue checkbox before closure; do not use a PR auto-close while
live verification remains outstanding.

## Verification checklist (end-to-end)

Run from a fresh clone of `aind-dynamic-foraging-bfm-studies` using the same
offline commands CI uses. Network-backed W&B pulls remain separate producer
targets, never CI prerequisites.

```bash
python3 -m pip install -r requirements-dev.txt
for s in studies/0*/; do
  if test -f "$s/Makefile"; then make -C "$s" all; else echo "EXEMPT $s"; fi
done
```

- [ ] `MIGRATION.md` records the source repository, exact dispatcher/AGENTS
      SHAs, filter-repo version, operator, assembly commit, initial-push/tag
      exception, and every abandoned branch disposition.
- [ ] Pre-edit source/extract tree manifests match exactly; every later changed
      or added path appears in `migration/assembly-changes.tsv` with an allowed
      reason.
- [ ] Representative `git log --follow` checks across studies retain pre-split
      history, including study 01 back through `1e30716`.
- [ ] `migration/source-launch-records.sha256` verifies every historical
      `launch_record/**` file byte-for-byte.
- [ ] Every active cross-repo path resolves; every remaining old/absolute path
      is classified as immutable history or intentional.
- [ ] `make all` exits 0 for studies `01`–`07`; study 08 either passes its own
      Makefile or has a visible, owned temporary CI exception.
- [ ] The normalized artifact comparator reports no unclassified JSON, CSV,
      report-block, image, numeric, or narrative difference.
- [ ] New analysis JSON contains `_meta.studies_git_sha`,
      `_meta.studies_git_dirty`, and the three exact sorted source-SHA arrays;
      it never relabels the studies HEAD as dispatcher provenance.
- [ ] External-sweep tests and the Beaker `--no-submit` record contain all exact
      launch-record and W&B fields in the provenance schema table; Beaker and
      HPC derive the same values.
- [ ] Studies CI passes in a fresh clone; dispatcher CI passes after removal.
- [ ] Repository visibility, `studies-split-v1` tag target, assembly SHA, and
      `main` protection match the recorded decisions.
- [ ] A real launch either passes with all four W&B source commits or is deferred
      to a linked, owned follow-up issue with explicit user direction.

## Studies-repo `AGENTS.md` template

The new file is deliberately self-contained for critical rules. Links provide
detail; they do not substitute for rules an agent must obey from a studies-only
checkout.

```markdown
# AGENTS.md — aind-dynamic-foraging-bfm-studies

Scientific study definitions, launch manifests, frozen inputs, analysis, and
reports for dynamic-foraging-bfm. The sibling dispatcher owns launch/runtime
control-plane code.

## Required workflow

- Every tracked change starts with an assigned GitHub issue on project 184.
- One scientific question per `studies/NN-{model}-{purpose}/`; conditions are
  self-contained variants, not new studies. Never renumber an accession.
- Treat `launch_record/**` as immutable history. Do not rewrite old paths,
  refs, names, or results to match the current tree.
- Scientific launches use clean checkouts and immutable full SHAs for studies,
  dispatcher, wrapper, and foraging-models. Always provide the required label
  and scientific note; verify schedulable capacity before a large launch and
  use only approved hub clusters.
- Reports are code: regenerate from committed frozen inputs, keep W&B pull
  caches ignored, and inspect numeric/artifact diffs before claiming a result.
- Use Conventional Commits. Open a PR for changes; never squash-merge and never
  merge any PR without explicit user confirmation. Resolve every review thread.
- Never global-replace `disrnn`: it remains a model-family name and is also
  frozen in historical records.

## Repository relationship

Launch commands assume this side-by-side layout:

    aind-dynamic-foraging-bfm-dispatcher/
    aind-dynamic-foraging-bfm-studies/

From this repo, invoke launchers through
`../aind-dynamic-foraging-bfm-dispatcher/code/`. The detailed operational rules
and skills live in the dispatcher:

- https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/AGENTS.md
- https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/tree/main/aind-behavior-foundation-model-skills
- https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher/blob/main/docs/study-organization.md

Load `study-conventions`, `posthoc-reporting`, and the relevant launch skill
before creating a study, producing a report, or submitting work.
```

## Non-goals (explicitly out of scope)

- **Do not** promote the `code/` launchers to a pip-installable package
  in this migration. That's a separate refactor; do it after the split
  proves stable.
- **Do not** split individual studies into their own repos. Eight studies
  still share conventions, utilities, and ownership; revisit only when a clear
  ownership or release boundary appears, not at an arbitrary study count.
- **Do not** move `docs/` into the studies repo. Framework conventions
  belong with the framework; studies link back.
- **Do not** rewrite dispatcher history (only `studies/` is filter-repo'd,
  and only in the extract clone).

## Resolved decisions and execution-time checks

1. **Path prefix — resolved:** keep `studies/`. Both launchers derive W&B
   study/variant metadata from that literal component. Flattening is a later
   launcher-interface change, not part of this migration.
2. **Code Ocean — resolved:** launcher/control-plane only. It does not consume
   `studies/`; do not add a studies clone without evidence that a runtime path
   needs it.
3. **Shared dispatcher templates — resolved:** `code/config/` and
   `code/beaker/` stay in dispatcher. Exact study-specific sweep and experiment
   manifests stay under the variant. Move a template only when it stops being
   shared, through a separate reviewed change.
4. **Pins and provenance — superseded 2026-09-03:** do not treat one
   `.dispatcher_pin` per study as authoritative history. Record resolved
   studies, dispatcher, wrapper, and foraging-models SHAs per launch. A default
   dispatcher ref may exist for convenience, but launch records and W&B
   metadata are the source of truth.
5. **CI — resolved:** the existing dispatcher workflow remains with launcher
   tests. The studies repo gets an offline regeneration/provenance workflow and
   its own pinned development dependencies.
6. **Branches — execution-time hard gate:** enumerate open PRs and all remote
   branches, including branches with no PR. No study commit may remain only off
   `main` when extraction begins.
7. **Visibility — execution-time user decision:** match the source repository
   unless the user explicitly chooses otherwise. Public creation is never an
   inferred default.

## Code Ocean: launch surface and archive, not a home (decided 2026-07-16)

Considered and rejected: migrating `studies/` + the Makefile mechanism into
the Code Ocean ecosystem. The current division of labor is correct and the
split does not change it — CO = launch control plane (the capsule's app panel
runs `launch_beaker.py`), Beaker/HPC = compute, W&B = experiment record,
git = code and living documents. Three reasons:

- **Wrong data plane.** Every analysis producer opens `wandb.Api()` and pulls
  by group name; there is no local-data input. CO's core value (immutable
  mounted data assets, lineage) buys nothing when the input is a cloud API —
  a capsule "Reproducible Run" would still depend on mutable external W&B
  state. The provenance actually relied on (`_meta` git SHAs,
  `environment.lock`, W&B group names) already travels with git.
- **Wrong iteration model.** The reports-are-code loop — edit producer,
  `make rN` in seconds against `.gitignore`'d W&B caches, diff the
  regenerated `BEGIN/END` region, review in a PR — is incremental and
  git-native. A capsule has one entrypoint, no incremental targets, no cache
  persistence between runs, and immutable per-run results. Study history is
  full of `fix(study-04): correct overstated claim` commits: reports are
  living documents corrected under review, git's home turf.
- **A fourth home that retires none of the other three,** for a
  2-person-plus-AI team with no external consumers of intermediate state.

**Where CO does earn its place — a hermetic analysis-regeneration capsule**
(analysis only; it never triggers Beaker/HPC — that stays the existing launch
capsule's job). Because studies 03–05 already follow the "freeze the numbers"
convention (committed curated grid CSVs; `make all` runs offline), a capsule
that clones the studies repo at a pinned SHA and runs every study's `make
all` needs no W&B secret, no network, and no data asset. Design, priorities
(CO explicitly last), and the remaining normalization work (study 01,
study 05 r4): [[report-publication-and-reproducibility]]. Out of scope for
this split.

## Committed artifacts policy: keep committing figures & CSVs (decided 2026-07-16)

Measured 2026-07-16: `studies/` tracks 45 PNGs (21 MB) and 46 CSVs (5.2 MB);
largest single files ~1–1.5 MB (study 04 JSONs/CSVs). Decision: **keep
committing them, in the new studies repo.** Rationale:

- Reports embed the figures; without committed PNGs the `r*.md` reports are
  unreadable on GitHub, defeating reports-are-code.
- Regenerated-artifact diffs are a real review signal (history shows them
  catching dropped CSV rows and overstated claims).
- "Regenerable from W&B" decays: if runs are deleted or age out, the
  committed artifacts are the only durable record of a study's answer.
- The dispatcher-weight concern is solved by the split itself — study
  artifacts no longer ride along in the runtime clone that Beaker jobs pull
  at startup.

Discipline in the studies repo: W&B pull caches stay `.gitignore`'d (never
commit them); keep single artifacts under a few MB (downsample figures,
aggregate CSVs — raw pulls belong in the cache layer); revisit git-LFS only
if the repo's pack size approaches a few hundred MB, which at the current
~27 MB of history-blob weight is years away.

## Related

- [[report-publication-and-reproducibility]] — the post-split publication
  layers (freeze-the-numbers completion, CI regen check, docs site, CO
  analysis capsule) that build on this split.
- [[study-organization]] — the intra-study layout that already anticipates
  this split ("one folder per study", "self-contained variants").
- [[posthoc-analysis]] — analysis conventions that already live *inside*
  each study folder and travel with it.
- [[beaker-playbook]] — launcher-side conventions that stay in dispatcher.
- [[AGENTS]] §5 (HPC safety), §6 (Conventional Commits), §9 (PR merge
  policy, no squash), §10 (Beaker), §11 (verify-with-data), §12 (posthoc).
