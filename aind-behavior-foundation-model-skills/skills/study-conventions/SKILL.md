---
name: study-conventions
description: Organize studies, variants, and launch provenance in the studies repo (aind-dynamic-foraging-bfm-studies) — studies/<study>/variants/<variant>/ layout, study naming (NN-{model}-{purpose}), one W&B project per study, group naming <variant>@<launch_id>, the meta.* provenance block, and study wrap-up housekeeping. Use when creating a new study or variant, deciding where a launch's configs live, naming/interpreting W&B groups, or closing out a finished study.
---

# Study & experiment organization

This skill is the source of truth for study organization (the one-line rule lives
in `AGENTS.md` §8).

Studies live in **`aind-dynamic-foraging-bfm-studies`**, a sibling checkout of the
dispatcher (split out at tag `studies-split-v1`, dispatcher #124). Every `studies/...` path
below is relative to that repo's root; the launchers stay in the dispatcher.

**The fields and the layout are defined in the studies repo, not here** — this skill keeps
the workflow and the why. Read them there rather than copying an old study, whose files may
predate a rule:

- [`CONTEXT.md`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/main/CONTEXT.md) — the vocabulary: study, variant, launch, intervention,
  W&B group, report, correction, upstream, producer, freeze.
- [`schemas/`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/main/schemas/README.md) — `layout.yaml` (study root allowlist, required
  files, Makefile targets, one `launch_record/` per variant), study and report frontmatter,
  analysis `_meta` v2, launch-record v2.
- [ADR 0001](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies/blob/main/docs/adr/0001-one-launch-one-wandb-group.md) — one launch, one W&B group.

`make validate` (studies repo root) checks every study against them, prints a fix for each
new finding, and is the required `provenance` CI check. Fix the study; never loosen a schema
or add to `schemas/provenance-baseline.json` to get green.

**Start from the scaffolder, never from a copy of an old study** (studies repo root; a copy
carries whatever rules that study predates). Each writes files that pass `make validate`
from the first commit, plus TODO prompts, and never overwrites anything:

```bash
make new-study   NAME=12-gru-foo QUESTION="Does ...?" PROJECT=gru_foo VARIANT=base MOTIVATED_BY=06/r2 [ISSUE=...]
make new-variant STUDY=12-gru-foo NAME=wider
```

`MOTIVATED_BY` is required: the earlier results this study follows up (`05/r1,03/r2`, a study
`05`, an issue `#23`), or `none` for a root study. It fills `motivated_by` in the frontmatter
(why the study exists; studies ADR 0002), each entry with a note to complete.
`new-study` refuses a used accession number and writes the README frontmatter, CHANGELOG,
Makefile, `.gitignore`, reports INDEX and the first variant's `notes.md`. `new-variant`
writes `notes.md` only; add `sweep.yaml` / `experiment.yaml` next to it before launching.
Reports: `make new-report` (posthoc-reporting skill).

## Core scheme

- A **study answers one scientific question**; its conditions are **variants**, not new
  studies. New top-level `studies/<name>/` only for a genuinely different question
  (different model family, metric, or task).
- **Study folder name = `NN-{model}-{purpose}`**, kebab-case, no dates. `NN` is a
  zero-padded 2-digit **accession number** (stable creation-order id, assigned at
  creation from the next free number — NOT an execution order; never renumber or
  reuse). `{model}` is the model family (`gru`, `disrnn`; use `gru-vs-disrnn` for a
  cross-model comparison study). `{purpose}` is the question
  (`scaling-law`, `scaling-law-ignore`, `beta-scan`). Examples: `01-gru-scaling-law`,
  `02-gru-scaling-law-ignore`, `03-disrnn-beta-scan`. The folder name is independent
  of the W&B **project** name (e.g. folder `02-gru-scaling-law-ignore` ↔ project
  `mice_ignore_scaling`) and of the immutable `meta.study` stamped on already-logged
  runs — when renaming an existing study, note `formerly: <old-name>` in its README so
  the folder ↔ run-stamp mapping stays explicit, and leave historical launch records
  (which record the path used at launch time) unrewritten.
- **One folder per study**: `studies/<study>/`. Its README opens with the study cover
  as YAML frontmatter (question, status, W&B project, verdict) and keeps a **Variants
  index** table (what differs, status, W&B group, Beaker experiment id). Code and data
  live under `analysis/` or a variant, never at the study root (`schemas/layout.yaml`).
- **Variants as self-contained subfolders**: `studies/<study>/variants/<variant>/` with
  its `sweep.yaml`, `experiment.yaml`, `notes.md` (what differs + result + W&B group +
  Beaker exp id), and **one** `launch_record/` directory (subdirectories inside it are
  fine). Name descriptively (`v2-postwarmup`, `hsize-scan`), lowercase, never by date.
- **One W&B project per study, one group per variant** (group set via the sweep's
  `name:`) — keeps variants side-by-side comparable; never a project per variant.

## Provenance (one launch == one "pseudo-sweep")

- **W&B group = `<variant>@<launch_id>`**, launch_id = Seattle timestamp; the prefix is
  exactly the variant directory name (ADR 0001). The launch_id is also folded into run ids
  so repeated launches never collide. A group cannot be renamed once runs exist, so the
  launchers **refuse** a study launch whose group would break this, or that lacks
  `--label` / `--note`, before submitting (dispatcher `provenance.check_launch_identity`).
- **`meta.{study,variant,launch_id,label,note,config_hash}`** — portable across
  CO / Beaker / Allen HPC. `launch_beaker_resumable.py` derives study/variant from the
  `studies/<study>/variants/<variant>/` path and injects `BFM_META_*` env; the
  wrapper's `start_wandb_run` stamps it onto the run.
- **Always pass `--note`** ("why this run exists + what we want to learn") so the
  scientific intent is readable straight from the W&B record.
- Platform-native ids are stamped alongside: `BEAKER_EXPERIMENT_ID`, `BEAKER_JOB_ID`,
  `CO_COMPUTATION_ID`, plus `wrapper_commit`, `dispatcher_commit`,
  `foraging_models_commit` and `meta.studies_commit` (the commit of the checkout that
  held the sweep; equal to the dispatcher commit until the studies split, #124). Both
  Beaker launchers (`launch_beaker_resumable.py` and `launch_beaker.py`) implement the
  portable launch metadata identically; the wrapper records the resolved source commits.
- **The launch record is launch-record v2** (studies `schemas/launch-record.v2.schema.json`):
  the launchers fill `kind: initial`, `platform`, `label`, `note`, `job_refs`, `revisions`
  (studies / dispatcher / wrapper / foraging-models full SHAs; an unresolved one is left
  out so the checker reports it), `manifests`, `n_tasks`, `data` (the data config, its
  snapshot, and the sha256 of the snapshot's manifest) and `image` (the Beaker image name
  and the immutable id it resolved to, or the HPC conda env and a digest of its packages),
  and keep the legacy launcher keys
  (`studies_git_commit`, `dispatcher_commit`, `studies_dirty_override`,
  `provenance_warnings`, ...) for older readers. A launch whose
  `studies/<study>/` directory has uncommitted changes is **refused**; commit first, or
  pass `--diagnostic-dirty-ok "<reason>"` for a diagnostic run (the reason is recorded).
  A launch whose data config reads the live database (no `snapshot` pinned) is also
  **refused**: its results could never be regenerated. Pin `data.snapshot`, or pass
  `--allow-live-data` for a smoke test.
  `launch_hpc.py` writes `hpc_sweep.json` to `--output-dir` the same way the Beaker
  launchers write `beaker_resumable.json` / `beaker_sweep.json`.

## Interventions after the first launch (resubmit / rescue / probe / tier change / backfill)

**Rule (ADR 0001):** an intervention joins the launch's W&B group and reuses its manifests,
differing only in what it lists under `deviations`; only `extend` may change `n_steps`. Any
other change to scientific parameters is a new launch, usually of a new variant.

The launchers record the **first** launch. Everything afterwards used to be hand-rolled, and
that is precisely where provenance was lost: study 06 accumulated five follow-up actions with
five different ad-hoc JSON shapes, **none carrying a timestamp**, plus 6 post-hoc-recovered
metric values documented only in prose. Interventions are the surprising events — the ones
provenance exists for — so they get a fixed schema:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "util"))
from launch_record import write_intervention
write_intervention(
    path=variant / "launch_record" / "<backend>_<label>.json",
    kind="rescue",                    # resubmit|rescue|probe|tier-change|backfill|extend
    platform="beaker",                # beaker | hpc | none (none == submits nothing)
    wandb_group="<variant>@<launch_id>",   # the join key the validator reconciles on
    study_root=STUDY,
    job_refs=[{"type": "beaker_experiment", "id": exp.id}],   # see backend table below
    supersedes=["<older experiment/sweep id>"],
    trigger={"symptom": ..., "cause": ..., "evidence": {...}},   # symptom+cause REQUIRED
    tasks=[{"orig_task": ..., "new_task": ..., "orig_wandb_run": ..., "new_wandb_run": ...}],
    deviations={"fresh_run_ids": True, "renamed_tasks": True},
    cost={"progress_discarded_steps": {...}},
)
```

**Works for both backends** — a study can use both (AGENTS.md §13: GPU → Beaker, CPU → HPC
SLURM). They differ only in what identifies a submitted unit:

| platform | `job_refs` entry | note |
|---|---|---|
| `beaker` | `{"type": "beaker_experiment", "id": "01K…"}` | |
| `hpc` | `{"type": "wandb_sweep", "id": "…"}` | **prefer this** — durable |
| `hpc` | `{"type": "slurm_array_job" \| "slurm_job", "id": "…"}` | recycled, ages out of `sacct` |
| `none` | *(omit)* | only for `kind="backfill"` |

`launch_hpc.py` writes the *first* launch's record (`hpc_sweep.json`, dispatcher #176), like the
Beaker launchers; anything after that, and any hand-run `sbatch` array, is recorded with this
helper. A hand-run array also needs a record naming its runtime SHAs (`studies_git_commit`,
`dispatcher_git_commit`, `wrapper_git_commit`, `foraging_models_git_commit` plus `wandb_group`),
since an intervention record's `_meta` describes the writer, not the job's checkouts.

It wraps `_meta.build_meta()`, so timestamp + studies/wrapper SHAs come for free, and it
**refuses** a record with no `trigger.cause`, an unknown platform/ref type, or a work-submitting
kind with no `job_refs`.

Record the **`deviations`** honestly — every study-06 intervention made a deliberate choice
(fresh vs reused W&B run ids, renamed tasks, changed priority tier) for a real reason. Reused
run ids are only safe when the old run logged **nothing**; a from-scratch rerun replays step 0
while W&B's counter does not rewind.

## Checklist for a new variant launch

1. `make new-variant STUDY=<study> NAME=<variant>`, then add `sweep.yaml` +
   `experiment.yaml` next to the generated `notes.md`.
2. Fill in `notes.md`: what differs from the sibling variants and what you expect.
3. Include `WRAPPER_REF`, `DISPATCHER_REF`, and `FORAGING_MODELS_REF` in the Beaker
   YAML. Branch names may remain for readability; the launcher resolves all three
   to full SHAs in the submitted launch record.
4. Launch via the beaker-launch or hpc-launch skill with `--label` and `--note`.
5. Add a row to the study README's Variants index.
6. After the group settles, write `launch_record/results.md` (one launch_record/ per
   variant; put a later launch's in a subdirectory, `launch_record/<label>/results.md`).
   Contract in the posthoc-reporting skill.
7. Run `make validate` and fix every new finding.
8. **Reconcile provenance before calling the variant done:**

   ```bash
   python studies/util/validate_provenance.py studies/<study> --variant <variant> \
          --beaker --wandb --strict
   ```

   Cross-checks three sources that otherwise drift silently: the launch records, the live
   backends (Beaker experiments / W&B sweeps) targeting the group, and any post-hoc-recovered
   values flagged in `grid.csv`. Advisory by default; `--strict` exits 1 on findings — use
   that at wrap-up. Checks 1–2 are offline (no credentials needed); `--beaker` / `--wandb`
   add the live reconciliation.

## References (read on demand)

- `references/study-wrapup.md` — the verified end-to-end procedure for closing out
  a finished study: normalize the folder, clean launch records, clean stale W&B
  runs (irreversible — confirm first), the two-repo PR flow to
  `main` (wrapper first, never squash), branch retirement, and
  the `git mv` re-staging trap.
