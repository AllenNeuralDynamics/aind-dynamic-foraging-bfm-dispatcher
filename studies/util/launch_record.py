"""Schema + writer for INTERVENTION launch records (all studies).

WHY THIS EXISTS. A study's *first* launch is well instrumented: the launchers write
`launch_record/beaker_resumable.json` with a fixed schema, and stamp the portable `meta.*` fields into each W&B run config.
wedged tasks, probing a failure, changing priority tier, backfilling a lost metric — has
historically been done with a bespoke script that invented its own JSON shape.

Study 06 is the worked example of the cost. Its five launch records looked like this:

    beaker_resumable.json  (launcher)  {chunk_size, mode, n_chunks, n_tasks, parts}
    beaker_resubmit20.json (by hand)   {bad_node, parts, reason}
    beaker_nanretry1.json  (by hand)   {experiment_id, purpose, tasks}
    beaker_rescue1.json    (by hand)   {bad_node, bad_node_hostname, experiment_id,
                                        progress_discarded, purpose, tasks}
    beaker_tier1.json      (by hand)   {experiment_id, purpose, superseded_experiment,
                                        tasks, thrash_evidence}

No two alike, **none carrying a timestamp**, none using `_meta`. Interventions are exactly
the surprising events provenance exists for, and they were the least well recorded. Prose in
notes.md filled the gap, but prose is not queryable and drifts from reality.

This module fixes the shape, not the prose. `write_intervention()` wraps the existing
`_meta.build_meta()` so every record gets a Seattle-time timestamp and both git SHAs for free.

    from launch_record import write_intervention

    write_intervention(
        path=variant / "launch_record" / "beaker_rescue1.json",
        kind="rescue",
        platform="beaker",
        wandb_group="mult-d-grid@20260718-151409",
        job_refs=[{"type": "beaker_experiment", "id": "01KYXNJYA7KEY47JXWJGEG4M3Y"}],
        study_root=STUDY,
        supersedes=["01KYB43A78GH0X95K1W70EFNSZ"],
        trigger={"symptom": "3 tasks pending 5 days",
                 "cause": "node aidc-h200-prd2 image-pull failure",
                 "evidence": {"node": "01KPVKJYXNWNJCH7ZFK0TBXPW5"}},
        tasks=[{"orig_task": "...-060", "new_task": "rescue1-060",
                "orig_wandb_run": "...", "new_wandb_run": "..."}],
        deviations={"fresh_run_ids": True, "renamed_tasks": True},
        cost={"progress_discarded_steps": {"060": 76120}},
    )

BEAKER *AND* HPC. The two backends are handled by one schema because a study can and does
use both (AGENTS.md section 13: GPU jobs to Beaker, CPU jobs to HPC SLURM, one repo driving
both). They differ only in what identifies a submitted unit, which is why `job_refs` is a
list of typed refs rather than a Beaker-shaped `experiment_ids`:

    beaker  ->  {"type": "beaker_experiment",  "id": "01K..."}
    hpc     ->  {"type": "wandb_sweep",        "id": "abc123"}     # the durable handle
                {"type": "slurm_array_job",    "id": "12345"}      # ages out of sacct
                {"type": "slurm_job",          "id": "12345_7"}

For HPC prefer the **W&B sweep id**: SLURM job ids are recycled and drop out of `sacct`
history, so a sweep id is the only ref that stays resolvable months later. (`launch_hpc.py`
currently writes no launch record at all — an HPC intervention must call this helper by hand
until that launcher is taught to do it.)

Times are Seattle per AGENTS.md section 7.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _meta import build_meta

# Kinds of post-launch intervention. Keep this list short and meaningful: it is what makes
# records queryable ("show me every rescue across all studies"). Add a kind only when an
# existing one genuinely does not describe the action.
KINDS = (
    "resubmit",      # re-run tasks that never produced usable data (bad node, lost image, ...)
    "rescue",        # re-run tasks wedged/stuck, discarding partial progress
    "probe",         # diagnostic re-run to test a hypothesis, not to fill a cell
    "tier-change",   # same work resubmitted at a different priority/preemptible tier
    "backfill",      # recover a metric post-hoc WITHOUT recompute (no new Beaker job)
    "extend",        # continue a finished run to a longer horizon
)

_REQUIRED_TRIGGER_KEYS = ("symptom", "cause")

# Compute backends. "none" is for kinds that create no job at all (backfill), so the field
# stays required and meaningful rather than being left blank.
PLATFORMS = ("beaker", "hpc", "none")

# Ref types per platform. Beaker identifies a submitted unit by experiment; HPC by W&B sweep
# (durable) and SLURM job/array ids (recycled, and they age out of sacct — prefer the sweep).
REF_TYPES = ("beaker_experiment", "wandb_sweep", "slurm_array_job", "slurm_job")


def write_intervention(
    path: Path | str,
    *,
    kind: str,
    platform: str,
    wandb_group: str,
    study_root: Path | str,
    job_refs: list[dict[str, str]] | None = None,
    supersedes: list[str] | None = None,
    trigger: dict[str, Any] | None = None,
    tasks: list[dict[str, Any]] | None = None,
    deviations: dict[str, Any] | None = None,
    cost: dict[str, Any] | None = None,
    produced_by: str = "launch_record.write_intervention",
) -> dict:
    """Write a schema-conformant intervention record and return it.

    Args:
        path: destination JSON, conventionally
            ``studies/<study>/variants/<variant>/launch_record/beaker_<label>.json``.
        kind: one of :data:`KINDS`.
        platform: one of :data:`PLATFORMS` — ``"beaker"``, ``"hpc"``, or ``"none"`` for a
            kind that submits nothing (backfill).
        wandb_group: the group these runs belong to — the join key that lets the
            validator reconcile records against what actually landed in W&B.
        study_root: ``studies/<study>/``, for the wrapper SHA in ``_meta``.
        job_refs: typed handles for the submitted unit(s), e.g.
            ``[{"type": "beaker_experiment", "id": "01K..."}]`` on Beaker or
            ``[{"type": "wandb_sweep", "id": "abc123"}]`` on HPC. Omit for
            ``kind="backfill"``, which by definition creates no job. On HPC prefer the
            W&B sweep id — SLURM ids are recycled and age out of ``sacct``.
        supersedes: experiment ids (or run ids) this action replaces, so a reader can
            follow the chain backwards without reading prose.
        trigger: why this was needed — requires ``symptom`` and ``cause``, plus any
            ``evidence``. The point is to record the *observation*, not just the verdict.
        tasks: per-task mapping, typically
            ``{orig_task, new_task, orig_wandb_run, new_wandb_run}``.
        deviations: deliberate departures from a plain resubmit — e.g.
            ``{"fresh_run_ids": True, "renamed_tasks": True,
               "context_changed": {"priority": "normal", "preemptible": False}}``.
            Every study-06 intervention made such a choice for a real reason; capturing it
            here makes the reason auditable instead of buried in a commit message.
        cost: what the action gave up, e.g. ``{"progress_discarded_steps": {...}}``.

    Raises:
        ValueError: on an unknown ``kind``/``platform``/ref type, a ``trigger`` missing
            required keys, or a job-creating kind with no ``job_refs`` (a silent no-op
            record is worse than no record).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if platform not in PLATFORMS:
        raise ValueError(f"unknown platform {platform!r}; expected one of {PLATFORMS}")

    trigger = dict(trigger or {})
    missing = [k for k in _REQUIRED_TRIGGER_KEYS if not trigger.get(k)]
    if missing:
        raise ValueError(
            f"trigger is missing required key(s) {missing}. An intervention record without "
            "an observed symptom and a cause is not provenance, it is a rumour."
        )

    job_refs = [dict(r) for r in (job_refs or [])]
    for r in job_refs:
        if r.get("type") not in REF_TYPES:
            raise ValueError(f"job_ref {r!r} has unknown type; expected one of {REF_TYPES}")
        if not r.get("id"):
            raise ValueError(f"job_ref {r!r} has no id")

    if kind == "backfill":
        if platform != "none":
            raise ValueError('kind="backfill" submits nothing, so platform must be "none"')
        if job_refs:
            raise ValueError('kind="backfill" submits nothing, so job_refs must be empty')
    else:
        if platform == "none":
            raise ValueError('platform="none" is only valid for kind="backfill"')
        if not job_refs:
            raise ValueError(
                f"kind={kind!r} submits work, so job_refs must be non-empty "
                '(beaker: [{"type":"beaker_experiment","id":...}]; '
                'hpc: [{"type":"wandb_sweep","id":...}])'
            )

    allowed_ref_types = {
        "beaker": {"beaker_experiment"},
        "hpc": {"wandb_sweep", "slurm_array_job", "slurm_job"},
    }
    allowed = allowed_ref_types.get(platform)
    if allowed is not None:
        for r in job_refs:
            if r.get("type") not in allowed:
                raise ValueError(
                    f"job_ref {r!r} not valid for platform {platform!r}; expected type in {sorted(allowed)}"
                )

    record = {
        "_meta": build_meta(produced_by, [wandb_group], study_root=study_root),
        "kind": kind,
        "platform": platform,
        "wandb_group": wandb_group,
        "job_refs": job_refs,
        "supersedes": list(supersedes or []),
        "trigger": trigger,
        "tasks": list(tasks or []),
        "deviations": dict(deviations or {}),
        "cost": dict(cost or {}),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_records(launch_record_dir: Path | str) -> list[dict]:
    """Load every JSON in a ``launch_record/`` dir, newest schema or legacy alike.

    Legacy records (pre-schema, e.g. study 06's hand-written ones) are returned as-is with
    ``_schema: "legacy"`` so the validator can report them without crashing. Being able to
    read the old shape is what lets a study adopt this incrementally.
    """
    out = []
    root = Path(launch_record_dir)
    directories = [root, *sorted(root.parent.glob(root.name + "_*"))]
    for p in sorted(path for directory in directories for path in directory.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            out.append({"_schema": "unreadable", "_path": str(p), "_error": f"{type(e).__name__}: {e}"})
            continue
        if "_meta" in d and "kind" in d:
            schema = "current"
        elif d.get("mode") == "resumable-grid":
            schema = "launcher"
        else:
            schema = "legacy"
        d.setdefault("_schema", schema)
        d["_path"] = str(p)
        out.append(d)
    return out


def refs_in(records: list[dict], ref_type: str) -> set[str]:
    """Every id of ``ref_type`` mentioned by any record, current schema or legacy.

    Legacy shapes are handled explicitly rather than guessed at, so a study can adopt the
    schema without a migration: study 06 used both a flat ``experiment_id`` and a
    ``parts: [{experiment_id: ...}]`` list, and `launch_hpc.py` wrote nothing at all.
    """
    ids: set[str] = set()
    for r in records:
        for ref in r.get("job_refs", []) or []:
            if isinstance(ref, dict) and ref.get("type") == ref_type and ref.get("id"):
                ids.add(str(ref["id"]))
        if ref_type != "beaker_experiment":
            continue
        # legacy Beaker shapes
        for key in ("experiment_ids", "experiment_id"):
            v = r.get(key)
            if isinstance(v, str):
                ids.add(v)
            elif isinstance(v, list):
                ids.update(x for x in v if isinstance(x, str))
        for part in r.get("parts", []) or []:
            if isinstance(part, dict) and isinstance(part.get("experiment_id"), str):
                ids.add(part["experiment_id"])
    return ids


def experiment_ids_in(records: list[dict]) -> set[str]:
    """Beaker experiment ids across current and legacy records."""
    return refs_in(records, "beaker_experiment")
