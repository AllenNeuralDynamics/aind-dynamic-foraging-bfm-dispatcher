#!/usr/bin/env python3
"""Reconcile a variant's launch records against what actually happened.

WHY. Three sources drift independently and nothing checks them against each other:

  1. `launch_record/*.json`  — what we say we launched
  2. Beaker                  — what experiments actually exist for this W&B group
  3. `analysis/grid.csv`     — which values are real vs recovered post-hoc

Study 06 drifted on all three at once and none of it was noticed until a manual audit:
five interventions were launched, 13 Beaker experiments existed where the docs said 8, and
6 held-out values had been reconstructed post-hoc with that fact recorded only in a skill
reference and a script docstring — not where a reader of `grid.csv` would ever look.

Every check here exists because that specific drift happened.

    python studies/util/validate_provenance.py studies/06-... --variant mult-d-grid
    python studies/util/validate_provenance.py studies/06-... --variant mult-d-grid --beaker

ADVISORY BY DEFAULT: exit 0 with findings printed, so it can be wired into a wrap-up
checklist without blocking anyone mid-flight. `--strict` exits 1 on any finding, for CI or
for the final study wrap-up where drift should genuinely stop the show. A validator that
blocks too early just gets bypassed, and a bypassed validator checks nothing.

BEAKER/W&B ARE OPTIONAL: checks 1-2 are fully offline (they read committed files), so the
common case needs no credentials. `--beaker` adds the live reconciliation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_record import experiment_ids_in, load_records, refs_in  # noqa: E402


def _finding(findings: list, level: str, check: str, msg: str) -> None:
    findings.append({"level": level, "check": check, "message": msg})


def check_records_parse(records: list[dict], findings: list) -> None:
    """Check 1 — records exist and report which still use the pre-schema shape."""
    if not records:
        _finding(findings, "ERROR", "records", "no launch_record/*.json found at all")
        return
    unreadable = [r for r in records if r.get("_schema") == "unreadable"]
    for r in unreadable:
        _finding(findings, "ERROR", "records",
                 f"{Path(r['_path']).name}: unreadable JSON ({r.get('_error')})")

    legacy = [Path(r["_path"]).name for r in records if r.get("_schema") == "legacy"]
    if legacy:
        _finding(findings, "WARN", "records",
                 f"{len(legacy)}/{len(records)} record(s) predate the schema (no _meta/kind): "
                 f"{', '.join(legacy)}")

    # Current-schema records must carry a cause; a record that says what was launched but not
    # why is not provenance. Kept as its own loop -- it applies to every current record, not
    # only to the ones reached while reporting legacy files.
    for r in records:
        if r.get("_schema") == "launcher":
            required = ("experiment_id", "created_utc", "sweep_file", "n_tasks")
            missing = [key for key in required if not r.get(key)]
            if missing or not re.fullmatch(r"[0-9a-f]{40}", r.get("dispatcher_commit") or ""):
                _finding(findings, "ERROR", "records",
                         f"{Path(r['_path']).name}: incomplete launcher record "
                         f"(missing={missing}, dispatcher_commit={r.get('dispatcher_commit')!r})")
            continue
        if r.get("_schema") != "current":
            continue
        if not r.get("trigger", {}).get("cause"):
            _finding(findings, "ERROR", "records",
                     f"{Path(r['_path']).name}: kind={r.get('kind')} has no trigger.cause")


def check_backfill_documented(analysis_dir: Path, records: list[dict], findings: list) -> None:
    """Check 2 — any post-hoc recovered value in grid.csv needs a backfill record.

    This is the check that matters most for anyone *using* the data: a reader must be able
    to tell natively-logged values from reconstructed ones, and find out how they were
    reconstructed. In study 06 that link existed only in prose.
    """
    grid = analysis_dir / "grid.csv"
    if not grid.exists():
        return
    with grid.open() as f:
        rows = list(csv.DictReader(f))
    flag_cols = [c for c in (rows[0].keys() if rows else []) if "backfill" in c.lower()]
    if not flag_cols:
        return
    n_backfilled = sum(1 for r in rows for c in flag_cols if str(r.get(c)).lower() == "true")
    has_backfill_record = any(r.get("kind") == "backfill" for r in records)
    if n_backfilled and not has_backfill_record:
        _finding(findings, "ERROR", "backfill",
                 f"grid.csv has {n_backfilled} post-hoc recovered value(s) "
                 f"(column(s): {', '.join(flag_cols)}) but no launch record with "
                 'kind="backfill" explains how they were produced')
    elif n_backfilled:
        _finding(findings, "INFO", "backfill",
                 f"{n_backfilled} recovered value(s), documented by a backfill record")


def check_beaker(wandb_group: str, records: list[dict], workspace: str,
                 scan_limit: int, findings: list) -> None:
    """Check 3 — every Beaker experiment for this group appears in some record.

    Catches the failure mode directly: someone launched something and never wrote it down.
    Scans recent workspace experiments and matches on each task's WANDB_RUN_GROUP env, which
    is the only reliable link from a Beaker experiment back to a study's W&B group.
    """
    try:
        from beaker import Beaker, Config
    except ImportError:
        _finding(findings, "WARN", "beaker", "beaker-py not importable; skipped")
        return
    token = os.environ.get("BEAKER_TOKEN")
    if not token:
        try:
            import yaml
            token = yaml.safe_load(Path.home().joinpath(".beaker/config.yml").read_text())["user_token"]
        except Exception:
            _finding(findings, "WARN", "beaker", "no BEAKER_TOKEN and ~/.beaker/config.yml unreadable; skipped")
            return

    b = Beaker(Config(user_token=token, default_org="ai1", default_workspace=workspace))
    b._timeout = 60
    recorded = experiment_ids_in(records)

    found: set[str] = set()
    for exp in list(b.workspace.experiments(limit=scan_limit)):
        try:
            spec = b.experiment.spec(exp.id)
        except Exception:
            continue
        for t in spec.tasks:
            grp = next((e.value for e in (t.env_vars or []) if e.name == "WANDB_RUN_GROUP"), None)
            if grp == wandb_group:
                found.add(exp.id)
                break

    unrecorded = found - recorded
    missing = recorded - found
    if unrecorded:
        _finding(findings, "ERROR", "beaker",
                 f"{len(unrecorded)} Beaker experiment(s) target group {wandb_group} but appear "
                 f"in NO launch record: {', '.join(sorted(unrecorded))}")
    if missing:
        _finding(findings, "WARN", "beaker",
                 f"{len(missing)} recorded experiment(s) not seen in the last {scan_limit} "
                 f"workspace experiments (may simply be older than the scan window): "
                 f"{', '.join(sorted(missing))}")
    if not unrecorded and not missing:
        _finding(findings, "INFO", "beaker",
                 f"all {len(found)} Beaker experiment(s) for this group are recorded")


def check_wandb_sweeps(wandb_group: str, records: list[dict], project: str, findings: list) -> None:
    """Check 4 (HPC side) — every W&B sweep feeding this group appears in some record.

    This is the HPC analogue of the Beaker experiment check. On SLURM the durable handle is
    the **W&B sweep id**: SLURM job ids are recycled and age out of `sacct`, so a sweep id is
    the only ref that still resolves months later. Harmless for pure-Beaker studies, whose
    runs simply have no sweep attached.
    """
    try:
        import wandb
    except ImportError:
        _finding(findings, "WARN", "wandb", "wandb not importable; skipped")
        return
    if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        _finding(findings, "WARN", "wandb", "no WANDB_API_KEY and no ~/.netrc; skipped")
        return

    os.environ.setdefault("WANDB_SILENT", "true")
    try:
        api = wandb.Api(timeout=30)
        runs = list(api.runs(project, filters={"group": wandb_group}, per_page=200))
    except Exception as e:  # noqa: BLE001
        _finding(findings, "WARN", "wandb", f"could not query W&B ({type(e).__name__}); skipped")
        return

    seen: set[str] = set()
    for r in runs:
        sweep = getattr(r, "sweep", None)
        if sweep is not None and getattr(sweep, "id", None):
            seen.add(str(sweep.id))
    if not seen:
        _finding(findings, "INFO", "wandb",
                 f"{len(runs)} run(s) in group, none attached to a sweep "
                 "(expected for Beaker-launched studies)")
        return

    recorded = refs_in(records, "wandb_sweep")
    unrecorded = seen - recorded
    if unrecorded:
        _finding(findings, "ERROR", "wandb",
                 f"{len(unrecorded)} W&B sweep(s) feed group {wandb_group} but appear in NO "
                 f"launch record: {', '.join(sorted(unrecorded))}")
    else:
        _finding(findings, "INFO", "wandb",
                 f"all {len(seen)} W&B sweep(s) for this group are recorded")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", type=Path, help="path to studies/<study>/")
    p.add_argument("--variant", required=True)
    p.add_argument("--wandb-group", default=None,
                   help="defaults to the group named by the variant's launch records")
    p.add_argument("--beaker", action="store_true",
                   help="reconcile Beaker experiments against the records")
    p.add_argument("--wandb", action="store_true",
                   help="reconcile W&B sweeps against the records (the HPC/SLURM side)")
    p.add_argument("--project", default="AIND-disRNN/disrnn_data_scaling",
                   help="W&B entity/project for --wandb")
    p.add_argument("--workspace", default="ai1/aind-dynamic-foraging-foundation-model")
    p.add_argument("--scan-limit", type=int, default=60,
                   help="how many recent workspace experiments to scan (default 60)")
    p.add_argument("--strict", action="store_true", help="exit 1 if any ERROR/WARN was found")
    args = p.parse_args()

    variant_dir = args.study / "variants" / args.variant
    records = load_records(variant_dir / "launch_record")
    findings: list = []

    check_records_parse(records, findings)
    check_backfill_documented(args.study / "analysis", records, findings)

    group = args.wandb_group or next(
        (
            (r.get("wandb_group") or r.get("group"))
            for r in records
            if r.get("wandb_group") or r.get("group")
        ),
        None,
    )
    if args.beaker or args.wandb:
        if not group:
            _finding(findings, "WARN", "reconcile",
                     "no W&B group known (legacy records omit it); pass --wandb-group")
        else:
            if args.beaker:
                check_beaker(group, records, args.workspace, args.scan_limit, findings)
            if args.wandb:
                check_wandb_sweeps(group, records, args.project, findings)

    print(f"provenance check: {args.study.name} / {args.variant}"
          + (f"  (group {group})" if group else ""))
    print(f"  {len(records)} launch record(s)")
    for f in findings:
        print(f"  [{f['level']:5s}] {f['check']}: {f['message']}")
    if not findings:
        print("  no findings")

    bad = [f for f in findings if f["level"] in ("ERROR", "WARN")]
    if bad and args.strict:
        print(f"\nFAIL (strict): {len(bad)} finding(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
