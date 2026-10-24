"""Beaker launcher for RESUMABLE grid runs (one autoResume task per grid point).

This is the Option-1 alternative to ``launch_beaker.py``. Instead of creating a
W&B *sweep* drained by packed ``wandb agent`` replicas (where a preempted trial
is abandoned and re-run from scratch), it expands a grid sweep YAML into one
self-contained Beaker task per grid point, each running ``run_hpc`` with the
grid point's overrides baked in. Combined with:

  * Beaker ``preemptible: true`` (the server applies ``autoResume: true`` for
    preemptible/low-priority tasks — verified via ``beaker experiment spec``), so
    a preempted task is restarted as the *same* task with the *same* /results,
  * ``BFM_RESUMABLE_OUTPUT_DIR`` (run_hpc anchors outputs at a stable path
    instead of the per-run W&B dir), and
  * the trainer's full-state checkpoint/resume,

a preempted run continues from its last checkpoint instead of restarting. W&B
dashboard continuity across the restart is preserved by pinning a deterministic
``WANDB_RUN_ID`` per grid point with ``WANDB_RESUME=allow``.

Only ``method: grid`` sweeps are supported (the whole point of Option 1 — the
trial set must be enumerable up front; there is no sweep controller to ask).

Beaker submission goes through ``code/beaker_client.py`` (beaker-py) rather than the
``beaker`` CLI, so this script runs unmodified whether invoked on Allen HPC / Code
Ocean or from the Claude Science sandbox (Mac-based orchestration; see
``docs/claude-science-workflow.md``) — see ``beaker_client.py``'s docstring for why
the CLI-based version couldn't do that.

Usage:
  python code/launch_beaker_resumable.py \
    --sweep code/beaker/sweep_gru_scaling.yaml \
    --experiment code/beaker/experiment_scaling.yaml \
    --workspace ai1/aind-dynamic-foraging-foundation-model
  python code/launch_beaker_resumable.py --sweep ... --experiment ... --no-submit
"""

import argparse
import copy
import hashlib
import itertools
import os
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

# Library-only Beaker client (beaker-py) -- no `beaker` CLI dependency; see
# code/beaker_client.py docstring for why.
from beaker_client import pin_runtime_refs, submit_beaker_experiment
from provenance import (
    DirtyStudiesError, check_studies_clean, launch_provenance, print_warnings,
    record_fields, runtime_shas_from_pins, with_runtime_shas,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Where the launch record is written. /results is the Code Ocean capsule path; it does not
# exist (and is not writable) on the HPC login node or the Claude Science Mac sandbox, where
# the launcher also runs, so fall back to a repo-local dir there instead of dying with
# PermissionError: [Errno 13] '/results'. Override with --output-dir.
RESULTS = Path("/results") if os.access("/results", os.W_OK) else REPO_ROOT / "results"

# Stable path (per task's own /results dataset) where run_hpc anchors outputs so
# autoResume restarts re-find their checkpoints. See run_hpc.py.
RESUMABLE_OUTPUT_DIR = "/results/run"
SWEEP_ARGS_PLACEHOLDER = "${args_no_hyphens}"


def _git_sha(repo_dir: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _grid_points(parameters: dict) -> list[dict[str, object]]:
    """Cartesian product of a W&B-style grid ``parameters`` block.

    Each parameter is ``{values: [...]}`` (grid axis) or ``{value: x}`` (fixed).
    """
    axes: list[tuple[str, list]] = []
    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            sys.exit(f"[resumable] parameter {key!r} must be a mapping, got {spec!r}")
        if "values" in spec:
            axes.append((key, list(spec["values"])))
        elif "value" in spec:
            axes.append((key, [spec["value"]]))
        else:
            sys.exit(
                f"[resumable] parameter {key!r} needs 'values' or 'value' "
                f"(distribution-based search is not supported in grid mode)"
            )
    keys = [k for k, _ in axes]
    return [dict(zip(keys, combo)) for combo in itertools.product(*(v for _, v in axes))]


def _override_str(key: str, value: object) -> str:
    """Render a Hydra override ``key=value``, matching W&B's grid formatting."""
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    return f"{key}={value}"


def _run_id(group: str, overrides: list[str]) -> str:
    """Deterministic W&B run id for a grid point (stable across resume)."""
    digest = hashlib.sha1((group + "|" + "|".join(sorted(overrides))).encode()).hexdigest()
    return f"{group}-{digest[:8]}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _seattle_launch_id() -> str:
    """Readable, unique-per-launch id = Seattle local timestamp (see AGENTS §7/§8)."""
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")


def _study_variant(sweep_path: Path) -> tuple[str, str]:
    """Derive (study, variant) from a studies/<study>/variants/<variant>/sweep.yaml path.

    Falls back to the file stem when the path isn't under that layout, so the launcher
    still works for ad-hoc sweeps.
    """
    parts = sweep_path.resolve().parts
    study = variant = None
    if "studies" in parts and parts.index("studies") + 1 < len(parts):
        study = parts[parts.index("studies") + 1]
    if "variants" in parts and parts.index("variants") + 1 < len(parts):
        variant = parts[parts.index("variants") + 1]
    return study or "adhoc", variant or _slug(sweep_path.stem)


def build_spec(sweep_file: str, experiment_file: str, label: str | None = None,
               note: str | None = None, provenance: dict | None = None) -> dict:
    """Expand a grid sweep + experiment template into a multi-task Beaker spec.

    ``provenance`` is the ``launch_provenance`` block for this sweep; computed here when not
    given, so the studies commit reaches the wrapper (``BFM_META_STUDIES_COMMIT``) on every
    route through this function.
    """
    sweep = yaml.safe_load(Path(sweep_file).read_text())
    if sweep.get("method") != "grid":
        sys.exit(f"[resumable] only method: grid is supported, got {sweep.get('method')!r}")
    base_command = sweep.get("command")
    if not base_command or SWEEP_ARGS_PLACEHOLDER not in base_command:
        sys.exit(
            f"[resumable] sweep 'command' must include {SWEEP_ARGS_PLACEHOLDER} "
            f"(the swept overrides); got {base_command!r}"
        )
    grid = _grid_points(sweep.get("parameters", {}))
    if not grid:
        sys.exit("[resumable] sweep has no grid points")

    spec = yaml.safe_load(Path(experiment_file).read_text())
    template_task = spec["tasks"][0]
    # The bash + entrypoint.sh prefix from the template (drop its wandb-agent tail).
    entry_prefix = list(template_task["command"][:2])

    # Provenance / tracking (see AGENTS §8): one launch == one pseudo-sweep.
    #  - W&B group = "<variant>@<launch_id>" uniquely names THIS launch (distinguishes
    #    repeats; readable: variant -> study folder, launch_id -> Seattle time).
    #  - meta.* (study/variant/launch_id/label/note/config_hash) injected via BFM_META_*
    #    env; the wrapper stamps them into the run config alongside the platform-native
    #    BEAKER_EXPERIMENT_ID / BEAKER_JOB_ID / CO_COMPUTATION_ID. `note` = free-text
    #    "why this run + what we want to learn", readable straight from the run record.
    #  - launch_id is also folded into the run id, so every launch gets unique ids
    #    (distinguishes repeats AND avoids the deleted-id resync problem).
    sweep_path = Path(sweep_file)
    study, variant = _study_variant(sweep_path)
    launch_id = _seattle_launch_id()
    config_hash = hashlib.sha1(sweep_path.read_bytes()).hexdigest()[:8]
    group = f"{variant}@{launch_id}"            # W&B group = this launch (pseudo-sweep)
    id_base = _slug(f"{variant}-{launch_id}")    # slug-safe base for task names + run ids
    provenance = provenance or launch_provenance(sweep_path, REPO_ROOT)
    meta_env = [
        {"name": "BFM_META_STUDY", "value": study},
        {"name": "BFM_META_VARIANT", "value": variant},
        {"name": "BFM_META_LAUNCH_ID", "value": launch_id},
        {"name": "BFM_META_CONFIG_HASH", "value": config_hash},
    ]
    if provenance.get("studies_git_commit"):
        # Which studies commit held this sweep -- distinct from DISPATCHER_REF (the
        # launcher) once studies/ lives in its own repository (#124).
        meta_env.append({"name": "BFM_META_STUDIES_COMMIT", "value": provenance["studies_git_commit"]})
    if label:
        meta_env.append({"name": "BFM_META_LABEL", "value": label})
    if note:
        meta_env.append({"name": "BFM_META_NOTE", "value": note})
    print(f"[resumable] study={study} variant={variant} launch_id={launch_id} "
          f"group={group} config_hash={config_hash}")

    tasks = []
    for index, point in enumerate(grid):
        overrides = [_override_str(k, v) for k, v in point.items()]
        # run_hpc command = sweep base command with the placeholder expanded.
        run_cmd: list = []
        for token in base_command:
            if token == SWEEP_ARGS_PLACEHOLDER:
                run_cmd.extend(overrides)
            else:
                run_cmd.append(token)

        task = copy.deepcopy(template_task)
        task.pop("replicas", None)
        task["name"] = f"{id_base}-{index:03d}"
        task["command"] = entry_prefix + run_cmd
        # Preemptible => Beaker auto-applies autoResume (verified via `beaker experiment
        # spec`; setting autoResume explicitly alongside preemptible is rejected), and the
        # restart re-uses this task's /results. Priority `low`: low-priority preemptible
        # jobs burst onto spare idle GPUs BEYOND the workspace's unallocated-slot budget
        # (measured: ~14 concurrent on H200 vs a normal-priority cap of 8), at the cost of
        # being evicted first — which autoResume recovers. So `low` maximizes fan-out
        # throughput. Use `normal`/`high` for jobs you need to resist eviction.
        #
        # This is the DEFAULT ONLY. It used to be an unconditional assignment, which
        # silently discarded any `context:` the experiment template set — so a template
        # asking for the protected/allocated tier ({priority: high, preemptible: false})
        # was rewritten to low/preemptible with no warning, and the launch looked
        # successful. Honour an explicit template context instead; fall back to the
        # burst default when the template says nothing.
        template_ctx = template_task.get("context")
        if template_ctx:
            task["context"] = copy.deepcopy(template_ctx)
        else:
            task["context"] = {"priority": "low", "preemptible": True}

        managed = {
            "BFM_RESUMABLE_OUTPUT_DIR", "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP",
            "BFM_META_STUDY", "BFM_META_VARIANT", "BFM_META_LAUNCH_ID",
            "BFM_META_CONFIG_HASH", "BFM_META_LABEL", "BFM_META_NOTE",
            "BFM_META_STUDIES_COMMIT",
        }
        env = [e for e in task.get("envVars", []) if e.get("name") not in managed]
        env.extend([
            {"name": "BFM_RESUMABLE_OUTPUT_DIR", "value": RESUMABLE_OUTPUT_DIR},
            {"name": "WANDB_RUN_ID", "value": _run_id(id_base, overrides)},
            {"name": "WANDB_RESUME", "value": "allow"},
            {"name": "WANDB_RUN_GROUP", "value": group},
            *meta_env,
        ])
        task["envVars"] = env
        tasks.append(task)

    spec["tasks"] = tasks
    spec["description"] = f"{group} — {len(tasks)} resumable grid tasks (autoResume)"
    return spec


def submit(spec: dict, workspace: str, output_dir: Path) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = output_dir / "experiment_resumable_submitted.yaml"
    rendered.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"[resumable] submitting {rendered.name} ({len(spec['tasks'])} tasks) to {workspace}")
    try:
        experiment_id = submit_beaker_experiment(str(rendered), workspace)
    except Exception as exc:
        sys.exit(f"[resumable] Beaker experiment submission failed: {exc}")
    print(f"[resumable] Experiment {experiment_id} submitted")
    return experiment_id


def spec_group(spec: dict) -> str | None:
    """The W&B group the rendered spec's tasks carry (``WANDB_RUN_GROUP``)."""
    for task in spec.get("tasks", []):
        for e in task.get("envVars", []):
            if e.get("name") == "WANDB_RUN_GROUP" and e.get("value"):
                return e["value"]
    return None


def save_record(sweep_file: str, experiment_id: str | None, n_tasks: int, output_dir: Path,
                provenance: dict | None = None, runtime_shas: dict | None = None,
                dirty_override: dict | None = None, wandb_group: str | None = None) -> None:
    """Write ``beaker_resumable.json`` (+ a copy of the sweep) to ``output_dir``.

    Carries the two-root provenance block (studies / dispatcher / wrapper / foraging-models
    SHAs, see ``provenance.py``). ``dispatcher_commit`` is kept alongside
    ``dispatcher_git_commit`` because the provenance validator and older readers key on it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_file, output_dir / Path(sweep_file).name)
    if provenance is None:
        provenance = launch_provenance(Path(sweep_file), REPO_ROOT)
    provenance = with_runtime_shas(provenance, runtime_shas)
    record = {
        "mode": "resumable-grid",
        "wandb_group": wandb_group,
        "n_tasks": n_tasks,
        "experiment_id": experiment_id,
        "experiment_url": f"https://beaker.org/ex/{experiment_id}" if experiment_id else None,
        "dispatcher_commit": provenance.get("dispatcher_git_commit") or _git_sha(REPO_ROOT),
        **record_fields(provenance, dirty_override),
        "sweep_file": str(sweep_file),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "beaker_resumable.json").write_text(json.dumps(record, indent=2))
    print(f"[resumable] saved record:\n{json.dumps(record, indent=2)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Launch a grid as one autoResume Beaker task per grid point."
    )
    p.add_argument("--sweep", default=str(REPO_ROOT / "code/beaker/sweep_gru_scaling.yaml"))
    p.add_argument("--experiment", default=str(REPO_ROOT / "code/beaker/experiment_scaling.yaml"),
                   help="experiment template providing image/cluster/resources/refs/secret")
    p.add_argument("--workspace", default="ai1/aind-dynamic-foraging-foundation-model")
    p.add_argument("--no-submit", action="store_true",
                   help="render the spec + record only; don't submit to Beaker")
    p.add_argument("--output-dir", default=str(RESULTS))
    p.add_argument("--label", default=None,
                   help="optional human label for this launch (stamped to W&B meta.label)")
    p.add_argument("--note", default=None,
                   help="free-text background + what we want to learn (stamped to W&B meta.note)")
    p.add_argument("--diagnostic-dirty-ok", default=None, metavar="REASON",
                   help="launch even though studies/<study>/ has uncommitted changes; REASON is "
                        "written into the launch record (diagnostic runs only, never a result)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    provenance = launch_provenance(Path(args.sweep), REPO_ROOT)
    print_warnings(provenance, "[resumable] ")
    try:
        dirty_override = check_studies_clean(provenance, args.diagnostic_dirty_ok)
    except DirtyStudiesError as exc:
        sys.exit(f"[resumable] refusing to launch: {exc}")
    spec = build_spec(args.sweep, args.experiment, label=args.label, note=args.note,
                      provenance=provenance)
    try:
        pins = pin_runtime_refs(spec)
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"[resumable] runtime ref resolution failed: {exc}")
    provenance = with_runtime_shas(provenance, runtime_shas_from_pins(pins))
    n_tasks = len(spec["tasks"])
    print(f"[resumable] expanded grid into {n_tasks} tasks")

    experiment_id = None
    if args.no_submit:
        rendered = output_dir / "experiment_resumable_submitted.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        rendered.write_text(yaml.safe_dump(spec, sort_keys=False))
        print(f"[resumable] --no-submit set; wrote {rendered}")
    else:
        experiment_id = submit(spec, args.workspace, output_dir)

    save_record(args.sweep, experiment_id, n_tasks, output_dir,
                provenance=provenance, dirty_override=dirty_override,
                wandb_group=spec_group(spec))
    print("[resumable] done")


if __name__ == "__main__":
    main()
