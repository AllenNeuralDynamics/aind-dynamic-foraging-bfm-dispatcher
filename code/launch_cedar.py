"""Cedar launcher: one Cedar session per grid point (dispatcher #219).

Cedar (the Allen compute environment) runs interactive sessions, not batch jobs, so this
launcher follows ``launch_beaker_resumable.py`` with a session in place of a Beaker task. It
expands a ``method: grid`` W&B sweep into its grid points and, for each one:

  1. ``cedar session create`` from a ``session_cedar.yaml`` (workspace, cluster, image,
     resources, lifetime, secrets, env), authorizing every SSH key saved in Cedar;
  2. one ``ssh`` that starts ``beaker/entrypoint.sh python -m run_hpc <command + overrides>``
     under ``nohup setsid`` and returns at once. The entrypoint fetches the wrapper,
     dispatcher and foraging-models at the pinned full SHAs; outputs go to
     ``$CEDAR_OUTPUTS_DIR/run``, which outlives the session.

Each run gets the same deterministic ``WANDB_RUN_ID``, ``WANDB_RESUME=allow``, W&B group
``<variant>@<launch_id>`` and ``BFM_META_*`` as on Beaker. No W&B sweep controller is used, so
the sweep file is platform-neutral. The launch record (``cedar_launch.json``) is a
launch-record v2 with ``platform: cedar`` (studies #153) and one ``cedar_session`` job ref
per session.

The launcher is fire-and-forget: a CLI sign-in on Allen HPC lasts about 78 minutes
(AllenOCTO/cedar#544), so it starts the runs and exits rather than watching them. Cedar does
not restart a run after a Pod restart; resuming is a later step (dispatcher #219).

Cedar limits a session's start to 1 minute on fixed-size clusters, which a first pull of a
multi-GB image does not meet (AllenOCTO/cedar#543). Use an image tag already on the node.

Session names are permanent in a workspace, so every launch makes new ones
(``<variant>-<launch_id>-<k>``) and never retries a name. If a session fails to start, the
launcher stops and records the sessions it already created.

Usage (from the studies root, signed in to Cedar):
  python ../aind-dynamic-foraging-bfm-dispatcher/code/launch_cedar.py \\
    --sweep studies/<study>/variants/<variant>/sweep.yaml \\
    --session studies/<study>/variants/<variant>/session_cedar.yaml \\
    --output-dir studies/<study>/variants/<variant>/launch_record/<label> \\
    --label <label> --note "..."
A ``--dry-run`` creates nothing; give it an ``--output-dir`` outside the variant (it refuses one
inside, since uncommitted files there would make the real launch refuse).
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from beaker_client import pin_runtime_refs
from launch_beaker_resumable import (SWEEP_ARGS_PLACEHOLDER, _grid_points, _override_str, _run_id,
                                     _seattle_launch_id, _slug, _study_variant)
from launch_inputs import DataConfigError, data_record, live_data_refusal, runtime_config_reader, runtime_dispatcher_sha
from provenance import (DirtyStudiesError, LaunchIdentityError, check_launch_identity, check_studies_clean,
                        launch_provenance, print_warnings, record_fields, runtime_shas_from_pins,
                        v2_record_fields, with_runtime_shas)

REPO_ROOT = Path(__file__).resolve().parent.parent
CEDAR = shutil.which("cedar") or str(Path.home() / ".cedar" / "cedar")
SESSION_KEYS = {"workspace", "cluster", "image", "gpu", "cpu", "ram", "max_live_time"}
# Set per session by the launcher; a value for one of these in session_cedar.yaml is dropped.
MANAGED_ENV = {
    "WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP", "BFM_OUTPUT_DIR", "BFM_RESUMABLE_OUTPUT_DIR",
    "BFM_META_STUDY", "BFM_META_VARIANT", "BFM_META_LAUNCH_ID", "BFM_META_CONFIG_HASH",
    "BFM_META_LABEL", "BFM_META_NOTE", "BFM_META_STUDIES_COMMIT",
}
RECORD_NAME = "cedar_launch.json"
RENDERED_NAME = "cedar_sessions_submitted.yaml"


def load_session_config(path: str) -> dict:
    """Read ``session_cedar.yaml``; refuse missing keys and an image without a tag."""
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    missing = sorted(SESSION_KEYS - set(cfg))
    if missing:
        sys.exit(f"[cedar] {path} is missing {missing}")
    if ":" not in str(cfg["image"]):
        sys.exit(f"[cedar] image {cfg['image']!r} needs an explicit tag (name:tag); `latest` moves")
    cfg.setdefault("secrets", {})
    cfg.setdefault("env", {})
    return cfg


def _wandb_routing(sweep: dict, base_command: list) -> list[str]:
    """``wandb.entity=`` / ``wandb.project=`` overrides from the sweep's top level.

    No sweep controller runs, so nothing else routes the runs: the wrapper passes Hydra's
    ``wandb`` config to ``wandb.init``, whose defaults are AIND-disRNN/test. A command token
    that already sets the same value is kept; a different value, or a grid axis over either
    key, is refused.
    """
    extra = []
    for key in ("entity", "project"):
        value = sweep.get(key)
        if not value:
            continue
        if f"wandb.{key}" in (sweep.get("parameters") or {}):
            sys.exit(f"[cedar] the grid varies wandb.{key}; one launch logs to one W&B {key}")
        given = [str(t).split("=", 1)[1] for t in base_command if str(t).startswith(f"wandb.{key}=")]
        if any(g != str(value) for g in given):
            sys.exit(f"[cedar] the sweep's {key} is {value!r} but its command sets wandb.{key}={given[0]}")
        if not given:
            extra.append(f"wandb.{key}={value}")
    return extra


def build_plan(sweep_file: str, cfg: dict, label: str | None, note: str | None, provenance: dict) -> dict:
    """One session per grid point: its name, run_hpc command and environment."""
    sweep_path = Path(sweep_file)
    sweep = yaml.safe_load(sweep_path.read_text())
    if sweep.get("method") != "grid":
        sys.exit(f"[cedar] only method: grid is supported, got {sweep.get('method')!r}")
    base_command = sweep.get("command") or []
    if SWEEP_ARGS_PLACEHOLDER not in base_command:
        sys.exit(f"[cedar] sweep 'command' must include {SWEEP_ARGS_PLACEHOLDER}; got {base_command!r}")
    grid = _grid_points(sweep.get("parameters", {}))
    if not grid:
        sys.exit("[cedar] sweep has no grid points")
    routing = _wandb_routing(sweep, base_command)

    study, variant = _study_variant(sweep_path)
    launch_id = _seattle_launch_id()
    group = f"{variant}@{launch_id}"
    id_base = _slug(f"{variant}-{launch_id}")
    meta = {"BFM_META_STUDY": study, "BFM_META_VARIANT": variant, "BFM_META_LAUNCH_ID": launch_id,
            "BFM_META_CONFIG_HASH": hashlib.sha1(sweep_path.read_bytes()).hexdigest()[:8]}
    if provenance.get("studies_git_commit"):
        meta["BFM_META_STUDIES_COMMIT"] = provenance["studies_git_commit"]
    if label:
        meta["BFM_META_LABEL"] = label
    if note:
        meta["BFM_META_NOTE"] = note
    base_env = {k: str(v) for k, v in cfg["env"].items() if k not in MANAGED_ENV}

    sessions = []
    for index, point in enumerate(grid):
        overrides = [_override_str(k, v) for k, v in point.items()]
        run_cmd = []
        for token in base_command:
            run_cmd.extend(overrides + routing if token == SWEEP_ARGS_PLACEHOLDER else [str(token)])
        name = f"{id_base}-{index}"
        if len(name) > 63:
            sys.exit(f"[cedar] session name {name!r} is longer than 63 characters; shorten the variant name")
        env = {**base_env, **meta, "WANDB_RUN_ID": _run_id(id_base, overrides),
               "WANDB_RESUME": "allow", "WANDB_RUN_GROUP": group}
        # pin_runtime_refs reads and rewrites Beaker-shaped envVars lists in place
        sessions.append({"name": name, "overrides": overrides, "run_cmd": run_cmd,
                         "envVars": [{"name": k, "value": v} for k, v in env.items()]})
    print(f"[cedar] study={study} variant={variant} launch_id={launch_id} group={group} "
          f"sessions={len(sessions)}")
    return {"group": group, "launch_id": launch_id, "sessions": sessions}


def create_command(cfg: dict, session: dict) -> list[str]:
    """The ``cedar session create`` argv for one session."""
    cmd = [CEDAR, "session", "create", session["name"], "--workspace-name", cfg["workspace"],
           "--cluster-name", cfg["cluster"], "--image", cfg["image"], "--gpu", str(cfg["gpu"]),
           "--cpu", str(cfg["cpu"]), "--ram", str(cfg["ram"]), "--max-live-time", str(cfg["max_live_time"]),
           "--all-ssh-keys"]
    for secret, var in cfg["secrets"].items():
        cmd += ["--secret", f"{secret}={var}"]
    for item in session["envVars"]:
        cmd += ["--env", f"{item['name']}={item['value']}"]
    return cmd


def kickoff_command(session: dict) -> str:
    """The remote shell line that starts the run detached and returns at once.

    Only the run itself goes to the background, with all three standard streams redirected:
    a trailing ``&`` after an ``&&`` chain would background the whole chain as a subshell
    that keeps the SSH connection open for as long as the run lasts.
    """
    run = " ".join(shlex.quote(t) for t in session["run_cmd"])
    return ('mkdir -p "$CEDAR_OUTPUTS_DIR/run" && cd /scratch || exit 1; '
            'BFM_OUTPUT_DIR="$CEDAR_OUTPUTS_DIR/run" BFM_RESUMABLE_OUTPUT_DIR="$CEDAR_OUTPUTS_DIR/run" '
            'nohup setsid bash "$BFM_ROOT/aind-dynamic-foraging-bfm-wrapper/beaker/entrypoint.sh" '
            f'{run} > "$CEDAR_OUTPUTS_DIR/run.log" 2>&1 < /dev/null & echo started')


def _cedar_json(*args: str, run=subprocess.run) -> dict:
    out = run([CEDAR, *args, "--format", "json"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or out.stdout.strip())
    return json.loads(out.stdout or "{}")


def image_record_cedar(cfg: dict, run=subprocess.run) -> dict:
    """The record's ``image``: the configured name:tag and the manifest digest it resolves to."""
    name, tag = str(cfg["image"]).rsplit(":", 1)
    digest = None
    try:
        versions = _cedar_json("image", "show", name, "--workspace-name", cfg["workspace"],
                               "--query", tag, run=run).get("versions") or []
        digest = next((v["digest"] for v in versions if tag in (v.get("tags") or [])), None)
        if digest is None:
            raise RuntimeError(f"no version tagged {tag!r}")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[cedar] WARNING: could not resolve image {cfg['image']!r} to its digest ({exc})")
    return {"platform": "cedar", "name": str(cfg["image"]), "digest": digest}


def create_session(cfg: dict, session: dict, run=subprocess.run) -> None:
    """Create one session; raise if it does not reach READY."""
    out = run(create_command(cfg, session), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"session {session['name']} did not start: "
                           f"{(out.stderr or out.stdout).strip().splitlines()[-1:]}")


def session_id(cfg: dict, session: dict, run=subprocess.run) -> str:
    """The id (sess_...) of a created session."""
    sid = (_cedar_json("session", "get", session["name"], "--workspace-name", cfg["workspace"], run=run)
           .get("session") or {}).get("id")
    if not sid:
        raise RuntimeError(f"session {session['name']} was created, but its id could not be read")
    return sid


def start_run(cfg: dict, session: dict, sid: str, run=subprocess.run) -> None:
    """Start the run in a created session, detached, and return once it is started."""
    host = f"cedar-{cfg['workspace']}-{session['name']}"
    kick = run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", host,
                kickoff_command(session)], capture_output=True, text=True, timeout=120)
    if kick.returncode != 0 or "started" not in kick.stdout:
        raise RuntimeError(f"session {session['name']} ({sid}) started, but its run did not: "
                           f"{kick.stderr.strip()[-300:]}")
    print(f"[cedar] {session['name']} ({sid}): run started")


def save_record(sweep_file: str, session_file: str, plan: dict, refs: list[dict], output_dir: Path,
                provenance: dict, dirty_override: dict | None, label: str | None, note: str | None) -> None:
    """Write ``cedar_launch.json`` and copy the manifests beside it.

    ``refs`` holds every session that was created; one whose id could not be read has
    ``id: None``, so it stays out of the v2 ``job_refs`` but is still listed under ``sessions``.
    ``n_tasks`` counts created sessions, ``n_planned`` the grid points.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in (sweep_file, session_file):
        shutil.copy2(f, output_dir / Path(f).name)
    by_name = {r["name"]: r["id"] for r in refs}
    record = {
        **v2_record_fields(
            provenance, sweep_path=sweep_file, platform="cedar", wandb_group=plan["group"], label=label,
            note=note, job_refs=refs, manifests=[Path(sweep_file).name, Path(session_file).name, RENDERED_NAME],
            n_tasks=len(refs), produced_by="launch_cedar.py"),
        "mode": "cedar-grid",
        "wandb_group": plan["group"],
        "n_tasks": len(refs),
        "n_planned": len(plan["sessions"]),
        "sessions": [{"name": s["name"], "created": s["name"] in by_name, "id": by_name.get(s["name"]),
                      "overrides": s["overrides"],
                      "wandb_run_id": next(e["value"] for e in s["envVars"] if e["name"] == "WANDB_RUN_ID")}
                     for s in plan["sessions"]],
        "dispatcher_commit": provenance.get("dispatcher_git_commit"),
        **record_fields(provenance, dirty_override),
        "sweep_file": str(sweep_file),
        "session_file": str(session_file),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / RECORD_NAME).write_text(json.dumps(record, indent=2))
    print(f"[cedar] saved record {output_dir / RECORD_NAME}")


def main() -> None:
    p = argparse.ArgumentParser(description="Launch a grid as one Cedar session per grid point.")
    p.add_argument("--sweep", required=True, help="method: grid W&B sweep YAML (platform-neutral)")
    p.add_argument("--session", required=True, help="session_cedar.yaml: workspace, cluster, image, resources")
    p.add_argument("--output-dir", required=True, help="where the record goes: the variant's launch_record/<label>")
    p.add_argument("--label", default=None)
    p.add_argument("--note", default=None, help="why this run exists and what it should teach us")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands and write the plan; create no session (uses no session names)")
    p.add_argument("--diagnostic-dirty-ok", default=None, metavar="REASON",
                   help="launch even though studies/<study>/ has uncommitted changes (diagnostic runs only)")
    p.add_argument("--allow-live-data", action="store_true",
                   help="launch on the live (unpinned) database; for smoke tests only, never a result")
    args = p.parse_args()

    # the CLI's update notice would land in the JSON the launcher parses
    os.environ["CEDAR_NO_UPDATE_CHECK"] = "1"
    cfg = load_session_config(args.session)
    output_dir = Path(args.output_dir)
    variant_dir = Path(args.sweep).resolve().parent
    if args.dry_run and (output_dir.resolve() == variant_dir or variant_dir in output_dir.resolve().parents):
        sys.exit("[cedar] --dry-run writes its plan to --output-dir; point it outside the variant, or the "
                 "uncommitted files make the real launch refuse")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        sys.exit(f"[cedar] {output_dir} is not empty; a launch record is never overwritten, so pick a new --output-dir")
    provenance = launch_provenance(Path(args.sweep), REPO_ROOT)
    print_warnings(provenance, "[cedar] ")
    try:
        dirty_override = check_studies_clean(provenance, args.diagnostic_dirty_ok)
    except DirtyStudiesError as exc:
        sys.exit(f"[cedar] refusing to launch: {exc}")
    plan = build_plan(args.sweep, cfg, args.label, args.note, provenance)
    try:
        check_launch_identity(args.sweep, plan["group"], args.label, args.note)
    except LaunchIdentityError as exc:
        sys.exit(f"[cedar] refusing to launch: {exc}")
    if not args.dry_run and subprocess.run([CEDAR, "auth", "whoami"], capture_output=True).returncode != 0:
        sys.exit("[cedar] not signed in to Cedar: run `cedar auth login` first")
    try:
        pins = pin_runtime_refs({"tasks": plan["sessions"]})
    except (RuntimeError, ValueError) as exc:
        sys.exit(f"[cedar] runtime ref resolution failed: {exc}")
    provenance = with_runtime_shas(provenance, runtime_shas_from_pins(pins))
    try:
        reader = runtime_config_reader(REPO_ROOT, runtime_dispatcher_sha(runtime_shas_from_pins(pins)),
                                       provenance.setdefault("provenance_warnings", []))
        data = data_record(yaml.safe_load(Path(args.sweep).read_text()), reader)
    except DataConfigError as exc:
        sys.exit(f"[cedar] refusing to launch: {exc}")
    if (why := live_data_refusal(data)) and not args.allow_live_data:
        sys.exit(f"[cedar] refusing to launch: {why}")
    provenance = {**provenance, "data": data, "image": image_record_cedar(cfg)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RENDERED_NAME).write_text(yaml.safe_dump(
        {"session_config": cfg, "sessions": [{**s, "create": create_command(cfg, s), "kickoff": kickoff_command(s)}
                                             for s in plan["sessions"]]}, sort_keys=False))
    refs, failure = [], None
    try:
        for s in plan["sessions"]:
            if args.dry_run:
                print("[cedar] would run:", shlex.join(create_command(cfg, s)))
                print("[cedar] then ssh:", kickoff_command(s))
                continue
            try:
                create_session(cfg, s)
                # recorded as soon as it exists, even if its id cannot be read or its run fails to start
                ref = {"type": "cedar_session", "id": None, "name": s["name"]}
                refs.append(ref)
                ref["id"] = session_id(cfg, s)
                start_run(cfg, s, ref["id"])
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                failure = exc
                break
    finally:
        # also on Ctrl+C: sessions already started keep running, so their record must exist
        save_record(args.sweep, args.session, plan, refs, output_dir, provenance, dirty_override,
                    args.label, args.note)
    if failure:
        sys.exit(f"[cedar] stopped after creating {len(refs)} of {len(plan['sessions'])} sessions: {failure}")
    print("[cedar] done" + (" (dry run: nothing created)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
