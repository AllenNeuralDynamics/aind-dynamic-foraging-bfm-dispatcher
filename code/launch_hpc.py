"""Create a W&B sweep and submit Allen HPC / SLURM sweep agents.

This launcher does three things:

1. Reads a sweep YAML (e.g. ``code/hpc/sweeps/synthetic_num_sessions_disrnn.yaml``).
2. Injects per-run lineage fields into the sweep's ``command`` list so every
   run started by every agent records where it came from. Injected fields:

       +meta.study                    study name from studies/<study>/...
       +meta.variant                  variant name from variants/<variant>/...
       +meta.launch_id                Seattle-time launch id
       +meta.config_hash              hash of the sweep YAML at launch
       +meta.label                    optional human launch label
       +meta.note                     optional launch intent / rationale
       +meta.slurm_job_id             SLURM_JOB_ID resolved on the compute node
       +meta.slurm_array_job_id       SLURM_ARRAY_JOB_ID resolved on the compute node
       +meta.slurm_array_task_id      SLURM_ARRAY_TASK_ID resolved on the compute node
       +meta.dispatcher_git_commit    dispatcher SHA at launch time
       +meta.dispatcher_git_branch    dispatcher branch at launch time
       +meta.dispatcher_git_dirty     dispatcher dirty flag
       +meta.wrapper_git_commit       wrapper SHA used by run_hpc
       +meta.wrapper_git_branch       wrapper branch used by run_hpc
       +meta.wrapper_git_dirty        wrapper dirty flag
       +meta.studies_commit           SHA of the repo holding the sweep (the studies
                                      repo after #124; the dispatcher before it)
       +meta.studies_git_branch       its branch
       +meta.studies_git_dirty        its dirty flag (whole checkout)
       +meta.sweep_yaml               sweep YAML path, relative to dispatcher root
       +meta.owner                    Unix user who launched the sweep
       +meta.launcher_cmd             exact argv used to invoke this launcher
       +meta.mode                     "cpu" or "gpu"

   These appear in each W&B run's config under ``meta.*`` and are filterable
   in the Runs/Sweeps UI, so any run can be traced back to the exact code
   and command that produced it without relying on a separate registry file.
   Runs launched from ``studies/<study>/variants/<variant>/`` are also grouped
   in W&B as ``<variant>@<launch_id>``, matching the Beaker launchers.

3. Creates the sweep via ``wandb sweep`` (using a temp YAML patched with the
   lineage fields) and submits a SLURM array job of agents.
4. Writes ``hpc_sweep.json`` (+ a copy of the sweep YAML) to ``--output-dir``
   (default ``results/hpc_<launch_id>/`` under the dispatcher root) with the W&B
   sweep id, the SLURM ids and the two-root provenance block from
   ``code/provenance.py`` -- the HPC counterpart of ``beaker_resumable.json``.
   Copy it into the variant's ``launch_record/`` like the Beaker one.

A sweep whose ``studies/<study>/`` directory has uncommitted changes is refused
(``--diagnostic-dirty-ok REASON`` overrides, and records the reason).

Notes on W&B sweep behavior:
    - When running under a sweep, W&B ignores per-run ``wandb.entity`` and
      ``wandb.project`` overrides; sweep routing comes from the top-level
      ``entity``/``project`` fields in the sweep YAML.
    - Hydra ``+meta.*`` overrides ARE honored on each run, which is why this
      launcher uses them to carry lineage.

Example:
    python code/launch_hpc.py \
        --sweep-yaml code/hpc/sweeps/synthetic_num_sessions_disrnn.yaml \
        --mode cpu \
        --sbatch-extra=--array=0-1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from provenance import (
    DirtyStudiesError, LaunchIdentityError, check_launch_identity, check_studies_clean,
    git_state, installed_vcs_commit, launch_provenance, print_warnings, record_fields,
    study_and_variant, v2_record_fields,
)


def _parse_sweep_id(stdout: str, stderr: str) -> str | None:
    text = "\n".join([stdout, stderr])

    patterns = [
        r"wandb\s+agent\s+([\w.-]+/[\w.-]+/[\w-]+)",
        r"Created sweep with ID:\s*([\w-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if "/" not in candidate and pattern.endswith("([\\w-]+)"):
                return None
            return candidate

    return None


def _run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )


def _load_export_env(path: Path) -> dict[str, str]:
    """Load simple `export KEY=value` assignments from a shell env file."""
    values: dict[str, str] = {}
    expansion_env = os.environ.copy()

    def _expand(value: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            key = match.group(1) or match.group(2)
            return expansion_env.get(key, match.group(0))

        return os.path.expanduser(
            re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)|\$\{([^}]+)\}", _replace, value)
        )

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("export "):
            continue
        for assignment in shlex.split(line[len("export "):]):
            if "=" not in assignment:
                continue
            key, value = assignment.split("=", 1)
            value = _expand(value)
            values[key] = value
            expansion_env[key] = value
    return values


def _path_for_meta(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_array_spec(sbatch_extra: str) -> str | None:
    tokens = shlex.split(sbatch_extra)
    for idx, token in enumerate(tokens):
        if token.startswith("--array="):
            return token.split("=", 1)[1]
        if token == "--array" and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def _read_array_spec_from_script(slurm_script: Path) -> str | None:
    for line in slurm_script.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#SBATCH") and "--array=" in stripped:
            return stripped.split("--array=", 1)[1].strip()
    return None


def _count_array_tasks(array_spec: str) -> int:
    spec = array_spec.split("%", 1)[0].strip()
    if not spec:
        return 1

    total = 0
    for chunk in spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue

        if "-" not in piece:
            total += 1
            continue

        range_part, *step_part = piece.split(":")
        start_s, end_s = range_part.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        step = int(step_part[0]) if step_part else 1
        if step <= 0:
            raise ValueError(f"Invalid array step in spec: {array_spec}")
        if end < start:
            raise ValueError(f"Invalid array range in spec: {array_spec}")
        total += (end - start) // step + 1

    return max(total, 1)


def _git_info(repo_root: Path) -> dict[str, str]:
    """Capture lineage info from the git repo at repo_root.

    Returns commit sha, branch, and a 'dirty' flag (yes/no) so each run in
    the sweep records the exact code state that launched it. Resolved per
    target worktree by ``provenance.git_state``, so a sandbox ``GIT_DIR`` /
    ``GIT_WORK_TREE`` exported for the dispatcher cannot make the wrapper
    query describe the dispatcher.
    """
    state = git_state(repo_root)
    return {
        "commit": state["commit"] or "unknown",
        "branch": state["branch"] or "unknown",
        "dirty": "yes" if state["dirty"] else "no",
    }


def _seattle_launch_id() -> str:
    """Readable, unique-per-launch id = Seattle local timestamp."""
    return _dt.datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")


def _runtime_env_site_packages(conda_sh: str | None, slurm_script: Path) -> list[Path]:
    """``site-packages`` dirs of the conda env the SLURM script activates.

    ``CONDA_SH`` (user.env) is ``<prefix>/etc/profile.d/conda.sh``. The script activates
    the renamed env and falls back to the pre-rename one (``conda activate a || conda
    activate b``); the first that exists under the prefix is the one the job will run in.
    Empty when nothing is known.
    """
    if not conda_sh:
        return []
    try:
        names = re.findall(r"conda activate\s+(\S+)", slurm_script.read_text())
    except OSError:
        return []
    prefix = Path(conda_sh).resolve().parents[2]  # strip etc/profile.d/conda.sh
    for name in names:
        env_dir = prefix / "envs" / name
        if env_dir.is_dir():
            return sorted(env_dir.glob("lib/python3.*/site-packages"))
    return []


def _studies_lineage(provenance: dict) -> dict[str, str]:
    """The ``+meta.*`` fields naming the checkout that held the sweep (see ``provenance.py``).

    ``studies_commit`` is the key the wrapper also fills from ``BFM_META_STUDIES_COMMIT`` on
    Beaker, so both routes land on one name; branch/dirty follow this launcher's existing
    ``*_git_branch`` / ``*_git_dirty`` convention.
    """
    return {
        "studies_commit": provenance.get("studies_git_commit") or "unknown",
        "studies_git_branch": provenance.get("studies_git_branch") or "unknown",
        "studies_git_dirty": "yes" if provenance.get("studies_git_dirty") else "no",
    }


# The lineage-patched sweep handed to `wandb sweep`, archived beside the launch record.
SUBMITTED_SWEEP = "sweep_submitted.yaml"


def _save_record(
    output_dir: Path, *, sweep_file: Path, sweep_path: str | None, group: str,
    array_jobid: str | None, autostop_jobid: str | None, n_array_tasks: int,
    agent_count: int, mode: str, provenance: dict, dirty_override: dict | None,
    label: str | None = None, note: str | None = None,
) -> dict:
    """Write ``hpc_sweep.json`` + a copy of the sweep YAML; the HPC launch record.

    A launch-record v2 (``provenance.v2_record_fields``) plus the legacy launcher keys.

    ``wandb_sweep_id`` (the short id) is the durable handle the provenance validator
    reconciles against W&B; SLURM ids are recycled and age out of ``sacct``.
    ``dispatcher_commit`` is kept alongside ``dispatcher_git_commit`` because the validator
    and older readers key on it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_file, output_dir / Path(sweep_file).name)
    entity_project, _, sweep_short = (sweep_path or "").rpartition("/")
    record = {
        **v2_record_fields(
            provenance, sweep_path=sweep_file, platform="hpc", wandb_group=group,
            label=label, note=note,
            job_refs=[{"type": "wandb_sweep", "id": sweep_short or None},
                      {"type": "slurm_array_job", "id": array_jobid}],
            manifests=[Path(sweep_file).name]
            + ([SUBMITTED_SWEEP] if (output_dir / SUBMITTED_SWEEP).exists() else []),
            n_tasks=n_array_tasks, produced_by="launch_hpc.py"),
        "mode": "hpc-sweep",
        "wandb_group": group,
        "wandb_sweep_id": sweep_short or None,
        "wandb_sweep_path": sweep_path,
        "sweep_url": (f"https://wandb.ai/{entity_project}/sweeps/{sweep_short}"
                      if sweep_path else None),
        "slurm_array_job_id": array_jobid,
        "slurm_autostop_job_id": autostop_jobid,
        "n_array_tasks": n_array_tasks,
        "agent_count": agent_count,
        "slurm_mode": mode,
        "dispatcher_commit": provenance.get("dispatcher_git_commit"),
        **record_fields(provenance, dirty_override),
        "sweep_file": str(sweep_file),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    (output_dir / "hpc_sweep.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved launch record: {output_dir / 'hpc_sweep.json'}")
    return record


def _study_variant(sweep_path: Path) -> tuple[str, str]:
    """Derive (study, variant) from studies/<study>/variants/<variant>/sweep.yaml."""
    study, variant = study_and_variant(sweep_path)
    return study or "adhoc", variant or sweep_path.stem


# Max characters for an injected +meta.* value. Long scalars written into a
# run's inputs.yaml get FOLDED across lines by the YAML dumper; the wrapper's
# post-training loader (_parse_simple_yaml) is a one-key-per-line parser that
# RAISES "Unexpected indentation" on any continuation line, which aborts the
# auto held-out fine-tuning step. Keeping every meta value on a single short
# line (whitespace-collapsed, capped well under the ~80-col fold width incl.
# indent) guarantees inputs.yaml stays reloadable. Full provenance still lives
# in git (commit/branch), the committed sweep YAML, and shell history.
_META_VALUE_MAX_LEN = 60


def _meta_safe(value: object) -> str:
    """Make an injected meta value safe to round-trip through inputs.yaml.

    A YAML scalar only gets FOLDED across lines by the dumper when it contains
    whitespace to break at; a long single-token value (e.g. a filesystem path)
    is always emitted on one line. So we only need to truncate values that
    both contain a space AND exceed the fold-safe width. Truncation uses an
    ASCII marker ('...') because a truncated no-marker value could still be
    fine, but the marker must never introduce a character Hydra's override
    lexer rejects (a Unicode ellipsis does). Values with spaces are quoted by
    _hydra_value for the CLI, so '...' inside them is harmless there too.
    """
    collapsed = " ".join(str(value).split())
    if " " in collapsed and len(collapsed) > _META_VALUE_MAX_LEN:
        collapsed = collapsed[: _META_VALUE_MAX_LEN - 3].rstrip() + "..."
    return collapsed


def _hydra_value(value: object) -> str:
    """Render a value for a Hydra CLI override."""
    sval = str(value)
    if any(c in sval for c in " =,'\"[]{}"):
        escaped = sval.replace("'", "\\'")
        return f"'{escaped}'"
    return sval


def _inject_lineage_into_command(
    sweep_cfg: dict,
    lineage: dict[str, str],
    *,
    group: str | None = None,
) -> dict:
    """Append Hydra provenance overrides to the sweep 'command' list.

    Sweep mode ignores per-run wandb.entity/project overrides, but Hydra
    overrides in the command list are still applied to each run's config,
    so each W&B run carries this lineage in its config under `meta.*`.
    """
    cmd = list(sweep_cfg.get("command", []))
    for key, value in lineage.items():
        # `+meta.<key>=<value>` adds the field (won't error if missing in schema).
        # Quote values containing chars that Hydra's override parser rejects
        # (spaces, '=', commas, etc.) so they survive as plain strings.
        # `_meta_safe` collapses whitespace + caps length so the value can't
        # fold across lines in the run's inputs.yaml (which would break the
        # post-training loader and abort auto held-out fine-tuning).
        cmd.append(f"+meta.{key}={_hydra_value(_meta_safe(value))}")
    if group:
        # Also pass the group in config so wandb.init receives it directly.
        cmd.append(f"+wandb.group={_hydra_value(group)}")
    # Platform-native HPC ids are only known on the compute node. Use OmegaConf
    # env resolvers so each run records its exact Slurm job/array coordinates.
    cmd.extend(
        [
            "+meta.slurm_job_id='${oc.env:SLURM_JOB_ID,unknown}'",
            "+meta.slurm_array_job_id='${oc.env:SLURM_ARRAY_JOB_ID,unknown}'",
            "+meta.slurm_array_task_id='${oc.env:SLURM_ARRAY_TASK_ID,unknown}'",
        ]
    )
    # HPC has no /results mount (that is the Beaker/Code Ocean container path the
    # model configs default to), so the end-of-training auto held-out fine-tune
    # aborts with "[Errno 13] Permission denied: '/results'" and the run logs NO
    # heldout/* metrics -- silently, as a WARNING, after training has succeeded.
    # Every HPC sweep needs the override, so inject it here rather than relying on
    # each sweep YAML to remember (which is how studies 01 and 02 both lost their
    # held-out metrics). Skip if the sweep already set it explicitly.
    # ...but only for models that HAVE a training block. The cognitive baselines (hb_*,
    # baseline_rl_*) do their own held-out scoring inside the trainer and define no
    # `model.training`, so an unconditional override is not merely redundant there -- Hydra
    # refuses to override a key that does not exist and the run dies during config
    # composition, before the trainer is ever constructed:
    #     ConfigCompositionException: Could not override
    #     'model.training.auto_heldout_finetune.output_root'
    # That is why the HB ladder submitted through a hand-written sbatch instead of this
    # launcher. Deciding from the model config itself, rather than a name pattern, keeps new
    # model configs working without touching this list.
    if not any("auto_heldout_finetune.output_root" in str(c) for c in cmd):
        if _model_has_training_block(cmd):
            cmd.append(
                "model.training.auto_heldout_finetune.output_root="
                "'${oc.env:BFM_HELDOUT_ROOT,${oc.env:DISRNN_HELDOUT_ROOT,${oc.env:HOME}/outputs/heldout_subject_finetuning}}'"
            )
    sweep_cfg["command"] = cmd
    return sweep_cfg


def _model_has_training_block(cmd: list) -> bool:
    """Does the model config this sweep selects define `training`?

    Read from the dispatcher's own config tree rather than inferred from the model name, so
    a new config is classified by what it contains. Unknown or unreadable configs return
    True, keeping the previous behaviour for everything that is not clearly a trainer-side
    model: the failure this guard prevents is loud and immediate, while the failure the
    override prevents (studies 01 and 02 silently losing every `heldout/*` metric) is not.
    """
    model_name = None
    for entry in cmd:
        text = str(entry)
        if text.startswith("model="):
            model_name = text.split("=", 1)[1].strip()
    if not model_name:
        return True
    config_path = Path(__file__).resolve().parent / "config" / "model" / f"{model_name}.yaml"
    if not config_path.exists():
        return True
    try:
        model_cfg = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError):
        return True
    return not isinstance(model_cfg, dict) or "training" in model_cfg


def _estimate_total_grid_runs(sweep_yaml: Path) -> int | None:
    sweep_cfg = yaml.safe_load(sweep_yaml.read_text())
    if not isinstance(sweep_cfg, dict):
        return None
    if str(sweep_cfg.get("method", "")).lower() != "grid":
        return None

    parameters = sweep_cfg.get("parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        return None

    counts: list[int] = []
    for _, param_cfg in parameters.items():
        if not isinstance(param_cfg, dict):
            return None
        values = param_cfg.get("values")
        if not isinstance(values, list) or len(values) == 0:
            return None
        counts.append(len(values))

    return math.prod(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create W&B sweep and submit Allen HPC SLURM array agents."
    )
    parser.add_argument(
        "--sweep-yaml",
        default=None,
        help=(
            "Path to W&B sweep YAML file. Defaults to "
            "code/hpc/sweeps/synthetic_num_sessions_disrnn.yaml."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Choose CPU or GPU sweep agent script.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=None,
        help="Override AGENT_COUNT passed to sbatch. If omitted, estimate from sweep size and array size.",
    )
    parser.add_argument(
        "--sbatch-extra",
        default="",
        help="Extra sbatch arguments, e.g. '--array=0-9'.",
    )
    parser.add_argument(
        "--gpu-type",
        default=None,
        help=(
            "Constrain GPU runs to a specific GPU type (e.g. 'v100', 'a100', "
            "'l40s', 'h200'). Only used when --mode gpu. Translates to "
            "'--gres=gpu:<type>:1' for sbatch, overriding the default in the "
            "GPU slurm script. See code/hpc/README.md for guidance."
        ),
    )
    parser.add_argument(
        "--no-autostop",
        action="store_true",
        help=(
            "Skip submitting the cleanup job that calls `wandb sweep --stop` "
            "after the agent array drains. Use this when you want the sweep "
            "to stay open so you can submit more agents to it later."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--wrapper-root",
        default=None,
        help=(
            "Path to the sibling aind-dynamic-foraging-bfm-wrapper repo. Defaults to "
            "../aind-dynamic-foraging-bfm-wrapper relative to the dispatcher root, "
            "falling back to ../aind-disrnn-wrapper for pre-rename checkouts."
        ),
    )
    parser.add_argument(
        "--user-env",
        default=None,
        help=(
            "Path to per-user SLURM env file. Defaults to "
            "code/hpc/slurm/user.env."
        ),
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional human label for this launch (stamped to W&B meta.label).",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Free-text background + intent (stamped to W&B meta.note).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for hpc_sweep.json and the sweep copy (the launch record). "
            "Defaults to results/hpc_<launch_id>/ under the dispatcher root."
        ),
    )
    parser.add_argument(
        "--diagnostic-dirty-ok",
        default=None,
        metavar="REASON",
        help=(
            "Launch even though studies/<study>/ has uncommitted changes; REASON is "
            "written into the launch record (diagnostic runs only, never a result)."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_sweep_yaml = repo_root / "code" / "hpc" / "sweeps" / "synthetic_num_sessions_disrnn.yaml"
    sweep_yaml_arg = args.sweep_yaml or str(default_sweep_yaml)
    sweep_yaml = (repo_root / sweep_yaml_arg).resolve()
    if not sweep_yaml.exists():
        raise FileNotFoundError(f"Sweep YAML not found: {sweep_yaml}")

    if args.wrapper_root:
        wrapper_root = Path(args.wrapper_root).expanduser().resolve()
    else:
        # Prefer the post-rename directory name, but keep resolving a checkout
        # still called aind-disrnn-wrapper: the GitHub rename does not rename
        # anyone's existing clone (ADR-0007).
        candidates = [
            (repo_root.parent / "aind-dynamic-foraging-bfm-wrapper").resolve(),
            (repo_root.parent / "aind-disrnn-wrapper").resolve(),
        ]
        wrapper_root = next((c for c in candidates if c.exists()), candidates[0])
    if not wrapper_root.exists():
        raise FileNotFoundError(f"Wrapper repo not found: {wrapper_root}")

    user_env = (
        Path(args.user_env).expanduser().resolve()
        if args.user_env
        else (repo_root / "code" / "hpc" / "slurm" / "user.env").resolve()
    )
    if not user_env.exists():
        raise FileNotFoundError(
            f"Per-user HPC env file not found: {user_env}. "
            "Copy code/hpc/slurm/user.env.example to user.env and edit it."
        )
    launch_env = os.environ.copy()
    launch_env.update(_load_export_env(user_env))

    slurm_script = repo_root / "code" / "hpc" / "slurm" / f"wandb_sweep_{args.mode}.slurm"
    if not slurm_script.exists():
        raise FileNotFoundError(f"SLURM script not found: {slurm_script}")

    array_spec = _extract_array_spec(args.sbatch_extra)
    if not array_spec:
        array_spec = _read_array_spec_from_script(slurm_script)
    num_array_tasks = _count_array_tasks(array_spec) if array_spec else 1

    computed_agent_count = args.agent_count
    total_grid_runs = _estimate_total_grid_runs(sweep_yaml)
    if computed_agent_count is None and total_grid_runs is not None:
        computed_agent_count = max(1, math.ceil(total_grid_runs / num_array_tasks))
        print(
            "Computed AGENT_COUNT="
            f"{computed_agent_count} from total_grid_runs={total_grid_runs} and num_array_tasks={num_array_tasks}"
        )
    elif computed_agent_count is None:
        # fallback to script defaults if sweep size cannot be inferred
        computed_agent_count = 3 if args.mode == "cpu" else 15
        print(
            "Could not infer total sweep runs; using fallback AGENT_COUNT="
            f"{computed_agent_count} for mode={args.mode}"
        )

    # Report planned coverage so the user can spot under-sampling early.
    # Each W&B agent runs `AGENT_COUNT` runs, so planned runs is the
    # product of array tasks and the agent count.
    planned_runs = num_array_tasks * computed_agent_count
    if total_grid_runs is not None:
        print(
            f"Planned coverage: {planned_runs} runs "
            f"({num_array_tasks} array tasks x {computed_agent_count} agents) "
            f"out of {total_grid_runs} grid points."
        )
        if planned_runs < total_grid_runs:
            print(
                f"WARNING: planned_runs ({planned_runs}) < total_grid_runs "
                f"({total_grid_runs}); only a subset of the grid will be sampled."
            )

    study, variant = _study_variant(sweep_yaml)
    launch_id = _seattle_launch_id()
    config_hash = hashlib.sha1(sweep_yaml.read_bytes()).hexdigest()[:8]
    group = f"{variant}@{launch_id}"
    provenance_env = {
        "WANDB_RUN_GROUP": group,
        "BFM_META_STUDY": study,
        "BFM_META_VARIANT": variant,
        "BFM_META_LAUNCH_ID": launch_id,
        "BFM_META_CONFIG_HASH": config_hash,
    }
    # Two-root provenance (#172): the checkout holding the sweep and the one running this
    # launcher are recorded separately, because after the studies split (#124) they differ.
    wrapper_git = _git_info(wrapper_root)
    # The models package is pip-installed into the runtime env (sometimes from a branch, via
    # git+https); its commit lives in the env's direct_url.json, not in any checkout here.
    models = installed_vcs_commit(
        _runtime_env_site_packages(launch_env.get("CONDA_SH"), slurm_script),
        "aind_dynamic_foraging_models",
    )
    provenance = launch_provenance(
        sweep_yaml, repo_root,
        {"aind-dynamic-foraging-bfm-wrapper": wrapper_git["commit"],
         "aind-dynamic-foraging-models": models["commit"]},
    )
    provenance["foraging_models_version"] = models["version"]
    if models["version"] and not models["commit"]:
        provenance["provenance_warnings"].append(
            f"foraging-models: version {models['version']} in the runtime env was not "
            "installed from git, so no commit is recorded")
    elif not models["version"]:
        provenance["provenance_warnings"].append(
            "foraging-models: not found in the runtime env's site-packages")
    print_warnings(provenance)
    try:
        dirty_override = check_studies_clean(provenance, args.diagnostic_dirty_ok)
    except DirtyStudiesError as exc:
        raise SystemExit(f"Refusing to launch: {exc}")
    if provenance.get("studies_git_commit"):
        provenance_env["BFM_META_STUDIES_COMMIT"] = provenance["studies_git_commit"]
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else repo_root / "results" / f"hpc_{launch_id}"
    )

    print(
        f"Study provenance: study={study} variant={variant} "
        f"launch_id={launch_id} group={group} config_hash={config_hash}"
    )
    try:
        check_launch_identity(sweep_yaml, group, args.label, args.note)
    except LaunchIdentityError as exc:
        raise SystemExit(f"Refusing to launch: {exc}")

    sweep_cmd = ["wandb", "sweep", str(sweep_yaml)]
    print("Running:", " ".join(shlex.quote(x) for x in sweep_cmd))

    export_pairs = {
        "AGENT_COUNT": str(computed_agent_count),
        "WRAPPER_ROOT": str(wrapper_root),
        **provenance_env,
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export_pairs.items())

    if args.dry_run:
        dry_sbatch_cmd: list[str] = [
            "sbatch",
            "--export",
            export_arg,
        ]
        if args.gpu_type and args.mode == "gpu":
            dry_sbatch_cmd.extend([f"--gres=gpu:{args.gpu_type}:1"])
        if args.sbatch_extra:
            dry_sbatch_cmd.extend(shlex.split(args.sbatch_extra))
        dry_sbatch_cmd.extend([str(slurm_script), "<SWEEP_ID>"])
        print("Running:", " ".join(shlex.quote(x) for x in dry_sbatch_cmd))
        print("Launch record provenance (what hpc_sweep.json would carry):")
        print(json.dumps(record_fields(provenance, dirty_override), indent=2))
        print("Dry-run mode enabled; skipping sweep creation and sbatch submit.")
        return

    # Write the launch record before anything is submitted, so an unwritable --output-dir
    # fails here and not after the sweep and the array exist; the final write below fills
    # in the ids.
    _save_record(
        output_dir, sweep_file=sweep_yaml, sweep_path=None, group=group,
        array_jobid=None, autostop_jobid=None, n_array_tasks=num_array_tasks,
        agent_count=computed_agent_count, mode=args.mode,
        provenance=provenance, dirty_override=dirty_override,
        label=args.label, note=args.note,
    )

    # --- Lineage injection ---
    # Build a temp sweep YAML that appends Hydra +meta.* overrides to the
    # command list, so every run in this sweep carries lineage in its W&B
    # config (filterable in the Sweeps/Runs view).
    dispatcher_git = _git_info(repo_root)
    launcher_cmd = " ".join(shlex.quote(a) for a in sys.argv)
    lineage = {
        "study": study,
        "variant": variant,
        "launch_id": launch_id,
        "config_hash": config_hash,
        "dispatcher_git_commit": dispatcher_git["commit"],
        "dispatcher_git_branch": dispatcher_git["branch"],
        "dispatcher_git_dirty": dispatcher_git["dirty"],
        "wrapper_git_commit": wrapper_git["commit"],
        "wrapper_git_branch": wrapper_git["branch"],
        "wrapper_git_dirty": wrapper_git["dirty"],
        **_studies_lineage(provenance),
        "sweep_yaml": _path_for_meta(sweep_yaml, repo_root),
        "owner": getpass.getuser(),
        "launcher_cmd": launcher_cmd,
        "mode": args.mode,
    }
    if args.label:
        lineage["label"] = args.label
    if args.note:
        lineage["note"] = args.note
    sweep_cfg = yaml.safe_load(sweep_yaml.read_text())
    # Suffix the sweep display name with a Seattle-local timestamp so
    # re-launches of the same YAML don't produce indistinguishable entries
    # in the W&B UI. The sweep ID stays unique either way; this only
    # affects the human-readable label. If the YAML has no `name`, leave
    # it alone (let W&B assign its random codename).
    if isinstance(sweep_cfg, dict) and sweep_cfg.get("name"):
        sweep_cfg["name"] = f"{sweep_cfg['name']}-{launch_id}"
        print(f"Sweep display name: {sweep_cfg['name']}")
    sweep_cfg = _inject_lineage_into_command(sweep_cfg, lineage, group=group)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="sweep_", delete=False
    ) as tmp:
        yaml.safe_dump(sweep_cfg, tmp, sort_keys=False)
        patched_yaml = Path(tmp.name)
    print(f"Patched sweep YAML with lineage at: {patched_yaml}")
    # What W&B actually receives (group + lineage overrides) is what reproduces the launch;
    # keep it beside the record, which lists it in `manifests`.
    shutil.copy2(patched_yaml, output_dir / SUBMITTED_SWEEP)

    sweep_cmd = ["wandb", "sweep", str(patched_yaml)]
    sweep_result = _run_command(sweep_cmd, cwd=repo_root)
    if sweep_result.stdout:
        print(sweep_result.stdout, end="")
    if sweep_result.stderr:
        print(sweep_result.stderr, end="")
    if sweep_result.returncode != 0:
        raise RuntimeError("wandb sweep command failed")

    sweep_id = _parse_sweep_id(sweep_result.stdout, sweep_result.stderr)
    if not sweep_id:
        raise RuntimeError(
            "Could not parse SWEEP_ID from wandb output. "
            "Please run `wandb sweep` manually and submit sbatch with that ID."
        )

    print(f"Parsed SWEEP_ID: {sweep_id}")
    # The sweep now exists in W&B: persist its id before sbatch, so a failed submission
    # still leaves a record with the durable handle to reconcile against.
    _save_record(
        output_dir, sweep_file=sweep_yaml, sweep_path=sweep_id, group=group,
        array_jobid=None, autostop_jobid=None, n_array_tasks=num_array_tasks,
        agent_count=computed_agent_count, mode=args.mode,
        provenance=provenance, dirty_override=dirty_override,
        label=args.label, note=args.note,
    )

    sbatch_cmd: list[str] = ["sbatch"]
    sbatch_cmd.extend(
        [
            "--export",
            export_arg,
        ]
    )
    if args.gpu_type:
        if args.mode != "gpu":
            print(
                f"WARNING: --gpu-type={args.gpu_type} is ignored in mode={args.mode}."
            )
        else:
            # Overrides the #SBATCH --gres line in the GPU slurm script.
            sbatch_cmd.extend([f"--gres=gpu:{args.gpu_type}:1"])
    if args.sbatch_extra:
        sbatch_cmd.extend(shlex.split(args.sbatch_extra))
    sbatch_cmd.extend([str(slurm_script), sweep_id])

    print("Running:", " ".join(shlex.quote(x) for x in sbatch_cmd))
    sbatch_result = subprocess.run(
        sbatch_cmd,
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
        env=launch_env,
    )
    if sbatch_result.stdout:
        print(sbatch_result.stdout, end="")
    if sbatch_result.stderr:
        print(sbatch_result.stderr, end="")
    if sbatch_result.returncode != 0:
        raise RuntimeError("sbatch command failed")

    array_jobid_match = re.search(r"Submitted batch job\s+(\d+)", sbatch_result.stdout)
    array_jobid = array_jobid_match.group(1) if array_jobid_match else None
    autostop_jobid = None

    # Submit the auto-stop cleanup job: a tiny task that calls
    # `wandb sweep --stop <id>` once every agent-array task has reached a
    # terminal state. Uses --dependency=afterany so a partially-failed sweep
    # still gets marked Finished (no further agents will claim work either way).
    if not args.no_autostop:
        if not array_jobid:
            print(
                "WARNING: could not parse array job ID from sbatch output; "
                "skipping auto-stop. Run `wandb sweep --stop "
                f"{sweep_id}` manually after the sweep drains."
            )
        else:
            stop_script = repo_root / "code" / "hpc" / "slurm" / "wandb_sweep_stop.slurm"
            stop_cmd = [
                "sbatch",
                f"--dependency=afterany:{array_jobid}",
                str(stop_script),
                sweep_id,
            ]
            print("Running:", " ".join(shlex.quote(x) for x in stop_cmd))
            stop_result = subprocess.run(
                stop_cmd,
                cwd=str(repo_root),
                check=False,
                capture_output=True,
                text=True,
                env=launch_env,
            )
            if stop_result.stdout:
                print(stop_result.stdout, end="")
            if stop_result.stderr:
                print(stop_result.stderr, end="")
            if stop_result.returncode != 0:
                print(
                    "WARNING: cleanup sbatch failed; run "
                    f"`wandb sweep --stop {sweep_id}` manually after the sweep drains."
                )
            else:
                stop_match = re.search(r"Submitted batch job\s+(\d+)", stop_result.stdout)
                autostop_jobid = stop_match.group(1) if stop_match else None

    _save_record(
        output_dir, sweep_file=sweep_yaml, sweep_path=sweep_id, group=group,
        array_jobid=array_jobid, autostop_jobid=autostop_jobid,
        n_array_tasks=num_array_tasks, agent_count=computed_agent_count, mode=args.mode,
        provenance=provenance, dirty_override=dirty_override,
        label=args.label, note=args.note,
    )

    # If the user opted out of auto-stop, remind them how to mark the sweep
    # finished manually after their bounded runs complete.
    if args.no_autostop:
        print(
            "\nNote: auto-stop disabled. After all agents finish, the W&B "
            "sweep will still show 'Running'. Stop it explicitly with:\n"
            f"    wandb sweep --stop {sweep_id}"
        )


if __name__ == "__main__":
    main()
