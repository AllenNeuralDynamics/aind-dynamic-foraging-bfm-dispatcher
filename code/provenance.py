"""Launch-time git provenance shared by the dispatcher launchers (dispatcher #172).

Why two roots. Until the studies split (#124) the sweep YAML and the launcher lived in one
repository, so one SHA located both. After it, a launch record has to answer two questions
separately: which *studies* commit held the sweep that was launched, and which *dispatcher*
commit ran the launcher. Field names follow ``docs/repo-split-plan.md`` § "Cross-repo
provenance contract". Pre-split both roots resolve to the same repository and every new
field equals the old ``dispatcher_commit``; the values only diverge once the sweep lives
elsewhere, which is the point.

Dirty checkouts. A launch from an uncommitted sweep cannot be reproduced from any SHA, so
:func:`check_studies_clean` refuses it unless a diagnostic-only override is given, and the
override is written into the record. The scope is the *study directory*
(``studies/<study>/``), not the whole checkout: an unrelated edit elsewhere does not change
what was launched, and pre-split the whole checkout is the dispatcher, where uncommitted
work is routine. The whole-root flag is still recorded. Decided 2026-10-23; see the plan's
"Resolved decisions".

Git environment. The Claude Science sandbox checks out the dispatcher through an external
git dir (``GIT_DIR``/``GIT_WORK_TREE`` exported, see the git-session-isolation skill; #109).
Those variables override repository discovery for *every* ``git`` call, so a query aimed at
a different checkout (the studies repo) would silently describe the dispatcher. ``_git``
therefore keeps the ambient variables only for a query inside ``GIT_WORK_TREE`` and strips
them otherwise, and every failed query is reported as a warning rather than swallowed.

Only the stdlib and ``git`` on PATH; importable from any launcher without network access.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# repo name as used in beaker_client.RUNTIME_REF_REPOSITORIES -> record field
RUNTIME_SHA_FIELDS = {
    "aind-dynamic-foraging-bfm-wrapper": "wrapper_git_commit",
    "aind-dynamic-foraging-models": "foraging_models_git_commit",
}


# Variables that redirect git's repository discovery away from `-C cwd`.
_GIT_DISCOVERY_VARS = (
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR",
)


class DirtyStudiesError(RuntimeError):
    """The study directory holding the sweep has uncommitted changes."""


class LaunchIdentityError(RuntimeError):
    """A study launch would mint a W&B group, label or note the studies schema rejects."""


# Launch-record v2: AllenNeuralDynamics/aind-dynamic-foraging-bfm-studies
# schemas/launch-record.v2.schema.json. The field definitions live there; this module only
# fills them. Rules for the group: that repo's docs/adr/0001-one-launch-one-wandb-group.md.
RECORD_SCHEMA_VERSION = 2
_VARIANT_NAME = re.compile(r"^[a-z0-9]+([._-][a-z0-9]+)*$")       # schemas/layout.yaml
_WANDB_GROUP = re.compile(r"^[A-Za-z0-9][\w.-]*@[0-9]{8}-[0-9]{6}$")
_MIN_NOTE = 10


def _git_env(cwd: Path) -> dict[str, str]:
    """The environment for a git query at ``cwd``: ambient only inside ``GIT_WORK_TREE``."""
    env = dict(os.environ)
    work_tree = env.get("GIT_WORK_TREE")
    if work_tree:
        try:
            Path(cwd).resolve().relative_to(Path(work_tree).resolve())
            return env
        except ValueError:
            pass
    for var in _GIT_DISCOVERY_VARS:
        env.pop(var, None)
    return env


def _git(cwd: Path, *args: str, strip: bool = True) -> tuple[str | None, str | None]:
    """``(stdout, None)`` on success, ``(None, error)`` on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10,
            env=_git_env(Path(cwd)),
        )
    except Exception as exc:  # git missing, timeout
        return None, f"{type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return None, (out.stderr.strip() or f"git exited {out.returncode}")
    return (out.stdout.strip() if strip else out.stdout), None


def git_state(path: Path | str) -> dict:
    """``root``/``commit``/``branch``/``dirty`` of the repository containing ``path``.

    ``path`` may be a file or a directory. Outside a checkout every value is None; ``dirty``
    is a bool (whole working tree) when known.
    """
    p = Path(path).resolve()
    top, err = _git(p if p.is_dir() else p.parent, "rev-parse", "--show-toplevel")
    if not top:
        return {"root": None, "commit": None, "branch": None, "dirty": None,
                "error": f"{p}: {err or 'no git repository found'}"}
    root = Path(top)
    commit, err = _git(root, "rev-parse", "HEAD")
    branch, _ = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status, _ = _git(root, "status", "--porcelain")
    return {
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
        "error": None if commit else f"{root}: {err or 'HEAD did not resolve'}",
    }


def _studies_index(parts: tuple[str, ...]) -> int | None:
    """Index of the ``studies`` component that roots ``studies/<study>/...``.

    Prefer the last ``studies`` with ``variants`` two components below it, so a checkout
    whose own path contains ``studies`` (``.../studies/studies/<study>/variants/...``, a
    worktree named ``studies``) still resolves to the study (#182); else the last
    ``studies`` with a component after it.
    """
    hits = [i for i, part in enumerate(parts[:-1]) if part == "studies"]
    anchored = [i for i in hits if i + 2 < len(parts) and parts[i + 2] == "variants"]
    return (anchored or hits or [None])[-1]


def study_and_variant(sweep_path: Path | str) -> tuple[str | None, str | None]:
    """``(study, variant)`` of a ``studies/<study>/variants/<variant>/`` path; None off the layout."""
    parts = Path(sweep_path).resolve().parts
    i = _studies_index(parts)
    if i is None:
        return None, None
    variant = parts[i + 3] if i + 3 < len(parts) and parts[i + 2] == "variants" else None
    return parts[i + 1], variant


def study_dir(sweep_path: Path | str) -> Path:
    """``studies/<study>/`` containing the sweep, or the sweep's own directory off that layout."""
    p = Path(sweep_path).resolve()
    i = _studies_index(p.parts)
    return Path(*p.parts[: i + 2]) if i is not None else p.parent


def dirty_paths(root: Path | str, subdir: Path | str) -> tuple[list[str] | None, str | None]:
    """``(paths, None)`` under ``subdir`` with uncommitted changes, or ``(None, error)``.

    ``None`` means the status query failed, which is *unknown*, not clean: callers must not
    treat it as a verified clean study (``check_studies_clean`` refuses it).
    """
    # -z: NUL-separated `XY path` entries; not stripped, because XY may start with a space.
    status, err = _git(Path(root), "status", "--porcelain", "-z", "--", str(subdir), strip=False)
    if status is None:
        return None, f"{root}: git status failed: {err}"
    return [entry[3:] for entry in status.split("\0") if len(entry) > 3], None


def is_tracked(root: Path | str, path: Path | str) -> bool | None:
    """Whether ``path`` is tracked by the repository at ``root``; None if git failed.

    ``git status`` says nothing about ignored files, so an empty status does not prove the
    sweep itself is committed: a sweep under an ignored directory (``results/``) would pass
    as clean while no commit could reproduce it.
    """
    out, err = _git(Path(root), "ls-files", "--error-unmatch", "--", str(path))
    if out is not None:
        return True
    if err and "did not match any file" in err:
        return False
    return None


def runtime_shas_from_pins(resolved: dict[tuple[str, str], str]) -> dict[str, str | list[str]]:
    """``{repo_name: sha}`` from ``beaker_client.pin_runtime_refs``'s ``{(repo, ref): sha}``.

    A repo pinned at two different refs within one spec is recorded as a sorted list rather
    than silently picking one.
    """
    by_repo: dict[str, set[str]] = {}
    for (repo, _ref), sha in resolved.items():
        by_repo.setdefault(repo, set()).add(sha)
    return {repo: (next(iter(s)) if len(s) == 1 else sorted(s)) for repo, s in by_repo.items()}


def installed_vcs_commit(site_packages: list[Path] | tuple[Path, ...], dist_prefix: str) -> dict:
    """Commit and version of a pip-installed distribution, from its ``direct_url.json``.

    pip writes ``vcs_info.commit_id`` for ``git+https://...@ref`` installs, which is how the
    HPC conda envs carry ``aind-dynamic-foraging-models`` when a branch is needed. A release
    install has no VCS info; then only the version is known and ``commit`` is None.
    """
    for sp in site_packages:
        for info in sorted(Path(sp).glob(f"{dist_prefix}-*.dist-info")):
            version = info.name[len(dist_prefix) + 1:-len(".dist-info")]
            commit = requested = None
            try:
                direct = json.loads((info / "direct_url.json").read_text())
                vcs = direct.get("vcs_info") or {}
                requested = vcs.get("requested_revision")
                if _FULL_SHA.fullmatch(str(vcs.get("commit_id", ""))):
                    commit = vcs["commit_id"]
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
            return {"commit": commit, "version": version, "requested_revision": requested}
    return {"commit": None, "version": None, "requested_revision": None}


def launch_provenance(
    sweep_path: Path | str,
    dispatcher_root: Path | str,
    runtime_shas: dict[str, str | list[str] | None] | None = None,
) -> dict:
    """The contract's launch-record fields for one launch.

    ``runtime_shas`` maps repo name (``aind-dynamic-foraging-bfm-wrapper``,
    ``aind-dynamic-foraging-models``) to the full SHA the launcher resolved for the job; a
    value that is not a full SHA is recorded as None rather than as a branch name.
    ``provenance_warnings`` lists every git query that failed, with its path and git's
    message, so a None commit is never silent (#109); launchers print them.
    """
    studies = git_state(sweep_path)
    dispatcher = git_state(dispatcher_root)
    study = study_dir(sweep_path)
    warnings = [f"{label}: {state['error']}"
                for label, state in (("studies", studies), ("dispatcher", dispatcher))
                if state.get("error")]
    tracked = None
    if studies["root"]:
        paths, err = dirty_paths(studies["root"], study)
        if err:
            warnings.append(f"studies: {err}")
        tracked = is_tracked(studies["root"], sweep_path)
        if tracked is False:
            warnings.append(f"studies: {Path(sweep_path).resolve()} is not tracked by git "
                            "(untracked or ignored), so no commit reproduces it")
    else:
        paths = None   # no repository: cleanliness cannot be verified either
    out = {
        "studies_git_root": studies["root"],
        "studies_git_commit": studies["commit"],
        "studies_git_branch": studies["branch"],
        "studies_git_dirty": studies["dirty"],
        "studies_dirty_paths": paths,
        "studies_sweep_tracked": tracked,
        "dispatcher_git_commit": dispatcher["commit"],
        "dispatcher_git_branch": dispatcher["branch"],
        "dispatcher_git_dirty": dispatcher["dirty"],
        "provenance_warnings": warnings,
    }
    for repo, field in RUNTIME_SHA_FIELDS.items():
        value = (runtime_shas or {}).get(repo)
        if isinstance(value, list):
            out[field] = [v for v in value if _FULL_SHA.fullmatch(str(v))] or None
        else:
            out[field] = value if value and _FULL_SHA.fullmatch(str(value)) else None
    return out


def with_runtime_shas(provenance: dict, runtime_shas: dict | None) -> dict:
    """``provenance`` with the wrapper / foraging-models fields filled from ``runtime_shas``."""
    if not runtime_shas:
        return dict(provenance)
    out = dict(provenance)
    for repo, field in RUNTIME_SHA_FIELDS.items():
        value = runtime_shas.get(repo)
        if isinstance(value, list):
            out[field] = [v for v in value if _FULL_SHA.fullmatch(str(v))] or None
        elif value and _FULL_SHA.fullmatch(str(value)):
            out[field] = value
    return out


def check_studies_clean(provenance: dict, override_reason: str | None) -> dict | None:
    """Refuse a launch whose study directory is dirty, unless ``override_reason`` is given.

    Returns the override block to write into the record (``None`` when the study directory
    was verified clean), or raises :class:`DirtyStudiesError`. ``studies_dirty_paths`` of
    ``None`` means the check could not run (no repository, or ``git status`` failed); that is
    refused too, because an unverified study is not a clean one.
    """
    paths = provenance.get("studies_dirty_paths")
    if paths == [] and provenance.get("studies_sweep_tracked") is False:
        if override_reason:
            return {"reason": override_reason, "paths": [], "untracked_sweep": True}
        raise DirtyStudiesError(
            "the sweep file is not tracked by git (untracked or ignored), so no commit "
            "reproduces this launch. Commit it, or pass --diagnostic-dirty-ok '<reason>' for "
            "a diagnostic run; the reason is written into the launch record."
        )
    if paths == []:
        return None
    if paths is None:
        why = "; ".join(provenance.get("provenance_warnings") or []) or "no git repository found"
        if override_reason:
            return {"reason": override_reason, "paths": None, "unverified": why}
        raise DirtyStudiesError(
            "could not verify that the study directory is clean (" + why + "), so this "
            "launch could not be tied to a commit. Fix the checkout, or pass "
            "--diagnostic-dirty-ok '<reason>' for a diagnostic run; the reason is written "
            "into the launch record."
        )
    if override_reason:
        return {"reason": override_reason, "paths": list(paths)}
    shown = "\n  ".join(paths[:20]) + ("\n  ..." if len(paths) > 20 else "")
    raise DirtyStudiesError(
        "the study directory has uncommitted changes, so this launch could not be "
        "reproduced from any commit:\n  " + shown + "\nCommit (or stash) them, or pass "
        "--diagnostic-dirty-ok '<reason>' for a diagnostic run; the reason is written "
        "into the launch record."
    )


def record_fields(provenance: dict, dirty_override: dict | None) -> dict:
    """The provenance block as written into a launch record."""
    out = {k: v for k, v in provenance.items() if k != "studies_git_root"}
    out["studies_dirty_override"] = dirty_override
    return out


def print_warnings(provenance: dict, prefix: str = "") -> None:
    """Surface failed git queries on stderr; a null commit in a record must not be silent."""
    import sys
    for w in provenance.get("provenance_warnings") or []:
        print(f"{prefix}WARNING: provenance: {w}", file=sys.stderr)


def check_launch_identity(sweep_path: Path | str, group: str, label: str | None,
                          note: str | None) -> None:
    """Refuse, before anything is submitted, a study launch the studies schema would reject.

    A W&B group cannot be renamed once runs exist, so this is the only point where a wrong
    one is cheap. For a sweep under ``studies/<study>/variants/<variant>/`` (ADR 0001):
    the variant directory name is valid, the group is exactly ``<variant>@<launch_id>``, and
    ``--label`` / ``--note`` are given (the record needs both). An ad-hoc sweep off that
    layout is not a study launch and is not checked.
    """
    study, variant = study_and_variant(sweep_path)
    if not (study and variant):
        return
    problems = []
    if not _VARIANT_NAME.fullmatch(variant):
        problems.append(f"variant directory {variant!r} is not lowercase words joined by - _ or .")
    if not _WANDB_GROUP.fullmatch(group or "") or group.split("@")[0] != variant:
        problems.append(f"W&B group {group!r} is not {variant}@<YYYYMMDD-HHMMSS>")
    if not (label or "").strip():
        problems.append("--label is required for a study launch")
    if len((note or "").strip()) < _MIN_NOTE:
        problems.append(f"--note is required for a study launch (why the run exists and what it "
                        f"should teach us; at least {_MIN_NOTE} characters)")
    if problems:
        raise LaunchIdentityError("; ".join(problems))


def v2_record_fields(provenance: dict, *, sweep_path: Path | str, platform: str,
                     wandb_group: str | None, label: str | None, note: str | None,
                     job_refs: list[dict], manifests: list[str], n_tasks: int | None,
                     produced_by: str) -> dict:
    """The launch-record v2 fields for a submitted study launch; ``{}`` otherwise.

    Launchers merge these with their legacy keys, which older readers (the studies
    ``validate_provenance.py``, ``launch_record.load_records``) still use. An ad-hoc sweep
    off the ``studies/<study>/variants/<variant>/`` layout may omit ``--label``/``--note``
    and lands in no study, so it keeps the legacy shape rather than a v2 record the schema
    would reject. So does a record written before anything was submitted (``--no-submit``
    previews, the HPC launcher's pre-submit write): v2 requires a job reference, and the
    record becomes v2 on the write that has one. ``revisions`` holds only full SHAs; a repo
    whose SHA did not resolve is left out, so the studies checker reports the gap instead
    of the record hiding it.
    """
    study, variant = study_and_variant(sweep_path)
    job_refs = [r for r in job_refs if r.get("id")]
    if not (study and variant and job_refs):
        return {}
    import datetime
    from zoneinfo import ZoneInfo

    revisions = {}
    for repo, field in (("studies", "studies_git_commit"), ("dispatcher", "dispatcher_git_commit"),
                        ("wrapper", "wrapper_git_commit"),
                        ("foraging_models", "foraging_models_git_commit")):
        value = provenance.get(field)
        value = value[0] if isinstance(value, list) and len(value) == 1 else value
        if isinstance(value, str) and _FULL_SHA.fullmatch(value):
            revisions[repo] = value
    now = datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")
    return {
        "_meta": {"schema_version": RECORD_SCHEMA_VERSION, "produced_by": produced_by,
                  "produced_at_pt": now},
        "kind": "initial",
        "platform": platform,
        "wandb_group": wandb_group,
        "label": label,
        "note": note,
        "job_refs": job_refs,
        "revisions": revisions,
        "manifests": list(manifests),
        "n_tasks": n_tasks,
        # what the runs trained on and ran in (launch_inputs.py; studies#25)
        **{k: provenance[k] for k in ("data", "image") if provenance.get(k)},
    }
