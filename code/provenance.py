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


def study_dir(sweep_path: Path | str) -> Path:
    """``studies/<study>/`` containing the sweep, or the sweep's own directory off that layout."""
    p = Path(sweep_path).resolve()
    parts = p.parts
    if "studies" in parts and parts.index("studies") + 1 < len(parts):
        return Path(*parts[: parts.index("studies") + 2])
    return p.parent


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
    if studies["root"]:
        paths, err = dirty_paths(studies["root"], study)
        if err:
            warnings.append(f"studies: {err}")
    else:
        paths = None   # no repository: cleanliness cannot be verified either
    out = {
        "studies_git_root": studies["root"],
        "studies_git_commit": studies["commit"],
        "studies_git_branch": studies["branch"],
        "studies_git_dirty": studies["dirty"],
        "studies_dirty_paths": paths,
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
