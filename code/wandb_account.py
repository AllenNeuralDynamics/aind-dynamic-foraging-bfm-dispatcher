"""Pick the W&B account for a study launch (dispatcher #225).

A study's README names its W&B entity and project (``wandb:`` frontmatter). A study launch must
agree with it, and the launcher must use the key that belongs to that entity, so a forgotten
``export WANDB_API_KEY`` cannot send a ``NeuralDynamics`` study to the personal account.

- ``check_wandb_account``: refuse a launch whose sweep YAML names another entity than the README.
- ``use_key_for``: if ``~/.config/wandb/api_key.<entity>`` exists, set ``WANDB_API_KEY`` in this
  process (sbatch ``--export=ALL`` carries it to the jobs). Otherwise the SDK uses ``~/.netrc``.
  The key is never printed, recorded or written to an archived file.
- ``check_entity_access``: refuse before anything is created if the key cannot reach the entity.

``code/config/config.yaml`` keeps ``AIND-disRNN`` as the default entity, so studies that do not
name another one are not touched.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from provenance import study_and_variant, study_dir

DEFAULT_ENTITY = "AIND-disRNN"
KEY_DIR_ENV = "BFM_WANDB_KEY_DIR"
_ENTITY_OVERRIDE = re.compile(r"(?:^|\s)\+?wandb\.entity=[\"']?([^\s\"']+)")


class WandbAccountError(RuntimeError):
    pass


def key_file(entity: str) -> Path:
    """Where the key for ``entity`` lives by convention (the file may not exist)."""
    base = Path(os.environ.get(KEY_DIR_ENV) or Path.home() / ".config" / "wandb")
    return base / f"api_key.{entity}"


def study_wandb(sweep_path: Path | str) -> dict | None:
    """The ``wandb:`` block of the study README above a sweep; None off the study layout."""
    study, _ = study_and_variant(sweep_path)
    if not study:
        return None
    readme = study_dir(sweep_path) / "README.md"
    if not readme.is_file():
        return None
    text = readme.read_text()
    if not text.startswith("---\n"):
        return None
    front = yaml.safe_load(text.split("\n---", 1)[0][4:]) or {}
    return front.get("wandb") or None


def _sweep_entity(sweep: dict) -> tuple[str | None, str | None]:
    """``(top-level entity, entity override in the command)`` of a sweep YAML."""
    override = None
    for item in sweep.get("command") or []:
        m = _ENTITY_OVERRIDE.search(str(item))
        if m:
            override = m.group(1)
    return sweep.get("entity"), override


def check_wandb_account(sweep_path: Path | str, platform: str) -> str | None:
    """Refuse a study launch whose sweep sends runs to another entity than the study README.

    ``platform`` is ``"hpc"`` (``wandb sweep`` routes on the top-level ``entity``) or
    ``"beaker"`` (grid points run directly, so only the ``wandb.entity=`` override in the
    command counts). Returns the entity, or None when the sweep is not a study launch.
    """
    declared = study_wandb(sweep_path)
    if not declared or not declared.get("entity"):
        return None
    want = declared["entity"]
    top, override = _sweep_entity(yaml.safe_load(Path(sweep_path).read_text()) or {})
    seen = {"top-level `entity`": top, "`wandb.entity=` override": override}
    problems = [f"{where} is {got!r}" for where, got in seen.items() if got and got != want]
    if want != DEFAULT_ENTITY:
        need, got = ("top-level `entity`", top) if platform == "hpc" else ("`wandb.entity=` override", override)
        if not got:
            problems.append(f"the sweep sets no {need}, so runs would go to the default {DEFAULT_ENTITY!r}")
    if problems:
        raise WandbAccountError(
            f"the study README names W&B entity {want!r} but " + "; ".join(problems)
            + f". Set the sweep to {want!r} (and the project to {declared.get('project')!r}).")
    return want


def use_key_for(entity: str) -> Path | None:
    """Export the entity's key to this process if its key file exists; return the file used."""
    f = key_file(entity)
    if not f.is_file():
        return None
    key = f.read_text().strip()
    if not key:
        raise WandbAccountError(f"{f} is empty")
    os.environ["WANDB_API_KEY"] = key
    return f


def check_entity_access(entity: str, teams_fn=None) -> None:
    """Refuse if the key now in use cannot reach ``entity``. A non-default entity only.

    ``teams_fn`` returns ``(username, [teams])`` for the current key; the default asks W&B.
    If W&B cannot be asked (no network or SDK), the check is skipped with a warning.
    """
    if entity == DEFAULT_ENTITY:
        return
    try:
        username, teams = (teams_fn or _viewer)()
    except Exception as exc:  # network, missing SDK, bad key: the launch itself will say more
        print(f"WARNING: could not check W&B access to {entity!r}: {type(exc).__name__}")
        return
    if entity != username and entity not in teams:
        raise WandbAccountError(
            f"the W&B key in use (user {username!r}) is not a member of {entity!r}. "
            f"Put that entity's key in {key_file(entity)} (the launcher exports it).")


def _viewer() -> tuple[str, list[str]]:
    import wandb
    v = wandb.Api(timeout=30).viewer
    return v.username, list(v.teams)


def check_beaker_secret(spec: dict, entity: str) -> None:
    """Refuse a non-default entity whose spec reads ``WANDB_API_KEY`` from another entity's secret.

    By convention the secret name carries the entity, lowercase (``han-wandb-api-key-neuraldynamics``).
    """
    if entity == DEFAULT_ENTITY:
        return
    for task in spec.get("tasks", []):
        for e in task.get("envVars", []):
            secret = e.get("secret")
            if e.get("name") == "WANDB_API_KEY" and secret and entity.lower() not in secret.lower():
                raise WandbAccountError(
                    f"the experiment's WANDB_API_KEY comes from Beaker secret {secret!r}, which is not "
                    f"for {entity!r}. Use the secret whose name ends in -wandb-api-key-{entity.lower()}.")


def prepare(sweep_path: Path | str, platform: str, teams_fn=None, spec: dict | None = None) -> str | None:
    """The launcher's one call: check the sweep, pick the key, check access. Returns the entity."""
    entity = check_wandb_account(sweep_path, platform)
    if entity is None:
        return None
    used = use_key_for(entity)
    print(f"W&B account: entity={entity} key={'file ' + str(used) if used else '~/.netrc'}")
    if platform == "hpc":
        check_entity_access(entity, teams_fn)
    elif spec is not None:  # Beaker jobs read their key from a Beaker secret, not this process
        check_beaker_secret(spec, entity)
    elif entity != DEFAULT_ENTITY:
        print(f"WARNING: the experiment template's WANDB_API_KEY secret must hold the {entity!r} key "
              "(not checked here)")
    return entity
