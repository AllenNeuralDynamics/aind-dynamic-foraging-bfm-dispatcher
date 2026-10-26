"""Library-only clients for W&B sweep creation and Beaker submission.

WHY THIS MODULE EXISTS: we want ONE launcher code path
(`launch_beaker.py` / `launch_beaker_resumable.py`) that runs unmodified on
BOTH the Allen HPC / Code Ocean side AND the Claude Science sandbox (the Mac
agent environment orchestrating launches per
`docs/claude-science-workflow.md`) -- not two maintained copies. Both
launchers used to shell out to the `wandb` and `beaker` CLIs via `subprocess`,
which works fine on Code Ocean / HPC (both installed by
`environment/postInstall`) but breaks the sandbox in two independent ways:

  1. `wandb sweep <file>` spawns a local `wandb-core` helper process that binds
     a Unix socket and writes to `~/.config/wandb` -- both blocked in the
     Claude Science sandbox, so sweep creation fails outright there.
  2. The `beaker` CLI is a compiled Go binary with no portable install path in
     the sandbox (no apt/brew, no prebuilt wheel) -- it simply can't be put on
     PATH there.

Rather than special-case the sandbox (an `if sandbox: ... else: subprocess...`
branch, or a second sandbox-only launcher), this module swaps BOTH CLI calls
for their underlying HTTP/library equivalents, which work identically
everywhere Python + network access are available -- HPC, Code Ocean, and the
sandbox alike:

  * `create_wandb_sweep` -- the same GraphQL `UpsertSweep` mutation the `wandb`
    CLI issues under the hood (bypasses the `wandb-core` service process
    entirely, so it can't hit the socket/config-file restriction above).
  * `get_beaker_client` / `submit_beaker_experiment` -- the official `beaker-py`
    client, which talks to the Beaker server directly over HTTPS (no CLI
    binary required, so there's nothing to install).

These are used unconditionally by both launchers now, on every platform --
not just when running in the sandbox. That is the point: one code path that
happens to also be more portable than the CLI it replaced, rather than a
sandbox-specific fork. Verified end-to-end (sweep creation, Beaker submission,
run completion) from the Claude Science sandbox, where the old CLI-based code
could not run at all -- see `docs/claude-science-workflow.md` for the broader
Mac/HPC/Beaker architecture this fits into.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

WANDB_GQL_URL = "https://api.wandb.ai/graphql"
GITHUB_API_URL = "https://api.github.com"
RUNTIME_REF_REPOSITORIES = {
    "WRAPPER_REF": "aind-dynamic-foraging-bfm-wrapper",
    "DISPATCHER_REF": "aind-dynamic-foraging-bfm-dispatcher",
    "FORAGING_MODELS_REF": "aind-dynamic-foraging-models",
}
_FULL_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")

_UPSERT_SWEEP_MUTATION = """
mutation UpsertSweep(
    $id: ID,
    $config: String,
    $description: String,
    $entityName: String,
    $projectName: String,
    $controller: JSONString,
    $scheduler: JSONString,
    $state: String,
    $priorRunsFilters: JSONString,
    $displayName: String,
) {
    upsertSweep(input: {
        id: $id,
        config: $config,
        description: $description,
        entityName: $entityName,
        projectName: $projectName,
        controller: $controller,
        scheduler: $scheduler,
        state: $state,
        priorRunsFilters: $priorRunsFilters,
        displayName: $displayName,
    }) {
        sweep {
            name
            project { id name entity { id name } }
        }
        configValidationWarnings
    }
}
"""


def resolve_github_ref(repository: str, ref: str) -> str:
    """Resolve a GitHub branch, tag, or commit to a full commit SHA."""
    if _FULL_GIT_SHA.fullmatch(ref):
        return ref.lower()

    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    url = (
        f"{GITHUB_API_URL}/repos/AllenNeuralDynamics/{repository}/commits/"
        f"{quote(ref, safe='')}"
    )
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"could not resolve AllenNeuralDynamics/{repository}@{ref}: {exc}"
        ) from exc
    sha = response.json().get("sha", "")
    if not _FULL_GIT_SHA.fullmatch(sha):
        raise RuntimeError(
            f"GitHub returned an invalid commit for "
            f"AllenNeuralDynamics/{repository}@{ref}: {sha!r}"
        )
    return sha.lower()


def pin_runtime_refs(
    spec: dict, cache: dict[tuple[str, str], str] | None = None
) -> dict[tuple[str, str], str]:
    """Replace runtime REF values in every task with full GitHub commit SHAs."""
    resolved = cache if cache is not None else {}
    for task in spec.get("tasks", []):
        env_by_name = {
            item.get("name"): item
            for item in task.get("envVars", [])
            if item.get("name") in RUNTIME_REF_REPOSITORIES
        }
        missing = set(RUNTIME_REF_REPOSITORIES) - set(env_by_name)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                f"task {task.get('name', '<unnamed>')!r} is missing runtime refs: {names}"
            )
        for name, repository in RUNTIME_REF_REPOSITORIES.items():
            item = env_by_name[name]
            ref = item.get("value")
            if not isinstance(ref, str) or not ref:
                raise ValueError(
                    f"task {task.get('name', '<unnamed>')!r} has invalid {name}: {ref!r}"
                )
            key = (repository, ref)
            if key not in resolved:
                resolved[key] = resolve_github_ref(repository, ref)
                print(f"[runtime-refs] {name}: {ref} -> {resolved[key]}")
            item["value"] = resolved[key]
    return resolved


def create_wandb_sweep(sweep_file: str, wandb_api_key: str | None = None) -> str:
    """Create a W&B sweep from a sweep YAML file via the GraphQL API directly.

    Equivalent to `wandb sweep <sweep_file>`, but makes one HTTPS POST instead of
    spawning the `wandb` CLI (which starts a local `wandb-core` service process).
    Returns the full sweep path `entity/project/sweep_id`, same format the CLI's
    `wandb agent <SWEEP_ID>` line would have printed.
    """
    api_key = wandb_api_key or os.environ["WANDB_API_KEY"]
    cfg = yaml.safe_load(Path(sweep_file).read_text())
    entity, project = cfg["entity"], cfg["project"]

    resp = requests.post(
        WANDB_GQL_URL,
        auth=("api", api_key),
        json={
            "query": _UPSERT_SWEEP_MUTATION,
            "variables": {
                "config": yaml.safe_dump(cfg),
                "entityName": entity,
                "projectName": project,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"wandb UpsertSweep failed: {body['errors']}")
    sweep_name = body["data"]["upsertSweep"]["sweep"]["name"]
    return f"{entity}/{project}/{sweep_name}"


def beaker_token() -> str:
    """The Beaker user token: `BEAKER_TOKEN`, else `user_token` from the CLI's config file.

    The file is `$BEAKER_CONFIG` or `~/.beaker/config.yml` (what `beaker account login`
    writes). A missing or unreadable file (sandboxes that block dotfiles under `$HOME`)
    counts as absent, so the environment variable alone still works everywhere.
    """
    token = os.environ.get("BEAKER_TOKEN")
    if token:
        return token
    path = Path(os.environ.get("BEAKER_CONFIG") or Path.home() / ".beaker" / "config.yml")
    try:
        token = (yaml.safe_load(path.read_text()) or {}).get("user_token")
    except (OSError, yaml.YAMLError, AttributeError):
        token = None
    if not token:
        raise KeyError(f"no Beaker token: BEAKER_TOKEN is unset and {path} has no readable user_token")
    return token


def get_beaker_client(workspace: str | None = None):
    """Build an authenticated `beaker.Beaker` client from `beaker_token()`.

    Uses `beaker.Config(user_token=...)` directly instead of `Beaker.from_env()` /
    `Beaker.from_config()`, so a sandbox that blocks `~/.beaker/config.yml` still works
    from `BEAKER_TOKEN`; the file is only a fallback when the variable is unset.
    """
    from beaker import Beaker, Config

    token = beaker_token()
    cfg = Config(user_token=token, default_org="ai1", default_workspace=workspace)
    beaker = Beaker(cfg)
    beaker._timeout = 60  # beaker-py's default is too short for large-payload requests
    # (e.g. an 80-task experiment.create()); matches the check_gpu_availability.py workaround.
    return beaker


def submit_beaker_experiment(spec_path: str, workspace: str, name: str | None = None) -> str:
    """Submit a rendered Beaker experiment spec file. Returns the experiment id."""
    from beaker import ExperimentSpec

    beaker = get_beaker_client(workspace=workspace)
    spec = ExperimentSpec.from_file(spec_path)
    # Pass `name`/`workspace` as keywords only -- ExperimentClient.create()'s
    # positional-arg parser rejects `name=None` passed positionally (it can't tell
    # that apart from a malformed call), so `create(name, spec, ...)` breaks when
    # name is None. `create(spec, name=name, ...)` handles None cleanly.
    experiment = beaker.experiment.create(spec, name=name, workspace=workspace)
    return experiment.id
