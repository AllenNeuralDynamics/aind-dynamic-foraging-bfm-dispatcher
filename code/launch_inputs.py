"""What a launch trains on and runs in, for its launch record (studies#25).

``data_record`` reads the Hydra data config a sweep selects and the snapshot it pins, and
digests that snapshot's ``snapshot_manifest.json`` (public HTTPS, stdlib only), so a result
names the exact copy of the foraging database it came from. ``image_record_beaker`` resolves
the experiment template's Beaker image name to the immutable image id it points at now;
``image_record_hpc`` names the conda env a SLURM job activates and digests its package list.

Kept apart from ``provenance.py``, which stays stdlib-only: these read YAML and Beaker.
Field shapes: the studies repo's ``schemas/launch-record.v2.schema.json`` (``data``,
``image``).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Callable

import yaml

SNAPSHOT_PREFIX = "s3://aind-scratch-data/aind-dynamic-foraging-cache"
SNAPSHOT_HTTPS = "https://aind-scratch-data.s3.amazonaws.com/aind-dynamic-foraging-cache"
# data config types that read the foraging database (wrapper utils/load_mice_database.py),
# where an unset `snapshot` means the live, growing database
DATABASE_TYPES = {"mice_snapshot"}


class DataConfigError(ValueError):
    """The sweep's data cannot be pinned to one record (e.g. it sweeps data.snapshot)."""


# wrapper entry points that compose the dispatcher's Hydra config; any other runner (e.g. study
# 09's external_transfer.gru_transfer) reads its own data, recorded as `external`
HYDRA_RUNNERS = {"run_hpc"}
ARGS_MACROS = {"${args}", "${args_no_hyphens}", "${args_no_boolean_flags}", "${args_json}", "${args_json_file}"}


def _decode(value) -> str | None:
    """An override value with Hydra's scalar semantics: quoted is a literal string, null is None."""
    if value is None:
        return None
    s = str(value).strip()
    if "${" in s:                       # OmegaConf resolves interpolations even inside quotes
        raise DataConfigError(f"the data override {s!r} is an interpolation resolved only at run time; "
                              "pin a literal value so the launch record can name it")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    if s.lower() == "null" or s in ("", "~"):
        return None
    return s


def _runner(sweep: dict) -> str | None:
    """The module or script the sweep runs (`python -m run_hpc ...` -> run_hpc).

    `-m` is Python's module flag only directly after the interpreter; later it is Hydra's
    multirun flag (`python run_hpc.py -m data=...`)."""
    cmd = [x for x in sweep.get("command") or [] if isinstance(x, str)]
    while cmd and cmd[0] == "${env}":
        cmd = cmd[1:]
    if cmd and (Path(cmd[0]).name.startswith("python") or cmd[0] == "${interpreter}"):
        if len(cmd) > 2 and cmd[1] == "-m":
            return cmd[2]
        cmd = cmd[1:]
    # the program is W&B's `${program}` (the sweep's `program:`) or the first `.py` script
    prog = sweep.get("program") if ("${program}" in cmd or not cmd) else next((x for x in cmd if x.endswith(".py")), None)
    return Path(prog).stem if prog and not str(prog).startswith("$") else None


def runtime_dispatcher_sha(runtime_shas: dict) -> str | None:
    """The one dispatcher commit the tasks run; several mean several possible data configs."""
    sha = runtime_shas.get("aind-dynamic-foraging-bfm-dispatcher")
    if isinstance(sha, list):
        raise DataConfigError(f"the tasks pin different DISPATCHER_REFs {sha}; launch one dispatcher commit "
                              "per launch so one data config describes it")
    return sha


def _key(k: str) -> str:
    """A Hydra override key without its +, ++ or ~ prefix."""
    return str(k).lstrip("+~")


def _effective(sweep: dict, key: str) -> set[str | None] | None:
    """The values ``key`` takes across the sweep's runs, in Hydra's override order.

    The command list is applied in order; W&B expands ``parameters`` where its ``${args...}``
    macro sits (at the end when there is none), so a later override wins. ``~key`` and a null
    value clear it (None). Returns None when the sweep never sets the key (the config default).
    """
    params = {_key(k): spec for k, spec in (sweep.get("parameters") or {}).items() if isinstance(spec, dict)}

    def from_params():
        spec = params[key]
        values = [spec["value"]] if "value" in spec else list(spec.get("values") or [])
        return {_decode(v) for v in values}

    current, expanded = None, False
    cmd = [x for x in sweep.get("command") or [] if isinstance(x, str)]
    for i, item in enumerate(cmd):
        # a flag may take its value as the next token (`--dataset zid`)
        if key.startswith("--") and item == key and i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
            current = {_decode(cmd[i + 1])}
            continue
        if item.strip() in ARGS_MACROS:
            expanded = True
            if key in params:
                current = from_params()
        elif item.startswith("~") and _key(item.split("=", 1)[0]) == key:
            current = {None}
        elif "=" in item and not item.startswith("$") and _key(item.split("=", 1)[0]) == key:
            current = {_decode(item.split("=", 1)[1])}
    if not expanded and key in params:
        current = from_params()
    return current


def file_reader(config_root: Path | str) -> Callable[[str], str | None]:
    """Read ``<config_root>/<rel>`` from the working tree."""
    root = Path(config_root)
    return lambda rel: (root / rel).read_text() if (root / rel).exists() else None


def git_reader(repo_root: Path | str, sha: str) -> Callable[[str], str | None]:
    """Read ``code/config/<rel>`` at commit ``sha`` of the dispatcher checkout (the runtime ref)."""
    from provenance import _git
    return lambda rel: _git(Path(repo_root), "show", f"{sha}:code/config/{rel}", strip=False)[0]


def runtime_config_reader(repo_root: Path | str, runtime_sha: str | None, warnings: list) -> Callable[[str], str | None]:
    """The configs a Beaker run will actually load: the pinned DISPATCHER_REF, else the local tree."""
    local = file_reader(Path(repo_root) / "code" / "config")
    if not runtime_sha:
        return local
    at_ref = git_reader(repo_root, runtime_sha)
    if at_ref("config.yaml") is None:
        msg = (f"data: runtime dispatcher {runtime_sha[:12]} is not in this checkout; "
               "read the data config from the local tree instead (git fetch to check it)")
        warnings.append(msg)
        print(f"[launch_inputs] WARNING: {msg}")
        return local
    return at_ref


def _default_data_config(read: Callable[[str], str | None]) -> str | None:
    """The `data` entry of the root config's Hydra defaults list."""
    root = yaml.safe_load(read("config.yaml") or "") or {}
    for entry in root.get("defaults") or []:
        if isinstance(entry, dict) and "data" in entry:
            return str(entry["data"])
    return None


def fetch_manifest(snapshot: str, timeout: float = 15.0) -> tuple[str | None, dict]:
    """sha256 and parsed body of a snapshot's manifest; (None, {}) if it cannot be read."""
    url = f"{SNAPSHOT_HTTPS}/snapshots/{snapshot}/snapshot_manifest.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
        return hashlib.sha256(body).hexdigest(), json.loads(body)
    except (OSError, ValueError) as exc:
        print(f"[launch_inputs] WARNING: could not read {url} ({exc}); recording the snapshot without its digest")
        return None, {}


def data_record(sweep: dict, config_root: Path | str | Callable[[str], str | None],
                fetch: Callable[[str], tuple[str | None, dict]] = fetch_manifest) -> dict:
    """The launch record's `data`: snapshot (pinned), live (database, unpinned) or generated.

    ``config_root`` is the dispatcher's ``code/config`` directory, or a reader from
    ``runtime_config_reader`` (the configs at the run's pinned dispatcher ref).
    """
    read = config_root if callable(config_root) else file_reader(config_root)
    runner = _runner(sweep)
    if runner and runner not in HYDRA_RUNNERS:
        datasets = _effective(sweep, "--dataset") or set()
        dataset = next(iter(datasets)) if len(datasets) == 1 else None
        return {"source": "external", "config": runner + (f" --dataset={dataset}" if dataset else ""), "type": None}
    names = _effective(sweep, "data")
    if names is not None and len(names) > 1:
        raise DataConfigError(f"the sweep varies the data config over {sorted(map(str, names))}; launch one per launch")
    name = (next(iter(names)) if names else None) or _default_data_config(read)
    cfg = yaml.safe_load(read(f"data/{name}.yaml") or "") or {}
    dtype = cfg.get("type")
    snaps = _effective(sweep, "data.snapshot")
    if snaps is not None and len(snaps) > 1:
        raise DataConfigError(f"the sweep varies data.snapshot over {sorted(map(str, snaps))}; launch one snapshot per launch")
    snapshot = next(iter(snaps)) if snaps is not None else _decode(cfg.get("snapshot"))
    rec = {"config": name, "type": dtype}
    if dtype in DATABASE_TYPES:
        if not snapshot:
            return {"source": "live", **rec, "snapshot": None, "uri": None, "manifest_sha256": None}
        digest, manifest = fetch(snapshot)
        return {"source": "snapshot", **rec, "snapshot": snapshot,
                "uri": f"{SNAPSHOT_PREFIX}/snapshots/{snapshot}/", "manifest_sha256": digest,
                "build_id": manifest.get("latest_build_id"), "n_sessions": manifest.get("n_sessions")}
    if str(dtype or "").startswith("synthetic"):
        return {"source": "generated", **rec}
    # real data that no snapshot pins: as unreproducible as the live database
    return {"source": "live", **rec, "snapshot": None, "uri": None, "manifest_sha256": None}


def live_data_refusal(data: dict) -> str | None:
    """Why a launch on this data is refused without --allow-live-data, or None."""
    if data.get("source") != "live":
        return None
    return (f"data config {data.get('config')!r} reads unpinned data (the live database), so its results "
            "cannot be regenerated. Pin data.snapshot (e.g. the config's dated snapshot), or pass "
            "--allow-live-data for a smoke test.")


def image_record_beaker(template: dict, workspace: str | None,
                        client_factory: Callable | None = None) -> dict | None:
    """The launch record's Beaker `image`: the template's image name and the id it resolves to."""
    names = [((t.get("image") or {}).get("beaker")) for t in template.get("tasks") or []]
    names = [n for n in names if n]
    if not names:
        return None
    if len(set(names)) > 1:
        print(f"[launch_inputs] WARNING: tasks name several Beaker images {sorted(set(names))}; recording {names[0]}")
    image_id = None
    try:
        if client_factory is None:
            from beaker_client import get_beaker_client as client_factory
        image_id = getattr(client_factory(workspace).image.get(names[0]), "id", None)
    except Exception as exc:   # no token, no network, or an unknown name: record the name only
        print(f"[launch_inputs] WARNING: could not resolve Beaker image {names[0]!r} to its id ({exc})")
    return {"platform": "beaker", "name": names[0], "id": image_id}


def image_record_hpc(site_packages: list[Path]) -> dict:
    """The launch record's HPC `image`: the conda env prefix and a sha256 over its packages.

    The digest covers the conda package filenames (name-version-build) in ``conda-meta`` and
    the pip ``*.dist-info`` directory names, so it changes when any installed version does.
    """
    if not site_packages:
        return {"platform": "hpc", "conda_env": "unresolved", "lock_sha256": None}
    prefix = Path(site_packages[0]).parents[2]       # <prefix>/lib/pythonX.Y/site-packages
    names = sorted(p.name for p in (prefix / "conda-meta").glob("*.json"))
    names += sorted(p.name for sp in site_packages for p in Path(sp).glob("*.dist-info"))
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest() if names else None
    return {"platform": "hpc", "conda_env": str(prefix), "lock_sha256": digest}
