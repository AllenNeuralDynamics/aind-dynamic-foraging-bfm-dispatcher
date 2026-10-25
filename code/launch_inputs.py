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


def _overrides(sweep: dict) -> dict[str, list[str]]:
    """key -> values the sweep sets, from its command list (key=value) and parameters."""
    out: dict[str, list[str]] = {}
    for item in sweep.get("command") or []:
        if isinstance(item, str) and "=" in item and not item.startswith("$"):
            key, value = item.lstrip("+~").split("=", 1)
            out.setdefault(key, []).append(value)
    for key, spec in (sweep.get("parameters") or {}).items():
        if isinstance(spec, dict) and "value" in spec:
            out.setdefault(key, []).append(str(spec["value"]))
        elif isinstance(spec, dict) and "values" in spec:
            out.setdefault(key, []).extend(str(v) for v in spec["values"])
    return out


def _default_data_config(config_root: Path) -> str | None:
    """The `data` entry of the root config's Hydra defaults list."""
    root = yaml.safe_load((config_root / "config.yaml").read_text()) or {}
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


def data_record(sweep: dict, config_root: Path | str,
                fetch: Callable[[str], tuple[str | None, dict]] = fetch_manifest) -> dict:
    """The launch record's `data`: snapshot (pinned), live (database, unpinned) or generated."""
    config_root = Path(config_root)
    ov = _overrides(sweep)
    name = (ov.get("data") or [None])[-1] or _default_data_config(config_root)
    path = config_root / "data" / f"{name}.yaml"
    cfg = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    dtype = cfg.get("type")
    swept = set(ov.get("data.snapshot") or [])
    if len(swept) > 1:
        raise DataConfigError(f"the sweep varies data.snapshot over {sorted(swept)}; launch one snapshot per launch")
    snapshot = next(iter(swept)) if swept else cfg.get("snapshot")
    snapshot = None if snapshot in (None, "", "null", "None") else str(snapshot)
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
