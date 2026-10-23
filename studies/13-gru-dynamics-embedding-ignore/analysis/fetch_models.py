"""Download pinned source GRUs into runnable model_dirs under ``analysis/_cache/models/``.

    python analysis/fetch_models.py C3-s0 [C3-s1 ...]

Uses the wrapper's ``wandb_model_dir.hydrate_model_dir``, which resolves the artifact as
``:latest``. ``source_runs.json`` pins each run to an immutable artifact digest, so after the
download this asserts the ``:latest`` version IS the pinned digest and refuses otherwise.
Needs the wrapper importable (``code/`` on PYTHONPATH) and W&B credentials.
"""

from __future__ import annotations

import json
import netrc
import os
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
MANIFEST = STUDY / "source_runs.json"
CACHE = STUDY / "analysis" / "_cache" / "models"


def _ensure_wandb_key() -> None:
    if not os.environ.get("WANDB_API_KEY"):
        # Explicit path: netrc.netrc() with no argument rejects a ~/.netrc owned by another uid.
        credentials = netrc.netrc(os.path.expanduser("~/.netrc")).authenticators("api.wandb.ai")
        if not credentials or not credentials[2]:
            raise RuntimeError("WANDB_API_KEY is unset and api.wandb.ai is absent from ~/.netrc")
        os.environ["WANDB_API_KEY"] = credentials[2]


def _pinned_run(manifest: dict, key: str) -> dict:
    for model in manifest["models"].values():
        if key in model["runs"]:
            return model["runs"][key]
    raise KeyError(f"{key} is not in {MANIFEST.name}")


def main(keys: list[str]) -> None:
    _ensure_wandb_key()
    import wandb  # noqa: PLC0415
    from post_training_analysis.wandb_model_dir import hydrate_model_dir  # noqa: PLC0415

    manifest = json.loads(MANIFEST.read_text())
    for key in keys:
        run = _pinned_run(manifest, key)
        ref = f"{manifest['entity']}/{manifest['project']}/{run['artifact_name']}:latest"
        digest = wandb.Api().artifact(ref, type="training-output").digest
        if digest != run["artifact_digest"]:
            raise AssertionError(f"{ref} is digest {digest}, pinned {run['artifact_digest']}")
        model_dir = hydrate_model_dir(
            run["run_id"], project=manifest["project"], entity=manifest["entity"],
            dest=CACHE / key,
        )
        print(f"{key}: {model_dir} (digest {digest})")


if __name__ == "__main__":
    main(sys.argv[1:])
