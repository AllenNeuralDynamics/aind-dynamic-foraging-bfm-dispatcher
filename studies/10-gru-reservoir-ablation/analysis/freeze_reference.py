#!/usr/bin/env python
"""Attach original run/artifact identities to the frozen Study 01 reference.

This metadata extraction asserts that every live scalar is exactly the already
published value. It never replaces the historical likelihoods with later reruns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import wandb

from freeze_results import _config_value, _summary_likelihood, _training_artifact

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402

REFERENCE = STUDY / "reference" / "study01-trained-gru.json"
PROJECT = "AIND-disRNN/mice_data_scaling"
GROUP = "v2-sc-active@20260622-144622"
RUN_SUFFIXES = {
    10: ("4bb879a6", "f8f4f377", "c4f91751"),
    30: ("e8c3edb2", "5e008e40", "19a88c23"),
    100: ("4becb8b9", "e7f3851f", "35610e21"),
    300: ("212ad763", "c4ff4e08", "8c0d52a5"),
    614: ("ecd6f7e6", "3603897b", "d6462060"),
}


def main() -> None:
    reference = json.loads(REFERENCE.read_text())
    api = wandb.Api(timeout=30)
    for cell in reference["cells"]:
        d = int(cell.get("nominal_D", cell["D"]))
        seed = int(cell["seed"])
        run_id = "v2-sc-active-20260622-144622-" + RUN_SUFFIXES[d][seed]
        run = api.run(f"{PROJECT}/{run_id}")
        if run.state != "finished" or run.group != GROUP:
            raise ValueError(f"{run_id}: not the pinned finished source run")
        if _summary_likelihood(run) != cell["likelihood"]:
            raise ValueError(f"{run_id}: scalar differs from the frozen reference")
        if len(run.config["resolved_subject_ids"]) != cell["D"]:
            raise ValueError(f"{run_id}: realized D differs from the frozen reference")
        expected = {
            ("data", "seed"): seed,
            ("model", "architecture", "hidden_size"): 128,
            ("model", "architecture", "subject_embedding_size"): 4,
        }
        for path, value in expected.items():
            if _config_value(run, *path) != value:
                raise ValueError(f"{run_id}: wrong {'.'.join(path)}")
        artifact = _training_artifact(run)
        if cell.get("artifact_digest") and cell["artifact_digest"] != artifact.digest:
            raise ValueError(f"{run_id}: immutable artifact digest changed")
        cell.update(
            wandb_run_id=run_id,
            wandb_url=run.url,
            artifact_name=artifact.name,
            artifact_digest=artifact.digest,
        )
        print(f"verified D={d}, seed={seed}", flush=True)
    reference = {
        "_meta": build_meta("analysis/freeze_reference.py", [GROUP], study_root=STUDY),
        **{key: value for key, value in reference.items() if key != "_meta"},
    }
    REFERENCE.write_text(json.dumps(reference, indent=2) + "\n")


if __name__ == "__main__":
    main()
