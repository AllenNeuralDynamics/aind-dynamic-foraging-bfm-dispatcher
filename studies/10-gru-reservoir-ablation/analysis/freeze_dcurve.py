#!/usr/bin/env python
"""Freeze the H128/E4 reservoir scaling curve from its pinned W&B runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import wandb

from freeze_results import (
    RESERVOIR_PROJECT,
    _config_value,
    _frozen_parameter_audit,
    _summary_likelihood,
    _training_artifact,
    _training_steps_completed,
)


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402


OUTPUT = STUDY / "analysis" / "reservoir_curve_results.json"
DCURVE_GROUP = "frozen-random-core-h128-dcurve@20260908-111444"
PILOT_GROUPS = {
    0: "frozen-random-core-d614@20260907-175533",
    1: "frozen-random-core-d614@20260907-184115",
    2: "frozen-random-core-d614@20260907-184115",
}
CELLS = (
    (10, 0, 0.016, "frozen-random-core-h128-dcurve-20260908-111444-4f5a3b97"),
    (10, 1, 0.016, "frozen-random-core-h128-dcurve-20260908-111444-323b10e3"),
    (10, 2, 0.016, "frozen-random-core-h128-dcurve-20260908-111444-e3c22e2a"),
    (30, 0, 0.049, "frozen-random-core-h128-dcurve-20260908-111444-0ad45196"),
    (30, 1, 0.049, "frozen-random-core-h128-dcurve-20260908-111444-ac135db5"),
    (30, 2, 0.049, "frozen-random-core-h128-dcurve-20260908-111444-cf78e317"),
    (100, 0, 0.163, "frozen-random-core-h128-dcurve-20260908-111444-bd56f95c"),
    (100, 1, 0.163, "frozen-random-core-h128-dcurve-20260908-111444-5e07ffdb"),
    (100, 2, 0.163, "frozen-random-core-h128-dcurve-20260908-111444-17ff8d61"),
    (300, 0, 0.489, "frozen-random-core-h128-dcurve-20260908-111444-104f19be"),
    (300, 1, 0.489, "frozen-random-core-h128-dcurve-20260908-111444-983e7dea"),
    (300, 2, 0.489, "frozen-random-core-h128-dcurve-20260908-111444-533646d0"),
    (614, 0, 1.0, "frozen-random-core-d614-20260907-175533-bbcdf2f9"),
    (614, 1, 1.0, "frozen-random-core-d614-20260907-184115-0c259519"),
    (614, 2, 1.0, "frozen-random-core-d614-20260907-184115-d2f43380"),
)


def _validate(run: Any, *, nominal_d: int, seed: int, ratio: float) -> int:
    expected = {
        ("model", "architecture", "hidden_size"): 128,
        ("model", "architecture", "subject_embedding_size"): 4,
        ("model", "training", "freeze_gru_core"): True,
        ("model", "training", "auto_heldout_finetune", "n_steps"): 500,
        ("model", "training", "auto_heldout_finetune", "lr"): 0.001,
        ("data", "snapshot"): "20260603",
    }
    for path, value in expected.items():
        actual = _config_value(run, *path)
        if path == ("data", "snapshot"):
            actual = str(actual)
        if actual != value:
            raise ValueError(f"run {run.id} has {'.'.join(path)}={actual!r}")
    data = run.config["data"]
    if int(data["seed"]) != seed or float(data["subject_ratio"]) != ratio:
        raise ValueError(f"run {run.id} has the wrong ratio or seed")
    actual_d = len(run.config.get("resolved_subject_ids") or [])
    if nominal_d == 614 and actual_d != 614:
        raise ValueError(f"run {run.id} has D={actual_d}, expected 614")
    return actual_d


def main() -> None:
    api = wandb.Api(timeout=90)
    results = []
    for nominal_d, seed, ratio, run_id in CELLS:
        run = api.run(f"{RESERVOIR_PROJECT}/{run_id}")
        expected_group = DCURVE_GROUP if nominal_d < 614 else PILOT_GROUPS[seed]
        if run.state != "finished" or run.group != expected_group:
            raise ValueError(f"run {run.id} is not the pinned finished cell")
        actual_d = _validate(run, nominal_d=nominal_d, seed=seed, ratio=ratio)
        audit = _frozen_parameter_audit(run)
        artifact = _training_artifact(run)
        results.append(
            {
                "nominal_D": nominal_d,
                "D": actual_d,
                "seed": seed,
                "subject_ratio": ratio,
                "cohort": "pilot" if nominal_d == 614 else "dcurve",
                "wandb_run_id": run.id,
                "wandb_url": run.url,
                "heldout_likelihood": _summary_likelihood(run),
                "training_steps_completed": _training_steps_completed(run),
                "training_artifact": {"name": artifact.name, "digest": artifact.digest},
                "frozen_parameter_audit": audit,
            }
        )
    OUTPUT.write_text(
        json.dumps(
            {
                "_meta": build_meta(
                    "analysis/freeze_dcurve.py",
                    [DCURVE_GROUP, *sorted(set(PILOT_GROUPS.values()))],
                    study_root=STUDY,
                ),
                "cells": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
