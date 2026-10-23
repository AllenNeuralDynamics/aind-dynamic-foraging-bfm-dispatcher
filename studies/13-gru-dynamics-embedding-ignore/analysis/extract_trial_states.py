"""A1: per-trial GRU hidden states for one 3-way source run (stage A — needs weights, data, JAX).

    python analysis/extract_trial_states.py C3-s0

Reads the model_dir written by ``fetch_models.py``, runs the wrapper's
``extract_gru_trial_states`` over the training mice's eval split, adds each session's
``task`` from the pinned snapshot's session table, and writes
  * ``analysis/_cache/trial_states_<key>.parquet`` — the full table (gitignored);
  * ``analysis/trial_states_<key>.json`` — counts, checks and provenance (committed).

The 3-way head steps through ignored trials, so the table keeps them as ``action == 2``
rows (``trial_idx`` is the GRU timestep; ``engaged_idx`` counts L/R trials only).

Fails unless the table reproduces all three eval likelihoods the training run logged for the
same checkpoint — 3-way over every trial, L/R renormalised over engaged trials, and
engage-vs-ignore over every trial (``gru_trainer`` decomposition) — and the readout
reconstructs every logit from its row's state. Needs the wrapper's ``code/`` on PYTHONPATH.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY.parent / "util"))
from _meta import build_meta  # noqa: E402

CACHE = STUDY / "analysis" / "_cache"
SNAPSHOT_ROOT = "aind-scratch-data/aind-dynamic-foraging-cache/snapshots"
SPLIT = "eval"
LIKELIHOOD_TOLERANCE = 1e-4
# The readout check guards row alignment, not precision (Study 12: GPU TF32 leaves ~1e-3,
# a one-trial misalignment gives ~0.2 median).
READOUT_TOLERANCE = 1e-2
IGNORE = 2


def _wrapper_checkout() -> dict:
    import post_training_analysis  # noqa: PLC0415

    repo = Path(post_training_analysis.__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    return {"path": str(repo), "head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _inputs_data_cfg(model_dir: Path) -> dict:
    import yaml  # noqa: PLC0415

    return yaml.safe_load((model_dir / "inputs.yaml").read_text())["data"]


def _logged_checkpoint(model_dir: Path, step: int) -> dict:
    index = json.loads((model_dir / "outputs" / "checkpoints" / "index.json").read_text())
    return next(c for c in index["checkpoints"] if int(c["step"]) == step)


def _decomposed_likelihoods(trial_df: "pd.DataFrame") -> dict[str, float]:
    """The ``gru_trainer`` 3-way decomposition, recomputed from the table."""
    action = trial_df["action"].to_numpy(dtype=int)
    p_left = trial_df["p_model1_left"].to_numpy(dtype=float)
    p_right = trial_df["p_model1_right"].to_numpy(dtype=float)
    engaged = action != IGNORE
    p_side = np.where(action == 1, p_right, p_left)[engaged] / (p_left + p_right)[engaged]
    p_engage = np.where(engaged, p_left + p_right, trial_df["p_model1_ignore"].to_numpy(dtype=float))
    return {
        "LR_engaged": float(np.exp(np.mean(np.log(p_side)))),
        "engage": float(np.exp(np.mean(np.log(p_engage)))),
    }


def _session_tasks(snapshot: str) -> "pd.DataFrame":
    import pandas as pd  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    import s3fs  # noqa: PLC0415

    fs = s3fs.S3FileSystem(anon=True)  # the snapshot bucket is publicly readable
    path = f"{SNAPSHOT_ROOT}/{snapshot}/session_table.parquet"
    table = pq.read_table(fs.open(path), columns=["_session_id", "task"]).to_pandas()
    if table["_session_id"].duplicated().any():
        raise AssertionError(f"{path} has duplicate _session_id values")
    return pd.DataFrame({"session_id": table["_session_id"].astype(str), "task": table["task"]})


def main(key: str) -> None:
    from post_training_analysis.gru_trial_states import extract_gru_trial_states  # noqa: PLC0415

    manifest = json.loads((STUDY / "source_runs.json").read_text())
    model_key, _ = key.split("-")
    source = manifest["models"][model_key]
    run = source["runs"][key]
    model_dir = CACHE / "models" / key / "run"
    data_cfg = _inputs_data_cfg(model_dir)
    if data_cfg["ignore_policy"] != source["ignore_policy"]:
        raise AssertionError(f"{key} model_dir is ignore_policy={data_cfg['ignore_policy']}")

    trial_df, info = extract_gru_trial_states(
        model_dir, split=SPLIT, checkpoint_policy=manifest["checkpoint_policy"]
    )
    if info["n_actions"] != source["output_size"]:
        raise AssertionError(f"{key} has {info['n_actions']} actions, expected {source['output_size']}")
    snapshot = str(data_cfg["snapshot"])
    n_rows = len(trial_df)
    trial_df = trial_df.merge(_session_tasks(snapshot), on="session_id", how="left", validate="m:1")
    if len(trial_df) != n_rows:
        raise AssertionError("Joining session tasks changed the row count")

    logged = _logged_checkpoint(model_dir, int(info["checkpoint_step"]))
    recomputed = {"all": info["pooled_trial_likelihood"], **_decomposed_likelihoods(trial_df)}
    logged_keys = {"all": "eval_likelihood", "LR_engaged": "eval_likelihood_LR_engaged", "engage": "eval_likelihood_engage"}
    checks = {
        "likelihoods": {
            name: {
                "logged": float(logged[logged_keys[name]]),
                "recomputed": value,
                "abs_diff": abs(value - float(logged[logged_keys[name]])),
            }
            for name, value in recomputed.items()
        },
        "readout_max_abs_error": info["readout_max_abs_error"],
    }

    out_parquet = CACHE / f"trial_states_{key}.parquet"
    trial_df.to_parquet(out_parquet, index=False)
    sessions = trial_df.drop_duplicates("ses_idx")
    summary = {
        "_meta": {
            **build_meta("analysis/extract_trial_states.py", [source["group"]], study_root=STUDY),
            "wrapper_checkout": _wrapper_checkout(),
        },
        "source": {"key": key, **run},
        "extraction": {k: v for k, v in info.items() if k not in ("model_dir", "params_path")},
        "checks": checks,
        "snapshot": snapshot,
        "trials_by_action": {
            name: int((trial_df["action"] == a).sum()) for a, name in enumerate(("left", "right", "ignore"))
        },
        "sessions_by_task": sessions["task"].fillna("<missing>").value_counts().to_dict(),
        "sessions_by_session_curriculum": sessions["session_curriculum_name"].value_counts().to_dict(),
        "subjects_by_subject_curriculum": trial_df.drop_duplicates("subject_id")["curriculum_name"].value_counts().to_dict(),
        "columns": list(trial_df.columns),
        "cache_file": str(out_parquet.relative_to(STUDY)),
    }
    (STUDY / "analysis" / f"trial_states_{key}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"checks": checks, **summary["extraction"]}, indent=2))

    failed = {k: v for k, v in checks["likelihoods"].items() if v["abs_diff"] > LIKELIHOOD_TOLERANCE}
    if failed:
        raise AssertionError(f"Recomputed likelihoods do not match the logged values: {failed}")
    if checks["readout_max_abs_error"] > READOUT_TOLERANCE:
        raise AssertionError(f"Readout does not reconstruct the logits: {checks}")


if __name__ == "__main__":
    main(sys.argv[1])
