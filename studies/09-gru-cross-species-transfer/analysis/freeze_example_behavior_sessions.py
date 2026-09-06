"""Freeze deterministic example sessions from the pinned public datasets."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import sys
from dataclasses import asdict
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402
from external_bandit_datasets.adapters import build_dataset  # noqa: E402
from external_bandit_datasets.sources import (  # noqa: E402
    SOURCES,
    file_digest,
)


MATCHED_DATA = STUDY / "analysis" / "matched_half_results.json"
AUTHOR_DATA = STUDY / "analysis" / "author_baseline_results.json"
OUTPUT = STUDY / "analysis" / "example_behavior_sessions.json"
DATASETS = ("grossman", "chen", "zid")
QUANTILES = (
    (0.1, "lower tail", "lower"),
    (0.5, "median", "median"),
    (0.9, "upper tail", "upper"),
)
BASIC_ANALYSIS_COMMIT = "590e5f085711a8ba99ca0d86e94471f347318daa"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=STUDY / "data-cache" / "raw",
        help="Directory containing checksum-verified source files by dataset name.",
    )
    return parser


def _author_record(author_data: dict, dataset_name: str) -> tuple[str, dict]:
    selected = [
        (name, record)
        for name, record in author_data["records"].items()
        if record["dataset"] == dataset_name and record["author_selected"]
    ]
    if len(selected) != 1:
        raise AssertionError(f"Expected one author-selected model for {dataset_name}")
    return selected[0]


def _subject_deltas(
    dataset_name: str, author_data: dict, matched_data: dict
) -> tuple[str, list[tuple[float, str]]]:
    baseline_name, reference_record = _author_record(author_data, dataset_name)
    dataset = matched_data["datasets"][dataset_name]
    gru_rows = [row for row in dataset["gru"] if int(row["nominal_D"]) == 614]
    if len(gru_rows) != 3:
        raise AssertionError(f"Expected three D=614 GRU seeds for {dataset_name}")
    reference = reference_record["metrics"]["per_subject_mean_log_likelihood_nats"]
    gru = [row["metrics"]["per_subject_mean_log_likelihood_nats"] for row in gru_rows]
    if any(set(seed) != set(reference) for seed in gru):
        raise AssertionError(f"Subject keys do not align for {dataset_name}")
    deltas = []
    for subject_id in reference:
        gru_log_likelihood = statistics.mean(float(seed[subject_id]) for seed in gru)
        delta = math.exp(gru_log_likelihood) - math.exp(float(reference[subject_id]))
        deltas.append((delta, subject_id))
    return baseline_name, sorted(deltas, key=lambda item: (item[0], item[1]))


def _select_examples(sorted_deltas: list[tuple[float, str]]) -> list[dict]:
    examples = []
    for quantile, label, slug in QUANTILES:
        center_rank = round(quantile * (len(sorted_deltas) - 1))
        ranks = range(center_rank - 1, center_rank + 2)
        for category_index, rank in enumerate(ranks, start=1):
            delta, subject_id = sorted_deltas[rank]
            examples.append(
                {
                    "selection_label": label,
                    "selection_slug": slug,
                    "category_index": category_index,
                    "target_quantile": quantile,
                    "rank_zero_based": rank,
                    "subject_id": subject_id,
                    "gru_d614_minus_author_normalized_likelihood": delta,
                }
            )
    if len({example["subject_id"] for example in examples}) != len(examples):
        raise AssertionError("Example subject selection contains duplicates")
    return examples


def _manifest_subject(manifest: dict, subject_id: str) -> dict:
    matches = [row for row in manifest["subjects"] if row["subject_id"] == subject_id]
    if len(matches) != 1:
        raise AssertionError(f"Expected one split record for {subject_id}")
    return matches[0]


def _session_record(df, manifest: dict, example: dict, dataset_name: str) -> dict:
    split = _manifest_subject(manifest, example["subject_id"])
    if dataset_name == "zid":
        session_id = split["session_id"]
        partition = "full session; adaptation prefix then held-out suffix"
        adapt_prefix_trials = int(split["adapt_prefix_trials"])
    else:
        session_id = split["test_session_ids"][0]
        partition = "first held-out session"
        adapt_prefix_trials = None
    rows = df[
        (df["subject_id"] == example["subject_id"]) & (df["ses_idx"] == session_id)
    ].sort_values("trial")
    if rows.empty:
        raise AssertionError(f"No rows for {example['subject_id']} / {session_id}")
    required = ["reward_probability_arm_0", "reward_probability_arm_1"]
    if rows[required].isna().any().any():
        raise AssertionError(f"Reward probabilities are incomplete for {dataset_name}")
    return {
        **example,
        "session_id": str(session_id),
        "partition": partition,
        "adapt_prefix_trials": adapt_prefix_trials,
        "n_trials": int(len(rows)),
        "trial": [int(value) for value in rows["trial"]],
        "choice": [int(value) for value in rows["animal_response"]],
        "reward": [int(value) for value in rows["earned_reward"]],
        "reward_probability_arm_0": [
            float(value) for value in rows["reward_probability_arm_0"]
        ],
        "reward_probability_arm_1": [
            float(value) for value in rows["reward_probability_arm_1"]
        ],
    }


def _plot_source() -> dict[str, str]:
    module = importlib.import_module(
        "aind_dynamic_foraging_basic_analysis.plot.plot_foraging_session"
    )
    source_path = Path(module.__file__)
    return {
        "function": "plot_foraging_session",
        "repository": "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis",
        "commit": BASIC_ANALYSIS_COMMIT,
        "source_url": (
            "https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis/"
            f"blob/{BASIC_ANALYSIS_COMMIT}/src/aind_dynamic_foraging_basic_analysis/plot/"
            "plot_foraging_session.py"
        ),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = _parser().parse_args()
    matched_data = json.loads(MATCHED_DATA.read_text())
    author_data = json.loads(AUTHOR_DATA.read_text())
    groups = sorted(
        set(matched_data["_meta"]["wandb_groups"])
        | set(author_data["_meta"]["wandb_groups"])
    )
    datasets = {}
    for dataset_name in DATASETS:
        source = SOURCES[dataset_name]
        source_path = args.raw_root / dataset_name / source.filename
        df, manifest, audit = build_dataset(dataset_name, source_path)
        baseline_name, deltas = _subject_deltas(dataset_name, author_data, matched_data)
        examples = [
            _session_record(df, manifest, example, dataset_name)
            for example in _select_examples(deltas)
        ]
        datasets[dataset_name] = {
            "author_reference": baseline_name,
            "source": asdict(source),
            "verified_source_digest": file_digest(source_path, source.digest_algorithm),
            "audit": audit,
            "examples": examples,
        }
    output = {
        "_meta": build_meta(
            "studies/09-gru-cross-species-transfer/analysis/"
            "freeze_example_behavior_sessions.py",
            groups,
            study_root=STUDY,
        ),
        "selection": {
            "criterion": (
                "three neighboring ranks centered on the 10th, 50th, and 90th percentiles "
                "of subject-level D=614 GRU minus author-selected normalized likelihood"
            ),
            "gru_source_seeds": 3,
            "multi_session_example": "first held-out session",
            "zid_example": "full session with adaptation/test boundary at trial 150",
        },
        "plotting": _plot_source(),
        "datasets": datasets,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
