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
OUTPUT = STUDY / "analysis" / "example_behavior_sessions.json"
DATASETS = (
    "grossman",
    "chen",
    "zid",
    "lebedeva",
    "beron",
    "kwak",
    "miller",
    "findling",
    "tang",
    "alsio",
    "eckstein",
    "costa",
    "lopez_mouse",
)
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


def _session_deltas(
    dataset_name: str, matched_data: dict
) -> list[tuple[float, str, str]]:
    dataset = matched_data["datasets"][dataset_name]
    gru_rows = [row for row in dataset["gru"] if int(row["nominal_D"]) == 614]
    if len(gru_rows) != 3:
        raise AssertionError(f"Expected three D=614 GRU seeds for {dataset_name}")
    q = {
        (row["subject_id"], row["ses_idx"]): row["mean_log_likelihood_nats"]
        for row in dataset["q"]["metrics"]["per_session"]
    }
    gru = [
        {
            (row["subject_id"], row["ses_idx"]): row["mean_log_likelihood_nats"]
            for row in seed["metrics"]["per_session"]
        }
        for seed in gru_rows
    ]
    if any(set(seed) != set(q) for seed in gru):
        raise AssertionError(f"Session keys do not align for {dataset_name}")
    deltas = []
    for subject_id, ses_idx in q:
        gru_log_likelihood = statistics.mean(
            float(seed[(subject_id, ses_idx)]) for seed in gru
        )
        delta = math.exp(gru_log_likelihood) - math.exp(
            float(q[(subject_id, ses_idx)])
        )
        deltas.append((delta, subject_id, ses_idx))
    return sorted(deltas, key=lambda item: (item[0], item[1], item[2]))


def _select_examples(sorted_deltas: list[tuple[float, str, str]]) -> list[dict]:
    examples = []
    if len(sorted_deltas) < 9:
        if len(sorted_deltas) != 4:
            raise AssertionError(
                f"Need at least nine held-out sessions for ranked examples, got {len(sorted_deltas)}"
            )
        selections = (
            ("lower tail", "lower", [0]),
            ("median", "median", [1, 2]),
            ("upper tail", "upper", [3]),
        )
        for label, slug, ranks in selections:
            for category_index, rank in enumerate(ranks, start=1):
                delta, subject_id, session_id = sorted_deltas[rank]
                examples.append(
                    {
                        "selection_label": label,
                        "selection_slug": slug,
                        "category_index": category_index,
                        "target_quantile": None,
                        "rank_zero_based": rank,
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "gru_d614_minus_q_normalized_likelihood": delta,
                    }
                )
        return examples
    for quantile, label, slug in QUANTILES:
        center_rank = round(quantile * (len(sorted_deltas) - 1))
        ranks = range(center_rank - 1, center_rank + 2)
        for category_index, rank in enumerate(ranks, start=1):
            delta, subject_id, session_id = sorted_deltas[rank]
            examples.append(
                {
                    "selection_label": label,
                    "selection_slug": slug,
                    "category_index": category_index,
                    "target_quantile": quantile,
                    "rank_zero_based": rank,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "gru_d614_minus_q_normalized_likelihood": delta,
                }
            )
    keys = {(example["subject_id"], example["session_id"]) for example in examples}
    if len(keys) != len(examples):
        raise AssertionError("Example session selection contains duplicates")
    return examples


def _manifest_subject(manifest: dict, subject_id: str) -> dict:
    matches = [row for row in manifest["subjects"] if row["subject_id"] == subject_id]
    if len(matches) != 1:
        raise AssertionError(f"Expected one split record for {subject_id}")
    return matches[0]


def _session_record(df, manifest: dict, example: dict, dataset_name: str) -> dict:
    split = _manifest_subject(manifest, example["subject_id"])
    session_id = example["session_id"]
    if "adapt_prefix_trials" in split:
        if str(split["session_id"]) != str(session_id):
            raise AssertionError(f"Manifest session mismatch for {example['subject_id']}")
        partition = "full session; adaptation prefix then held-out suffix"
        adapt_prefix_trials = int(split["adapt_prefix_trials"])
    else:
        if str(session_id) not in {str(value) for value in split["test_session_ids"]}:
            raise AssertionError(f"Selected session is not held out for {example['subject_id']}")
        partition = "held-out session"
        adapt_prefix_trials = None
    rows = df[
        (df["subject_id"] == example["subject_id"]) & (df["ses_idx"] == session_id)
    ].sort_values("trial")
    if rows.empty:
        raise AssertionError(f"No rows for {example['subject_id']} / {session_id}")
    probability_columns = ["reward_probability_arm_0", "reward_probability_arm_1"]
    probabilities_available = all(column in rows for column in probability_columns)
    if probabilities_available:
        probabilities_available = not rows[probability_columns].isna().any().any()
    if probabilities_available:
        probability_0 = [float(value) for value in rows[probability_columns[0]]]
        probability_1 = [float(value) for value in rows[probability_columns[1]]]
    else:
        probability_0 = [None] * len(rows)
        probability_1 = [None] * len(rows)
    return {
        **example,
        "session_id": str(session_id),
        "partition": partition,
        "adapt_prefix_trials": adapt_prefix_trials,
        "n_trials": int(len(rows)),
        "trial": [int(value) for value in rows["trial"]],
        "choice": [int(value) for value in rows["animal_response"]],
        "reward": [int(value) for value in rows["earned_reward"]],
        "reward_probability_available": probabilities_available,
        "reward_probability_arm_0": probability_0,
        "reward_probability_arm_1": probability_1,
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
    groups = sorted(set(matched_data["_meta"]["wandb_groups"]))
    datasets = {}
    for dataset_name in DATASETS:
        source = SOURCES[dataset_name]
        source_path = args.raw_root / dataset_name / source.filename
        df, manifest, audit = build_dataset(dataset_name, source_path)
        deltas = _session_deltas(dataset_name, matched_data)
        examples = [
            _session_record(df, manifest, example, dataset_name)
            for example in _select_examples(deltas)
        ]
        datasets[dataset_name] = {
            "reference": "common Q",
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
                "of held-out-session D=614 GRU minus common-Q normalized likelihood"
            ),
            "gru_source_seeds": 3,
            "multi_session_example": "ranked held-out real session",
            "within_session_example": "full session with the adaptation/test boundary shown",
            "small_cohort_exception": (
                "Tang has only four held-out real sessions; all four are shown once "
                "instead of duplicating sessions to manufacture three per category"
            ),
        },
        "plotting": _plot_source(),
        "datasets": datasets,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
