"""Fit Tang's block-type-specific RL model on the matched session split."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np


WHAT_BLOCK = 1
WHERE_BLOCK = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _domain_choice(rows) -> np.ndarray:
    """Return the option identity learned in one Tang block."""
    block_types = rows["block_type"].astype(int).unique().tolist()
    if len(block_types) != 1:
        raise ValueError(f"Tang block contains multiple block types: {block_types}")
    if block_types[0] == WHAT_BLOCK:
        return rows["trial_object"].to_numpy(dtype=int)
    if block_types[0] == WHERE_BLOCK:
        return rows["animal_response"].to_numpy(dtype=int)
    raise ValueError(f"Unexpected Tang block type {block_types[0]!r}")


def _probability_right(rows, probability_domain_one: np.ndarray) -> np.ndarray:
    """Map object/action probability back to canonical right-choice probability."""
    probability_domain_one = np.asarray(probability_domain_one, dtype=float)
    if len(probability_domain_one) != len(rows):
        raise ValueError("Tang probability and block lengths disagree")
    block_type = int(rows["block_type"].iloc[0])
    if block_type == WHERE_BLOCK:
        return probability_domain_one
    if block_type != WHAT_BLOCK:
        raise ValueError(f"Unexpected Tang block type {block_type!r}")

    chosen_action = rows["animal_response"].to_numpy(dtype=int)
    chosen_object = rows["trial_object"].to_numpy(dtype=int)
    if not set(np.unique(chosen_action)).issubset({0, 1}):
        raise ValueError("Tang actions must be binary")
    if not set(np.unique(chosen_object)).issubset({0, 1}):
        raise ValueError("Tang object identities must be binary")
    object_on_right = np.where(chosen_action == 1, chosen_object, 1 - chosen_object)
    return np.where(
        object_on_right == 1,
        probability_domain_one,
        1.0 - probability_domain_one,
    )


def _fit_domain(blocks: list, seed: int) -> tuple[dict[str, float], dict[str, float]]:
    from aind_dynamic_foraging_models.generative_model import (
        ForagerFeedbackDependentRL,
    )

    choices = [_domain_choice(block) for block in blocks]
    rewards = [block["earned_reward"].to_numpy(dtype=float) for block in blocks]
    agent = ForagerFeedbackDependentRL(seed=seed)
    fitting_result, _ = agent.fit(
        fit_choice_history=choices,
        fit_reward_history=rewards,
        fit_bounds_override={},
        clamp_params={},
        DE_kwargs={"polish": True, "workers": 1},
    )
    params = {
        name: float(value) for name, value in fitting_result.params.items()
    }
    fit = {
        "log_likelihood": float(fitting_result.log_likelihood),
        "normalized_likelihood": float(fitting_result.LPT),
        "AIC": float(fitting_result.AIC),
        "BIC": float(fitting_result.BIC),
        "n_trials": int(sum(map(len, choices))),
    }
    return params, fit


def _predict_block(rows, params: dict[str, float], seed: int) -> np.ndarray:
    from aind_dynamic_foraging_models.generative_model import (
        ForagerFeedbackDependentRL,
    )

    choices = _domain_choice(rows)
    rewards = rows["earned_reward"].to_numpy(dtype=float)
    agent = ForagerFeedbackDependentRL(seed=seed)
    agent.set_params(**params)
    agent.perform_closed_loop(choices, rewards)
    probability_domain_one = np.asarray(agent.choice_prob[1], dtype=float)
    return _probability_right(rows, probability_domain_one)


def _fit_subject(payload: dict) -> dict:
    subject_rows = payload["rows"]
    train_session_ids = set(payload["train_session_ids"])
    eval_session_ids = set(payload["eval_session_ids"])
    block_columns = ["ses_idx", "block_index"]
    train_blocks = {WHAT_BLOCK: [], WHERE_BLOCK: []}
    eval_blocks = []
    for (session_id, _), block in subject_rows.groupby(block_columns, sort=False):
        block = block.sort_values("trial")
        block_type = int(block["block_type"].iloc[0])
        if block_type not in train_blocks:
            raise ValueError(f"Unexpected Tang block type {block_type!r}")
        if str(session_id) in train_session_ids:
            train_blocks[block_type].append(block)
        elif str(session_id) in eval_session_ids:
            eval_blocks.append(block)
        else:
            raise AssertionError(f"Tang session {session_id!r} is outside the split")

    if any(not blocks for blocks in train_blocks.values()) or not eval_blocks:
        raise AssertionError(
            f"Tang subject {payload['subject_id']!r} lacks a block type or split"
        )
    domain_seeds = np.random.SeedSequence(payload["seed"]).spawn(2)
    params = {}
    fit = {}
    for block_type, label, seed_sequence in zip(
        (WHAT_BLOCK, WHERE_BLOCK),
        ("what_object", "where_action"),
        domain_seeds,
    ):
        params[label], fit[label] = _fit_domain(
            train_blocks[block_type],
            int(seed_sequence.generate_state(1)[0]),
        )

    predictions = []
    eval_seeds = np.random.SeedSequence([payload["seed"], 1]).spawn(
        len(eval_blocks)
    )
    for block, seed_sequence in zip(eval_blocks, eval_seeds):
        label = (
            "what_object"
            if int(block["block_type"].iloc[0]) == WHAT_BLOCK
            else "where_action"
        )
        probability_right = _predict_block(
            block,
            params[label],
            int(seed_sequence.generate_state(1)[0]),
        )
        predictions.extend(
            (int(row_index), float(probability))
            for row_index, probability in zip(block.index, probability_right)
        )
    return {
        "subject_id": payload["subject_id"],
        "params": params,
        "fit": fit,
        "n_eval_trials": len(predictions),
        "predictions": predictions,
    }


def _subject_payloads(bundle, seed: int) -> list[dict]:
    metadata = bundle.metadata
    if metadata.get("split_strategy") != "explicit_manifest":
        raise ValueError("Tang (macaque) requires its audited schema-v1 split")
    train_session_ids = {str(value) for value in metadata["train_session_ids"]}
    eval_session_ids = {str(value) for value in metadata["eval_session_ids"]}
    subject_seeds = np.random.SeedSequence(seed).spawn(
        int(metadata["num_subjects"])
    )
    payloads = []
    for subject_index, (subject_id, rows) in enumerate(
        bundle.raw.groupby("subject_id", sort=False)
    ):
        payloads.append(
            {
                "subject_id": (
                    subject_id.item()
                    if isinstance(subject_id, np.generic)
                    else subject_id
                ),
                "rows": rows,
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
                "seed": int(subject_seeds[subject_index].generate_state(1)[0]),
            }
        )
    return payloads


def main() -> None:
    args = _parser().parse_args()
    wrapper_code = Path(os.environ["BFM_WRAPPER_CODE"])
    sys.path.insert(0, str(wrapper_code))

    import wandb
    from data_loaders.external_bandit import ExternalBanditDatasetLoader
    from evaluation.target_transfer import (
        build_binary_trial_predictions,
        summarize_binary_trial_predictions,
    )

    output_dir = args.output_root / "tang-block-type-rl"
    outputs_dir = output_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.data_root / "tang.audit.json").read_text())
    config = {
        "model": {
            "agent_class": "TangBlockTypeFeedbackDependentRL",
            "base_agent_class": "ForagerFeedbackDependentRL",
            "optimizer": "differential_evolution",
            "what_block_type": WHAT_BLOCK,
            "where_block_type": WHERE_BLOCK,
            "citation": "Tang et al., Nature Communications (2021), Eqs. 1-3",
        },
        "target": {
            "dataset": "tang",
            "audit": audit,
            "condition": "matched_half",
        },
        "seed": args.seed,
        "meta": {
            "study": "09-gru-cross-species-transfer",
            "variant": os.environ.get("BFM_META_VARIANT"),
            "dispatcher_commit": os.environ.get("DISPATCHER_COMMIT"),
            "wrapper_commit": os.environ.get("WRAPPER_COMMIT"),
            "foraging_models_commit": os.environ.get("FORAGING_MODELS_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    run = wandb.init(
        entity="AIND-disRNN",
        project="gru_cross_species_transfer",
        group=os.environ.get("WANDB_RUN_GROUP"),
        name="tang-block-type-rl",
        dir=str(output_dir),
        config=config,
        tags=["external-transfer", "matched-half", "author-baseline"],
    )
    loader = ExternalBanditDatasetLoader(
        file_path=args.data_root / "tang.parquet",
        split_manifest_path=args.data_root / "tang.split.json",
        dataset_id="tang-bartolo-averbeck-2021",
        batch_size=None,
        batch_mode="single",
        adapt_sessions_per_subject=None,
        seed=args.seed,
        train_example_sessions_per_subject=0,
        eval_example_sessions_per_subject=0,
        heldout_example_sessions_per_subject=0,
    )
    try:
        bundle = loader.load()
        payloads = _subject_payloads(bundle, args.seed)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.subject_workers, len(payloads))
        ) as executor:
            subject_results = list(executor.map(_fit_subject, payloads))

        probability_right = np.full(len(bundle.raw), np.nan, dtype=float)
        for result in subject_results:
            for row_index, probability in result.pop("predictions"):
                probability_right[row_index] = probability
        predictions = build_binary_trial_predictions(
            bundle.raw,
            bundle.metadata,
            probability_choice_1=probability_right,
            model="tang_block_type_feedback_dependent_rl",
        )
        metrics = summarize_binary_trial_predictions(predictions)
        predictions_path = outputs_dir / "test_trial_predictions.csv"
        metrics_path = outputs_dir / "test_metrics.json"
        summaries_path = outputs_dir / "subject_fit_summaries.json"
        predictions.to_csv(predictions_path, index=False)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        summaries_path.write_text(
            json.dumps(
                {"num_subjects": len(subject_results), "subjects": subject_results},
                indent=2,
            )
            + "\n"
        )
        result = {
            "multisubject": True,
            "fit_strategy": "per_subject_separate_what_where",
            "agent_class": "TangBlockTypeFeedbackDependentRL",
            "num_subjects": len(subject_results),
            "test_metrics": metrics,
            "subject_artifacts": {
                "subject_fit_summaries_json": str(summaries_path),
                "test_trial_predictions_csv": str(predictions_path),
                "test_metrics_json": str(metrics_path),
            },
        }
        (outputs_dir / "baseline_rl_results.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                run.summary[f"target/test/{key}"] = value
        artifact = wandb.Artifact(
            f"baseline-rl-output-{run.id}", type="training-output"
        )
        artifact.add_dir(str(outputs_dir))
        run.log_artifact(artifact)
        print(json.dumps(metrics, indent=2))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
