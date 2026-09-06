"""Fit an author-aligned behavioral baseline on its matched target split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


BASELINES = {
    "grossman-meta-learning": {
        "dataset": "grossman",
        "dataset_id": "grossman-bari-cohen-2021",
        "agent_class": "ForagerGrossmanMetaLearning",
        "citation": "Grossman et al., Current Biology (2022)",
    },
    "chen-rlck": {
        "dataset": "chen",
        "dataset_id": "chen-et-al-2021",
        "agent_class": "ForagerRLCK",
        "citation": "Chen et al., eLife (2021)",
    },
    "zid-traditional-rlck": {
        "dataset": "zid",
        "dataset_id": "zid-et-al-2026-experiment-1",
        "agent_class": "ForagerRLCK",
        "citation": "Zid et al., Nature Communications (2026), Eq. 19",
    },
    "zid-history-kernel-foraging": {
        "dataset": "zid",
        "dataset_id": "zid-et-al-2026-experiment-1",
        "agent_class": "ForagerZidHistoryKernel",
        "citation": "Zid et al., Nature Communications (2026), Eq. 22",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject-workers", type=int, default=12)
    return parser


def main() -> None:
    args = _parser().parse_args()
    specification = BASELINES[args.baseline]
    dataset = specification["dataset"]
    wrapper_code = Path(os.environ["BFM_WRAPPER_CODE"])
    sys.path.insert(0, str(wrapper_code))

    import wandb
    from data_loaders.external_bandit import ExternalBanditDatasetLoader
    from model_trainers.baseline_rl_trainer import BaselineRLTrainer

    output_dir = args.output_root / args.baseline
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.data_root / f"{dataset}.audit.json").read_text())
    config = {
        "model": {
            "agent_class": specification["agent_class"],
            "optimizer": "differential_evolution",
            "polish": True,
            "citation": specification["citation"],
        },
        "target": {
            "dataset": dataset,
            "audit": audit,
            "condition": "matched_half",
        },
        "seed": 0,
        "meta": {
            "study": "09-gru-cross-species-transfer",
            "variant": os.environ.get("BFM_META_VARIANT"),
            "dispatcher_commit": os.environ.get("DISPATCHER_COMMIT"),
            "wrapper_commit": os.environ.get("WRAPPER_COMMIT"),
            "foraging_models_commit": os.environ.get("FORAGING_MODELS_COMMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    run = wandb.init(
        entity="AIND-disRNN",
        project="gru_cross_species_transfer",
        group=os.environ.get("WANDB_RUN_GROUP"),
        name=args.baseline,
        dir=str(output_dir),
        config=config,
        tags=["external-transfer", "matched-half", "author-baseline"],
    )
    loader = ExternalBanditDatasetLoader(
        file_path=args.data_root / f"{dataset}.parquet",
        split_manifest_path=args.data_root / f"{dataset}.split.json",
        dataset_id=specification["dataset_id"],
        batch_size=None,
        batch_mode="single",
        adapt_sessions_per_subject=None,
        seed=0,
        train_example_sessions_per_subject=0,
        eval_example_sessions_per_subject=0,
        heldout_example_sessions_per_subject=0,
    )
    trainer = BaselineRLTrainer(
        agent_class=specification["agent_class"],
        architecture={"multisubject": True},
        agent_kwargs={},
        fit_bounds_override={},
        clamp_params={},
        DE_kwargs={"polish": True},
        multisubject_subject_workers=args.subject_workers,
        output_dir=str(output_dir / "outputs"),
        seed=0,
    )
    try:
        result = trainer.fit(loader.load(), loggers={"wandb": run})
        for key, value in result["test_metrics"].items():
            if isinstance(value, (int, float)):
                run.summary[f"target/test/{key}"] = value
        print(json.dumps(result["test_metrics"], indent=2))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
