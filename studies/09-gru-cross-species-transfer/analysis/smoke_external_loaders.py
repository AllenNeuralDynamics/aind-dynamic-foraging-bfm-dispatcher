"""Load one real subject from every admitted dataset through the wrapper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STUDY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from external_bandit_datasets.sources import SOURCES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-code", type=Path, required=True)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=STUDY_DIR / "data-cache" / "canonical",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.wrapper_code))

    from data_loaders.external_bandit import ExternalBanditDatasetLoader

    records = []
    for name, source in SOURCES.items():
        manifest_path = args.canonical_root / f"{name}.split.json"
        manifest = json.loads(manifest_path.read_text())
        subject_id = manifest["subjects"][0]["subject_id"]
        bundle = ExternalBanditDatasetLoader(
            file_path=args.canonical_root / f"{name}.parquet",
            split_manifest_path=manifest_path,
            dataset_id=source.dataset_id,
            subject_ids=[subject_id],
            batch_size=None,
            batch_mode="single",
            adapt_sessions_per_subject=None,
            seed=0,
            train_example_sessions_per_subject=0,
            eval_example_sessions_per_subject=0,
            heldout_example_sessions_per_subject=0,
        ).load()
        train_shape = list(bundle.train_set.get_all()["xs"].shape)
        eval_shape = list(bundle.eval_set.get_all()["xs"].shape)
        if not train_shape[1] or not eval_shape[1]:
            raise AssertionError(f"{name}: empty adaptation or test loader output")
        if manifest["schema_version"] == 2 and (
            bundle.metadata["split_strategy"] != "within_session_prefix_suffix"
            or "adapt_sessions_per_subject" in bundle.metadata
        ):
            raise AssertionError(f"{name}: v2 did not preserve full-prefix semantics")
        records.append(
            {
                "dataset": name,
                "subject_id": str(subject_id),
                "schema_version": manifest["schema_version"],
                "train_shape": train_shape,
                "eval_shape": eval_shape,
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
