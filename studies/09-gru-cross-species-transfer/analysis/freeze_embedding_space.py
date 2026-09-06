"""Freeze D=614 source, held-out AIND, and external subject embeddings."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402


CACHE = STUDY / "analysis" / "_cache_embedding_space"
OUTPUT = STUDY / "analysis" / "embedding_space_results.json"
EMBEDDING_COLUMNS = [f"embedding_{index}" for index in range(1, 5)]
SOURCE_GROUP = "v2-sc-active@20260622-144622"
SOURCE_RUNS = {
    0: {
        "run_id": "v2-sc-active-20260622-144622-ecd6f7e6",
        "artifact_digest": "62e1a1e1615dbfbaa93af64b39708a77",
        "beaker_result_dataset": "01KVRMSBRV8QFHGS3XYRAS9P9A",
    },
    1: {
        "run_id": "v2-sc-active-20260622-144622-3603897b",
        "artifact_digest": "08e27c94c5308f48d7c4dca284a0a428",
        "beaker_result_dataset": "01KVRMSBW2H0EF8KZZJFVDDPGG",
    },
    2: {
        "run_id": "v2-sc-active-20260622-144622-d6462060",
        "artifact_digest": "c2a992ec6a687a094e955850807a03c0",
        "beaker_result_dataset": "01KVRMSBZBXJ8V0CQZ1V5PJEWM",
    },
}
EXTERNAL = {
    "grossman": {
        "label": "Grossman mouse",
        "species": "mouse",
        "n_subjects": 48,
        "group": "gru-grossman-matched-half@20260905-022602",
        "runs": {
            0: ("gru-grossman-matched-half-20260905-022602-b2549a1d", "b54d6203b28019482291062de3c90946"),
            1: ("gru-grossman-matched-half-20260905-022602-e8427c97", "8220dd5a4d5cb6897bca47839ea7dc67"),
            2: ("gru-grossman-matched-half-20260905-022602-dd860bb0", "343fb34fe456d935ec9b94560940817d"),
        },
    },
    "chen": {
        "label": "Chen mouse",
        "species": "mouse",
        "n_subjects": 32,
        "group": "gru-chen-matched-half@20260905-024731",
        "runs": {
            0: ("gru-chen-matched-half-20260905-024731-5ae04883", "e28ec502a5eff48ca378189071f9a6a2"),
            1: ("gru-chen-matched-half-20260905-024731-1cb7ef49", "eeaf5f83392dc371f95e1813bf73d24f"),
            2: ("gru-chen-matched-half-20260905-024731-94c97219", "e704179f5a68d4e655f5a0ab451dcd00"),
        },
    },
    "zid": {
        "label": "Zid human",
        "species": "human",
        "n_subjects": 258,
        "group": "gru-zid-matched-half@20260905-025752",
        "runs": {
            0: ("gru-zid-matched-half-20260905-025752-39eea37f", "66409894b5f57307686d90098429f1e8"),
            1: ("gru-zid-matched-half-20260905-025752-5f3c4f65", "a8ed8496a3cb33b366ce85809d34f530"),
            2: ("gru-zid-matched-half-20260905-025752-d6cedfeb", "6057d5ab3d41a3fc8f4b04040cd45a69"),
        },
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(directory: Path) -> dict[str, Path]:
    return {
        "embeddings": directory / "subject_embeddings.pkl",
        "index_map": directory / "subject_index_map.json",
        "summary": directory / "output_summary.json",
    }


def _load_table(path: Path) -> pd.DataFrame:
    table = pd.read_pickle(path)
    required = {"subject_index", "subject_id", *EMBEDDING_COLUMNS}
    if not required.issubset(table.columns):
        raise AssertionError(f"Missing embedding columns in {path}")
    table = table.sort_values("subject_index").reset_index(drop=True)
    if table["subject_index"].tolist() != list(range(len(table))):
        raise AssertionError(f"Non-contiguous subject indices in {path}")
    if table["subject_id"].astype(str).duplicated().any():
        raise AssertionError(f"Duplicate subject IDs in {path}")
    if not np.isfinite(table[EMBEDDING_COLUMNS].to_numpy(dtype=float)).all():
        raise AssertionError(f"Non-finite embeddings in {path}")
    return table


def _validate_index_map(table: pd.DataFrame, path: Path) -> None:
    mapping = json.loads(path.read_text())["subject_id_to_index"]
    observed = dict(zip(table["subject_id"].astype(str), table["subject_index"].astype(int)))
    if mapping != observed:
        raise AssertionError(f"Subject index map disagrees with table in {path}")


def _records(table: pd.DataFrame) -> list[dict]:
    return [
        {
            "subject_id": str(row.subject_id),
            "embedding": [float(getattr(row, column)) for column in EMBEDDING_COLUMNS],
        }
        for row in table.itertuples(index=False)
    ]


def _file_provenance(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {
        name: {
            "cache_path": str(path.relative_to(CACHE)),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
        if path.exists()
    }


def _adapted_group(
    paths: dict[str, Path], source: pd.DataFrame, expected_n: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    table = _load_table(paths["embeddings"])
    _validate_index_map(table, paths["index_map"])
    summary = json.loads(paths["summary"].read_text())
    if int(summary["seed"]) != seed:
        raise AssertionError(f"Seed mismatch in {paths['summary']}")
    if int(summary["n_steps"]) != 500 or float(summary["lr"]) != 0.001:
        raise AssertionError(f"Adaptation contract mismatch in {paths['summary']}")
    source_part = table.iloc[: len(source)]
    if source_part["subject_id"].astype(str).tolist() != source["subject_id"].astype(str).tolist():
        raise AssertionError("Adapted table source subject IDs do not match")
    if not np.array_equal(
        source_part[EMBEDDING_COLUMNS].to_numpy(),
        source[EMBEDDING_COLUMNS].to_numpy(),
    ):
        raise AssertionError("Adapted table changed source embeddings")
    adapted = table.iloc[len(source) :].reset_index(drop=True)
    expected_ids = [str(value) for value in summary["heldout_subject_ids"]]
    if len(adapted) != expected_n or adapted["subject_id"].astype(str).tolist() != expected_ids:
        raise AssertionError(f"Adapted subject contract mismatch in {paths['embeddings']}")
    return adapted, summary


def main() -> None:
    source_runs = json.loads((STUDY / "source_runs.json").read_text())
    frozen_seeds = []
    for seed in range(3):
        declared = source_runs["runs"][f"d614-s{seed}"]
        expected = SOURCE_RUNS[seed]
        if declared["run_id"] != expected["run_id"] or declared["artifact_digest"] != expected["artifact_digest"]:
            raise AssertionError(f"Pinned source run drift for seed {seed}")

        source_paths = _paths(CACHE / "beaker" / f"s{seed}" / "run" / "outputs")
        source_paths.pop("summary")
        source = _load_table(source_paths["embeddings"])
        _validate_index_map(source, source_paths["index_map"])
        if len(source) != 614:
            raise AssertionError(f"Expected 614 source subjects for seed {seed}")

        heldout_root = CACHE / "beaker" / f"s{seed}" / "heldout_subject_finetuning"
        heldout_directories = list(heldout_root.glob("**/outputs"))
        if len(heldout_directories) != 1:
            raise AssertionError(f"Expected one held-out output directory for seed {seed}")
        heldout_paths = _paths(heldout_directories[0])
        heldout, _ = _adapted_group(heldout_paths, source, 149, seed)

        groups = {
            "aind_source": {
                "subjects": _records(source),
                "provenance": {
                    **expected,
                    "files": _file_provenance(source_paths),
                },
            },
            "aind_heldout": {
                "subjects": _records(heldout),
                "provenance": {
                    "beaker_result_dataset": expected["beaker_result_dataset"],
                    "files": _file_provenance(heldout_paths),
                },
            },
        }
        for dataset, specification in EXTERNAL.items():
            paths = _paths(CACHE / "wandb" / dataset / f"s{seed}")
            adapted, _ = _adapted_group(
                paths, source, int(specification["n_subjects"]), seed
            )
            run_id, artifact_digest = specification["runs"][seed]
            groups[dataset] = {
                "subjects": _records(adapted),
                "provenance": {
                    "wandb_run_id": run_id,
                    "artifact": f"external-gru-output-{run_id}:v0",
                    "artifact_digest": artifact_digest,
                    "files": _file_provenance(paths),
                },
            }
        frozen_seeds.append({"seed": seed, "groups": groups})

    output = {
        "_meta": build_meta(
            "studies/09-gru-cross-species-transfer/analysis/freeze_embedding_space.py",
            [SOURCE_GROUP, *[specification["group"] for specification in EXTERNAL.values()]],
            study_root=STUDY,
        ),
        "contract": {
            "source_model": "GRU H=128, D=614",
            "embedding_dimensions": 4,
            "seeds": [0, 1, 2],
            "pca_fit_population": "614 source-training AIND mice, separately per seed",
            "adapted_populations": "149 held-out AIND mice and every external subject",
            "new_subject_initialization": "mean of the source-subject embeddings",
            "adaptation": {"trainable_parameters": "subject embedding only", "n_steps": 500, "lr": 0.001},
            "cross_seed_rule": "Never pool raw coordinates; fit and interpret each seed separately",
        },
        "groups": {
            "aind_source": {"label": "AIND source-training mice", "species": "mouse", "n_subjects": 614},
            "aind_heldout": {"label": "AIND held-out mice", "species": "mouse", "n_subjects": 149},
            **{
                name: {key: specification[key] for key in ("label", "species", "n_subjects")}
                for name, specification in EXTERNAL.items()
            },
        },
        "seeds": frozen_seeds,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
