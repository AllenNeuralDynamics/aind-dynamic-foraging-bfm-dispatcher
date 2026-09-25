"""Freeze D=614 source, held-out AIND, and external subject embeddings."""

from __future__ import annotations

import hashlib
import json
import netrc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(REPO / "studies" / "util"))
from _meta import build_meta  # noqa: E402


CACHE = STUDY / "analysis" / "_cache_embedding_space"
OUTPUT = STUDY / "analysis" / "embedding_space_results.json"
MATCHED_DATA = STUDY / "analysis" / "matched_half_results.json"
VALIDATION_DATA = STUDY / "analysis" / "dataset_suite_validation.json"
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
LABELS = {
    "grossman": "Grossman mouse",
    "chen": "Chen mouse",
    "zid": "Zid human",
    "lebedeva": "Lebedeva mouse",
    "beron": "Beron mouse",
    "kwak": "Kwak mouse",
    "miller": "Miller rat",
    "findling": "Findling human",
    "tang": "Tang macaque",
    "alsio": "Alsiö rat",
    "eckstein": "Eckstein human",
    "costa": "Costa macaque",
    "lopez_mouse": "López-Yépez mouse",
}
ARTIFACT_FILES_QUERY = """query ArtifactFiles($id:ID!){
  artifact(id:$id){files(first:1000){edges{node{name directUrl}}}}
}"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wandb_key() -> str:
    if key := os.environ.get("WANDB_API_KEY"):
        return key
    credentials = netrc.netrc().authenticators("api.wandb.ai")
    if not credentials or not credentials[2]:
        raise RuntimeError("W&B credentials are unavailable")
    return credentials[2]


def _wandb_graphql(query: str, variables: dict) -> dict:
    response = requests.post(
        "https://api.wandb.ai/graphql",
        auth=("api", _wandb_key()),
        json={"query": query, "variables": variables},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], indent=2))
    return payload["data"]


def _cache_artifact_files(artifact: dict, destination: Path) -> dict[str, Path]:
    data = _wandb_graphql(ARTIFACT_FILES_QUERY, {"id": artifact["id"]})
    files = [edge["node"] for edge in data["artifact"]["files"]["edges"]]
    wanted = ("subject_embeddings.pkl", "subject_index_map.json", "output_summary.json")
    paths = {}
    for filename in wanted:
        matches = [item for item in files if item["name"].endswith(filename)]
        if len(matches) != 1:
            raise AssertionError(
                f"Artifact {artifact['name']} has ambiguous {filename} files"
            )
        path = destination / filename
        if not path.exists():
            response = requests.get(matches[0]["directUrl"], timeout=300)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
        paths[filename] = path
    return {
        "embeddings": paths["subject_embeddings.pkl"],
        "index_map": paths["subject_index_map.json"],
        "summary": paths["output_summary.json"],
    }


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
    matched = json.loads(MATCHED_DATA.read_text())
    validation = {
        row["dataset"]: row for row in json.loads(VALIDATION_DATA.read_text())["datasets"]
    }
    external_names = tuple(matched["datasets"])
    if set(external_names) != set(validation):
        raise AssertionError("Embedding and validation dataset membership differ")
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
        for dataset in external_names:
            rows = [
                row
                for row in matched["datasets"][dataset]["gru"]
                if int(row["nominal_D"]) == 614 and int(row["seed"]) == seed
            ]
            if len(rows) != 1:
                raise AssertionError(f"Expected one D=614 seed={seed} run for {dataset}")
            run = rows[0]
            artifact = run["evaluation_artifact"]
            paths = _cache_artifact_files(
                artifact, CACHE / "wandb" / dataset / f"s{seed}"
            )
            adapted, _ = _adapted_group(
                paths, source, int(validation[dataset]["num_subjects"]), seed
            )
            groups[dataset] = {
                "subjects": _records(adapted),
                "provenance": {
                    "wandb_run_id": run["wandb_run_id"],
                    "artifact": artifact["name"],
                    "artifact_digest": artifact["digest"],
                    "files": _file_provenance(paths),
                },
            }
        frozen_seeds.append({"seed": seed, "groups": groups})

    output = {
        "_meta": build_meta(
            "studies/09-gru-cross-species-transfer/analysis/freeze_embedding_space.py",
            [
                SOURCE_GROUP,
                *[matched["datasets"][name]["gru_group"] for name in external_names],
            ],
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
                name: {
                    "label": LABELS[name],
                    "species": validation[name]["species"],
                    "n_subjects": validation[name]["num_subjects"],
                }
                for name in external_names
            },
        },
        "seeds": frozen_seeds,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
