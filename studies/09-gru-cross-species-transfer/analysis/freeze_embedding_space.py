"""Freeze D=614 source, held-out AIND, and external subject embeddings."""

from __future__ import annotations

import argparse
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


CACHE_ROOT = STUDY / "analysis" / "_cache_embedding_space"
OUTPUTS = {
    4: STUDY / "analysis" / "embedding_space_results.json",
    8: STUDY / "analysis" / "embedding_space_results_e8.json",
}
MATCHED_DATA = STUDY / "analysis" / "matched_half_results.json"
E8_DATA = (
    STUDY / "analysis" / "embedding_dimension_results.json",
    STUDY / "analysis" / "embedding_dimension_expansion_results.json",
)
E8_TARGET_GROUPS = (
    "gru-e8-d614-diagnostic@20260907-071400",
    "gru-e8-d614-expansion@20260907-132415",
    "gru-e8-d614-expansion@20260907-132625",
    "gru-hattori-matched-half@20260907-200328",
)
VALIDATION_DATA = STUDY / "analysis" / "dataset_suite_validation.json"
SOURCE_MANIFESTS = {
    4: STUDY / "source_runs.json",
    8: STUDY / "source_runs_e8.json",
}
E4_SOURCE_RESULT_DATASETS = {
    0: "01KVRMSBRV8QFHGS3XYRAS9P9A",
    1: "01KVRMSBW2H0EF8KZZJFVDDPGG",
    2: "01KVRMSBZBXJ8V0CQZ1V5PJEWM",
}
VALID_DATASETS = (
    "grossman",
    "chen",
    "zid",
    "lebedeva",
    "beron",
    "miller",
    "findling",
    "tang",
    "alsio",
    "eckstein",
    "costa",
    "lopez_mouse",
    "hattori",
)
LABELS = {
    "grossman": "Grossman (mouse)",
    "chen": "Chen (mouse)",
    "zid": "Zid (human)",
    "lebedeva": "Lebedeva (mouse)",
    "beron": "Beron (mouse)",
    "kwak": "Kwak (mouse)",
    "miller": "Miller (rat)",
    "findling": "Findling (human)",
    "tang": "Tang (macaque)",
    "alsio": "Alsiö (rat)",
    "eckstein": "Eckstein (human)",
    "costa": "Costa (macaque)",
    "lopez_mouse": "López-Yépez (mouse)",
    "hattori": "Hattori (mouse)",
}
ARTIFACT_FILES_QUERY = """query ArtifactFiles($id:ID!){
  artifact(id:$id){files(first:1000){edges{node{name directUrl}}}}
}"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(4, 8), default=4)
    return parser


def _dimension_inputs(dimension: int) -> tuple[dict, dict[str, dict], list[str]]:
    source_runs = json.loads(SOURCE_MANIFESTS[dimension].read_text())
    if dimension == 4:
        matched = json.loads(MATCHED_DATA.read_text())
        targets = {name: matched["datasets"][name] for name in VALID_DATASETS}
        groups = [targets[name]["gru_group"] for name in VALID_DATASETS]
    else:
        frozen = [json.loads(path.read_text()) for path in E8_DATA]
        targets = {
            name: dataset
            for document in frozen
            for name, dataset in document["datasets"].items()
        }
        groups = list(E8_TARGET_GROUPS)
    if set(targets) != set(VALID_DATASETS):
        raise AssertionError(f"E={dimension} target membership drifted")
    return source_runs, targets, list(dict.fromkeys(groups))


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
    cached = {
        "embeddings": destination / "subject_embeddings.pkl",
        "index_map": destination / "subject_index_map.json",
        "summary": destination / "output_summary.json",
    }
    if all(path.exists() for path in cached.values()):
        return cached
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


def _load_table(path: Path, embedding_columns: list[str]) -> pd.DataFrame:
    table = pd.read_pickle(path)
    required = {"subject_index", "subject_id", *embedding_columns}
    if not required.issubset(table.columns):
        raise AssertionError(f"Missing embedding columns in {path}")
    table = table.sort_values("subject_index").reset_index(drop=True)
    if table["subject_index"].tolist() != list(range(len(table))):
        raise AssertionError(f"Non-contiguous subject indices in {path}")
    if table["subject_id"].astype(str).duplicated().any():
        raise AssertionError(f"Duplicate subject IDs in {path}")
    if not np.isfinite(table[embedding_columns].to_numpy(dtype=float)).all():
        raise AssertionError(f"Non-finite embeddings in {path}")
    return table


def _validate_index_map(table: pd.DataFrame, path: Path) -> None:
    mapping = json.loads(path.read_text())["subject_id_to_index"]
    observed = dict(
        zip(table["subject_id"].astype(str), table["subject_index"].astype(int))
    )
    if mapping != observed:
        raise AssertionError(f"Subject index map disagrees with table in {path}")


def _records(table: pd.DataFrame, embedding_columns: list[str]) -> list[dict]:
    return [
        {
            "subject_id": str(row.subject_id),
            "embedding": [float(getattr(row, column)) for column in embedding_columns],
        }
        for row in table.itertuples(index=False)
    ]


def _file_provenance(
    paths: dict[str, Path], cache: Path
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "cache_path": str(path.relative_to(cache)),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
        if path.exists()
    }


def _adapted_group(
    paths: dict[str, Path],
    source: pd.DataFrame,
    expected_n: int,
    seed: int,
    embedding_columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    table = _load_table(paths["embeddings"], embedding_columns)
    _validate_index_map(table, paths["index_map"])
    summary = json.loads(paths["summary"].read_text())
    if int(summary["seed"]) != seed:
        raise AssertionError(f"Seed mismatch in {paths['summary']}")
    if int(summary["n_steps"]) != 500 or float(summary["lr"]) != 0.001:
        raise AssertionError(f"Adaptation contract mismatch in {paths['summary']}")
    source_part = table.iloc[: len(source)]
    if source_part["subject_id"].astype(str).tolist() != source[
        "subject_id"
    ].astype(str).tolist():
        raise AssertionError("Adapted table source subject IDs do not match")
    if not np.array_equal(
        source_part[embedding_columns].to_numpy(),
        source[embedding_columns].to_numpy(),
    ):
        raise AssertionError("Adapted table changed source embeddings")
    adapted = table.iloc[len(source) :].reset_index(drop=True)
    expected_ids = [str(value) for value in summary["heldout_subject_ids"]]
    if (
        len(adapted) != expected_n
        or adapted["subject_id"].astype(str).tolist() != expected_ids
    ):
        raise AssertionError(f"Adapted subject contract mismatch in {paths['embeddings']}")
    return adapted, summary


def main() -> None:
    dimension = _parser().parse_args().dimension
    embedding_columns = [f"embedding_{index}" for index in range(1, dimension + 1)]
    cache = CACHE_ROOT / f"e{dimension}"
    output_path = OUTPUTS[dimension]
    source_runs, targets, target_groups = _dimension_inputs(dimension)
    validation = {
        row["dataset"]: row for row in json.loads(VALIDATION_DATA.read_text())["datasets"]
    }
    external_names = VALID_DATASETS
    if not set(external_names).issubset(validation):
        raise AssertionError("Embedding datasets are absent from validation")
    frozen_seeds = []
    for seed in range(3):
        source_key = f"d614-s{seed}" if dimension == 4 else f"e8-d614-s{seed}"
        expected = source_runs["runs"][source_key]
        if int(expected["seed"]) != seed:
            raise AssertionError(f"Pinned source seed drift for seed {seed}")

        source_paths = _paths(cache / "beaker" / f"s{seed}" / "run" / "outputs")
        source_paths.pop("summary")
        source = _load_table(source_paths["embeddings"], embedding_columns)
        _validate_index_map(source, source_paths["index_map"])
        if len(source) != 614:
            raise AssertionError(f"Expected 614 source subjects for seed {seed}")

        heldout_root = cache / "beaker" / f"s{seed}" / "heldout_subject_finetuning"
        heldout_directories = list(heldout_root.glob("**/outputs"))
        if len(heldout_directories) != 1:
            raise AssertionError(f"Expected one held-out output directory for seed {seed}")
        heldout_paths = _paths(heldout_directories[0])
        heldout, _ = _adapted_group(
            heldout_paths, source, 149, seed, embedding_columns
        )

        groups = {
            "aind_source": {
                "subjects": _records(source, embedding_columns),
                "provenance": {
                    **expected,
                    "files": _file_provenance(source_paths, cache),
                },
            },
            "aind_heldout": {
                "subjects": _records(heldout, embedding_columns),
                "provenance": {
                    "beaker_result_dataset": expected.get(
                        "beaker_result_dataset_id",
                        expected.get(
                            "beaker_result_dataset", E4_SOURCE_RESULT_DATASETS.get(seed)
                        ),
                    ),
                    "files": _file_provenance(heldout_paths, cache),
                },
            },
        }
        for dataset in external_names:
            rows = [
                row
                for row in (
                    targets[dataset]["gru"]
                    if dimension == 4
                    else targets[dataset]["e8"]
                )
                if (dimension != 4 or int(row["nominal_D"]) == 614)
                and int(row["seed"]) == seed
            ]
            if len(rows) != 1:
                raise AssertionError(f"Expected one D=614 seed={seed} run for {dataset}")
            run = rows[0]
            artifact = run["evaluation_artifact"]
            paths = _cache_artifact_files(
                artifact, cache / "wandb" / dataset / f"s{seed}"
            )
            adapted, _ = _adapted_group(
                paths,
                source,
                int(validation[dataset]["num_subjects"]),
                seed,
                embedding_columns,
            )
            groups[dataset] = {
                "subjects": _records(adapted, embedding_columns),
                "provenance": {
                    "wandb_run_id": run["wandb_run_id"],
                    "artifact": artifact["name"],
                    "artifact_digest": artifact["digest"],
                    "files": _file_provenance(paths, cache),
                },
            }
        frozen_seeds.append({"seed": seed, "groups": groups})

    output = {
        "_meta": build_meta(
            "studies/09-gru-cross-species-transfer/analysis/freeze_embedding_space.py",
            [
                source_runs["group"],
                *target_groups,
            ],
            study_root=STUDY,
        ),
        "contract": {
            "source_model": f"GRU H=128, D=614, E={dimension}",
            "embedding_dimensions": dimension,
            "seeds": [0, 1, 2],
            "pca_fit_population": "614 source-training AIND mice, separately per seed",
            "adapted_populations": "149 held-out AIND mice and every external subject",
            "new_subject_initialization": "mean of the source-subject embeddings",
            "adaptation": {"trainable_parameters": "subject embedding only", "n_steps": 500, "lr": 0.001},
            "cross_seed_rule": "Never pool raw coordinates; fit and interpret each seed separately",
        },
        "groups": {
            "aind_source": {
                "label": "AIND source-training mice",
                "species": "mouse",
                "n_subjects": 614,
            },
            "aind_heldout": {
                "label": "AIND held-out mice",
                "species": "mouse",
                "n_subjects": 149,
            },
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
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
