"""Command-line entry point for downloading, adapting, and auditing the suite."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .adapters import build_dataset
from .schema import write_dataset
from .sources import SOURCES, download_source, file_digest, verify_source_file


def _names(value: str) -> list[str]:
    if value == "all":
        return list(SOURCES)
    if value not in SOURCES:
        raise argparse.ArgumentTypeError(
            f"Unknown dataset {value!r}; choose from {list(SOURCES)} or 'all'."
        )
    return [value]


def run(name: str, *, cache_root: Path, download: bool) -> dict[str, object]:
    source = SOURCES[name]
    raw_root = cache_root / "raw"
    source_path = raw_root / name / source.filename
    if download:
        source_path = download_source(name, raw_root)
    else:
        if not source_path.is_file():
            raise FileNotFoundError(
                f"No cached source file at {source_path}. Run without --no-download first."
            )
        verify_source_file(source_path, source)
    df, manifest, audit = build_dataset(name, source_path)
    table_path, manifest_path = write_dataset(
        df,
        manifest,
        output_root=cache_root / "canonical",
        stem=name,
    )
    audit.update(
        {
            "source": asdict(source),
            "verified_source_digest": file_digest(source_path, source.digest_algorithm),
            "canonical_table": str(table_path),
            "split_manifest": str(manifest_path),
        }
    )
    audit_path = cache_root / "canonical" / f"{name}.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=_names,
        help=f"one of {list(SOURCES)}, or all",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require already downloaded, checksum-matching source files.",
    )
    args = parser.parse_args()
    audits = [
        run(name, cache_root=args.cache_root, download=not args.no_download)
        for name in args.dataset
    ]
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    main()
