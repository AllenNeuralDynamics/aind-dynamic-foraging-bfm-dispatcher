#!/usr/bin/env python3
"""Offline regeneration check for every study: `make all` must reproduce what is committed.

This is the studies repository's CI gate (posthoc-reporting: reports are code, regenerated
from committed frozen inputs; two runs differ only in `_meta`). It runs each study's
`make all` with W&B disabled, then compares the working tree against HEAD:

  * `_meta` blocks in JSON are ignored (timestamps, checkout SHA, dirty flag);
  * numbers in JSON and CSV are compared with a relative tolerance, because the same
    producer on another machine can differ in the 15th significant digit;
  * rendered images (PNG / SVG / PDF) are ignored: their bytes depend on the plotting
    stack, and the numbers behind them are checked through the JSON / CSV;
  * everything else (report text inside the BEGIN/END markers, CSV rows, JSON keys) must
    match exactly.

Any remaining difference is a result change, which belongs in a reviewed commit, not in a
CI run. Exit 1 on a failed `make`, an unparseable artifact, or an unexplained difference.
Known gaps are listed in SKIP_MAKE / KNOWN_DIFFS with the issue that owns them and are
printed as EXEMPT, never silently skipped.

Usage: python3 scripts/check_regeneration.py [--study 08-hb-vs-gru-heldout ...] [--rtol 1e-9]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".svg", ".pdf"}

# Known gaps, each owned by an issue, printed as EXEMPT so they stay visible. Remove an
# entry when its issue closes. SKIP_MAKE: the study's `all` target needs W&B credentials.
# KNOWN_DIFFS: the path regenerates differently from what is committed.
SKIP_MAKE = {
    "02-gru-scaling-law-ignore": "dispatcher #177: `all` runs the W&B pull (no offline report target)",
    "07-gru-timing-inputs": "dispatcher #177: `all` runs the W&B pull (no offline report target)",
}
KNOWN_DIFFS = {
    "studies/04-gru-vs-disrnn-embedding-recovery/analysis/reports/r1-gru-ladder.md":
        "dispatcher #177: committed report text differs from what the producer writes",
    "studies/05-disrnn-scaling-law/analysis/reports/r4-generative-behavioral-match.md":
        "dispatcher #177: committed report text differs from what the producer writes",
    "studies/09-gru-cross-species-transfer/analysis/reports/r5-maturity-audit.md":
        "dispatcher #177: committed report text differs from what the producer writes",
    "studies/09-gru-cross-species-transfer/analysis/llm_task_similarity_results.json":
        "dispatcher #177: PCA sign indeterminacy",
}


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, **kw)


def _head(path: str) -> str | None:
    out = _run(["git", "show", f"HEAD:{path}"])
    return out.stdout if out.returncode == 0 else None


def _close(a, b, rtol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)
    return False


def _json_equal(a, b, rtol: float, path: str = "$") -> list[str]:
    """Differences between two parsed JSON values, ignoring `_meta` and float noise."""
    if isinstance(a, dict) and isinstance(b, dict):
        a = {k: v for k, v in a.items() if k != "_meta"}
        b = {k: v for k, v in b.items() if k != "_meta"}
        diffs = [f"{path}.{k}: only in {'HEAD' if k in a else 'regenerated'}"
                 for k in sorted(set(a) ^ set(b))]
        for k in sorted(set(a) & set(b)):
            diffs += _json_equal(a[k], b[k], rtol, f"{path}.{k}")
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out: list[str] = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _json_equal(x, y, rtol, f"{path}[{i}]")
        return out
    if a == b or _close(a, b, rtol):
        return []
    return [f"{path}: {a!r} != {b!r}"]


def _cell_equal(a: str, b: str, rtol: float) -> bool:
    if a == b:
        return True
    try:
        return _close(float(a), float(b), rtol)
    except ValueError:
        return False


def _csv_diffs(old: str, new: str, rtol: float) -> list[str]:
    ra, rb = list(csv.reader(io.StringIO(old))), list(csv.reader(io.StringIO(new)))
    if len(ra) != len(rb):
        return [f"rows {len(ra)} != {len(rb)}"]
    out = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        if len(x) != len(y):
            out.append(f"row {i}: {len(x)} != {len(y)} columns")
            continue
        for j, (p, q) in enumerate(zip(x, y)):
            if not _cell_equal(p, q, rtol):
                out.append(f"row {i} col {j}: {p!r} != {q!r}")
    return out[:20]


def compare(path: str, rtol: float) -> list[str]:
    """Differences between HEAD and the working tree for one path; [] when equivalent."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return []
    old = _head(path)
    if old is None:
        return ["not in HEAD (new file)"]
    new = (ROOT / path).read_text()
    if suffix == ".json":
        try:
            return _json_equal(json.loads(old), json.loads(new), rtol)
        except json.JSONDecodeError as exc:
            return [f"unparseable JSON: {exc}"]
    if suffix == ".csv":
        return _csv_diffs(old, new, rtol)
    return [] if old == new else ["content differs"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", action="append", default=None,
                   help="study folder name(s) to check; default: every studies/NN-* with a Makefile")
    p.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance for numbers")
    p.add_argument("--no-make", action="store_true", help="only compare; skip `make all`")
    args = p.parse_args()

    studies = ([ROOT / "studies" / s for s in args.study] if args.study
               else sorted(d for d in (ROOT / "studies").glob("[0-9]*") if d.is_dir()))
    env = {**os.environ, "WANDB_MODE": "disabled", "MPLBACKEND": "Agg"}
    env.pop("WANDB_API_KEY", None)
    failed = 0

    if not args.no_make:
        for study in studies:
            if not (study / "Makefile").exists():
                print(f"EXEMPT {study.name}: no Makefile")
                continue
            if study.name in SKIP_MAKE:
                print(f"EXEMPT {study.name}: {SKIP_MAKE[study.name]}")
                continue
            print(f"== make -C studies/{study.name} all", flush=True)
            out = subprocess.run(["make", "-C", str(study), "all"], cwd=ROOT, env=env,
                                 text=True, capture_output=True)
            if out.returncode != 0:
                failed += 1
                print(f"FAIL make all in {study.name} (exit {out.returncode})")
                print((out.stdout + out.stderr)[-4000:])

    changed = [l[3:] for l in _run(["git", "status", "--porcelain", "--", "studies"]).stdout.splitlines()]
    wanted = {s.name for s in studies}
    for path in sorted(changed):
        parts = Path(path).parts
        if len(parts) < 2 or parts[1] not in wanted:
            continue
        diffs = compare(path, args.rtol)
        if diffs and path in KNOWN_DIFFS:
            print(f"EXEMPT  {path}: {KNOWN_DIFFS[path]}")
            continue
        if diffs:
            failed += 1
            print(f"CHANGED {path}")
            for d in diffs[:10]:
                print(f"    {d}")
        else:
            print(f"ok      {path} (differs only in _meta / float noise / image bytes)")

    print(f"\n{'FAIL' if failed else 'OK'}: {len(studies)} studies checked, {failed} problem(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
