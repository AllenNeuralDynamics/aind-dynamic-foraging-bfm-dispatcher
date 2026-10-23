#!/usr/bin/env python
"""Re-check the frozen per-mouse ML tables against MANIFEST.json (no HPC needed)."""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).parent
man = json.loads((HERE / "MANIFEST.json").read_text())
bad = 0
for name, meta in man["files"].items():
    sha = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    n = sum(1 for _ in open(HERE / name)) - 1
    ok = sha == meta["sha256"] and n == meta["n_mice"]
    bad += not ok
    print(f"{'OK ' if ok else 'BAD'} {name} {n} mice")
sys.exit(1 if bad else 0)
