#!/usr/bin/env python3
"""Report *schedulable* free GPUs across Beaker (AI Hub) and Allen HPC (SLURM).

Why this exists
---------------
The naive "free GPU" counts each backend advertises are misleading:

* **Beaker** reports a node's ``free.gpu_count`` even when the node is
  ``cordoned`` — those GPUs are NOT schedulable. A cluster can show "16 free"
  while every one of them sits on a cordoned node (0 actually launchable).
* **SLURM** ``sinfo`` free counts include nodes in ``drain``/``down``/reserved
  states. The real figure is ``CfgTRES.gres/gpu - AllocTRES.gres/gpu`` on nodes
  that are not drained/down.

This script computes the *schedulable* figure for both, so a launch decision is
made against GPUs that can actually accept a job right now.

CONVENTION (AGENTS.md): **Always run this before launching any large job
(> 4 GPUs / > 4 concurrent tasks).** Route the job to the backend/cluster with
enough schedulable capacity; never assume the skill's preferred cluster order
has free slots.

Usage
-----
    # Beaker only (no VPN needed):
    python code/check_gpu_availability.py --beaker

    # HPC only (needs Allen network / VPN):
    python code/check_gpu_availability.py --hpc

    # Both (default):
    python code/check_gpu_availability.py

    # JSON for programmatic use:
    python code/check_gpu_availability.py --json

Beaker auth: ``BEAKER_TOKEN``, else ``~/.beaker/config.yml`` (``beaker_client.beaker_token``).
HPC: must be run from a host on the Allen network (login node, or the Claude
Science sandbox with VPN up); shells out to ``sinfo``/``scontrol``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

# Beaker hub clusters we are allowed to use (see beaker-launch skill / AGENTS.md §10).
# All known hub pools + the one verified non-hub exception. Dev pools
# (octo-hub-aws-l40s-dev, aihub-dev-aws) are intentionally omitted — we never
# submit to them (AGENTS.md §10).
BEAKER_HUB_CLUSTERS = [
    "ai1/octo.ai-aws-g6e",       # L40S 48GB  — non-hub; low/preemptible-only (verified exception)
    "ai1/octo.ai-aws-p5en",      # H200 141GB — non-hub; low/preemptible-only (verified exception)
    "ai1/octo-hub-aws-l40s",     # L40S 48GB
    "ai1/octo-hub-aws-h200",     # H200 141GB
    "ai1/octo-hub-gcp-h100",     # H100 80GB
    "ai1/octo-hub-onprem-h200",  # H200 141GB — needed for wide H256 (memory, not speed)
    "ai1/octo.hub-gcp-h200",     # H200 141GB
]
HPC_GPU_PARTITION = "aind"


def beaker_schedulable() -> dict:
    """Return {cluster: {schedulable, cordoned_free, total, queued, gpu_type}}."""
    from beaker import Beaker, Config  # lazy import

    from beaker_client import beaker_token  # lazy: same token lookup as the launchers

    try:
        token = beaker_token()
    except KeyError as exc:
        return {"_error": str(exc).strip("'\"")}
    cfg = Config(user_token=token, default_org="ai1",
                 default_workspace="ai1/aind-dynamic-foraging-foundation-model")
    b = Beaker(cfg)
    b._timeout = 30
    out = {}
    for cl in BEAKER_HUB_CLUSTERS:
        try:
            c = b.cluster.get(cl)
            u = b.cluster.utilization(c)
            nodes = getattr(u, "nodes", []) or []
            sched = cordoned_free = total = 0
            gtype = "?"
            for n in nodes:
                fg = getattr(getattr(n, "free", None), "gpu_count", 0) or 0
                tg = getattr(getattr(n, "limits", None), "gpu_count", 0) or 0
                gt = getattr(getattr(n, "limits", None), "gpu_type", None)
                if gt:
                    gtype = gt
                total += tg
                if getattr(n, "cordoned", False):
                    cordoned_free += fg
                else:
                    sched += fg
            out[cl] = {
                "schedulable": sched,
                "cordoned_free": cordoned_free,
                "total": total,
                "queued": getattr(u, "queued_jobs", None),
                "gpu_type": gtype,
            }
        except Exception as e:  # noqa: BLE001
            out[cl] = {"_error": str(e)[:120]}
    return out


def hpc_schedulable(partition: str = HPC_GPU_PARTITION) -> dict:
    """Return {gpu_type: free} for schedulable GPUs on a SLURM partition.

    schedulable = CfgTRES.gres/gpu - AllocTRES.gres/gpu, on nodes not in
    drain/down/reserved states.
    """
    if not shutil.which("sinfo"):
        return {"_error": "sinfo not found (not on Allen network / no SLURM here)"}
    # node -> gpu type
    try:
        typ_raw = subprocess.run(
            ["sinfo", "-h", "-p", partition, "-N", "-o", "%N|%G"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)[:120]}
    ntype = {}
    for line in typ_raw.splitlines():
        if "|" in line:
            node, gres = line.split("|", 1)
            m = re.search(r"gpu:([a-z0-9]+):", gres)
            ntype[node] = m.group(1) if m else "?"
    nodes = sorted(ntype)
    if not nodes:
        return {"_error": f"no GPU nodes on partition {partition}"}
    # per-node Cfg/Alloc via scontrol
    free_by_type: dict = {}
    detail = []
    for node in nodes:
        try:
            raw = subprocess.run(["scontrol", "show", "node", node],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:  # noqa: BLE001
            continue
        raw = raw.replace("\n", " ")
        state = (re.search(r"State=(\S+)", raw) or [None, ""])[1] if re.search(r"State=(\S+)", raw) else ""
        cfg = re.search(r"CfgTRES=(\S*)", raw)
        alloc = re.search(r"AllocTRES=(\S*)", raw)
        cg = re.search(r"gres/gpu=(\d+)", cfg.group(1)) if cfg else None
        ag = re.search(r"gres/gpu=(\d+)", alloc.group(1)) if alloc and alloc.group(1) else None
        cfg_g = int(cg.group(1)) if cg else 0
        alloc_g = int(ag.group(1)) if ag else 0
        fr = cfg_g - alloc_g
        st = state.upper()
        ok = fr > 0 and not any(x in st for x in ("DRAIN", "DOWN", "RESERV"))
        if ok:
            t = ntype.get(node, "?")
            free_by_type[t] = free_by_type.get(t, 0) + fr
            detail.append((node, t, fr, cfg_g, state))
    free_by_type["_total"] = sum(v for k, v in free_by_type.items() if not k.startswith("_"))
    free_by_type["_detail"] = detail
    return free_by_type


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--beaker", action="store_true", help="check Beaker only")
    p.add_argument("--hpc", action="store_true", help="check HPC SLURM only")
    p.add_argument("--partition", default=HPC_GPU_PARTITION, help="SLURM partition (default: aind)")
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args()
    do_beaker = args.beaker or not args.hpc
    do_hpc = args.hpc or not args.beaker

    result = {}
    if do_beaker:
        result["beaker"] = beaker_schedulable()
    if do_hpc:
        result["hpc"] = {args.partition: hpc_schedulable(args.partition)}

    if args.json:
        # drop non-serializable detail tuples into lists
        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, list):
                return [list(x) if isinstance(x, tuple) else _clean(x) for x in o]
            return o
        print(json.dumps(_clean(result), indent=2))
        return

    if "beaker" in result:
        print("=== Beaker (schedulable = free AND not cordoned) ===")
        print(f"{'cluster':30}{'sched':>7}{'cordoned':>10}{'total':>7}{'queued':>8}  gpu")
        for cl, d in result["beaker"].items():
            if "_error" in d:
                print(f"{cl:30}  ERROR: {d['_error']}")
            else:
                print(f"{cl:30}{d['schedulable']:>7}{d['cordoned_free']:>10}"
                      f"{d['total']:>7}{str(d['queued']):>8}  {d['gpu_type']}")
    if "hpc" in result:
        for part, d in result["hpc"].items():
            print(f"\n=== HPC SLURM partition '{part}' (schedulable = Cfg-Alloc, non-drain) ===")
            if "_error" in d:
                print(f"  ERROR: {d['_error']}")
                continue
            print(f"  TOTAL schedulable free GPUs: {d.get('_total', 0)}")
            for t, n in sorted(((k, v) for k, v in d.items() if not k.startswith("_")),
                               key=lambda x: -x[1]):
                print(f"    {t:10} {n}")


if __name__ == "__main__":
    main()
