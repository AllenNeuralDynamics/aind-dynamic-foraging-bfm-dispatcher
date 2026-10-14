#!/usr/bin/env python
"""Regenerate the producer-owned regions of study 08's reports from the frozen analysis JSON.

Each report has one region between ``<!-- BEGIN result-N -->`` and ``<!-- END result-N -->``. This script
replaces exactly that region and nothing else: it matches on the marker pair, never on adjacent prose or
line numbers, and fails if a marker pair is missing or duplicated. Prose outside the markers belongs to
the authors and is never touched.

    python update_reports.py --report r1        # one report
    python update_reports.py --report all       # all four
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPORTS = HERE / "reports"
PARAMS = ("learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen", "softmax_inverse_temperature", "bias_l")
PNAME = {"learn_rate_rew": "learning rate, rewarded", "learn_rate_unrew": "learning rate, unrewarded",
         "forget_rate_unchosen": "forgetting rate", "softmax_inverse_temperature": "inverse temperature", "bias_l": "left bias"}
REPORT_FILE = {"r1": "r1-three-arm-heldout.md", "r2": "r2-per-mouse-paired.md", "r3": "r3-shrinkage-by-parameter.md", "r4": "r4-convergence-and-sampler-cost.md"}
SCRIPT = {"r1": "three_arm.py", "r2": "per_mouse.py", "r3": "shrinkage.py", "r4": "convergence.py"}


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def f(x, n=5):
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def region_r1() -> str:
    d = load("three_arm.json")
    rows = [[str(r["D"]), f(r["hb_matched"]), f(r["gru_v2_mean"]), f"{r['gap_gru_minus_hb']:+.5f}", f(r["max_rhat"], 3),
             str(r["divergences"]), "no" if r["not_mixed"] else "yes (still above the 1.01 gate)"] for r in d["hb_headline"]]
    rl = d["rl"]
    return "\n".join([
        f"Held-out cohort: {d['cohort']['n_subjects']} mice, {d['cohort']['n_trials']:,} trials. {d['aggregation'].capitalize()}.",
        "", f"![Held-out likelihood of the hierarchical model, per-mouse classical RL and the GRU](../fig_three_arm.png)", "",
        "**Hierarchical model (HB) at each training-cohort size, against the GRU** (HB: one seed, warmup 2000, 500 draws, 4 chains; GRU: "
        f"{d['gru_variant']}, mean of its seeds):", "",
        table(["D", "HB", "GRU", "GRU minus HB", "HB max R-hat", "HB divergences", "R-hat <= 1.1?"], rows), "",
        "**Classical RL, one fit per mouse (same 149 mice):** "
        f"compare-to-threshold {rl['ctt']:.5f}, Bari {rl['bari']:.5f}, Hattori2019 (the model HB uses) {rl['hattori']:.5f}.",
    ])


def region_r2() -> str:
    d = load("per_mouse.json")
    s = [r for r in d["summary"]]
    by = {}
    for r in s:
        by.setdefault(r["D"], {})[r["rl_model"]] = r
    rows = []
    for D in sorted(by):
        h, b, c = by[D]["hattori"], by[D]["bari"], by[D]["ctt"]
        rows.append([str(D), f"{h['n_hb_worse']} / {h['n_mice']}", f"{h['median_diff']:+.5f}", f"{h['wilcoxon_p']:.1e}",
                     f"{b['n_hb_worse']} / {b['n_mice']}", f"{c['n_hb_worse']} / {c['n_mice']}", "no" if not h["not_mixed"] else "yes"])
    return "\n".join([
        f"Held-out cohort: {d['cohort']['n_mice']} mice. Pairing: {d['pairing']}.", "",
        "![Per-mouse HB likelihood against the same mouse's own fit](../fig_per_mouse.png)", "",
        "**Mice on which HB predicts worse than the same mouse's own fit** (median paired difference and Wilcoxon p are for the Hattori2019 fit, "
        "the same model family as HB):", "",
        table(["D", "vs Hattori2019 (n mice)", "median HB minus Hattori2019", "Wilcoxon p", "vs Bari", "vs compare-to-threshold", "chains not mixed"], rows),
    ])


def region_r3() -> str:
    d = load("shrinkage.json")
    rows_by = {}
    for r in d["rows"]:
        rows_by.setdefault(r["param"], {})[r["D"]] = r
    Ds = sorted({r["D"] for r in d["rows"]})
    hdr = ["parameter"] + [f"D = {D}" for D in Ds]
    body = [[PNAME[p]] + [f(rows_by[p][D]["theilsen_slope"], 2) + ("*" if rows_by[p][D]["not_mixed"] else "") for D in Ds] for p in PARAMS]
    n_rows = [["mice used (of all)"] + [f"{rows_by['learn_rate_rew'][D]['n_used']}/{rows_by['learn_rate_rew'][D]['n_all']}" for D in Ds]]
    rho = [[PNAME[p]] + [f(rows_by[p][D]["spearman"], 2) for D in Ds] for p in PARAMS]
    return "\n".join([
        f"Exclusions from the primary slope: {d['exclusion']}. Slopes need at least {d['min_n_for_slope']} mice. `*` marks a rung whose chains are not mixed.", "",
        "![HB posterior mean against each mouse's own fit, by parameter](../fig_shrinkage.png)", "",
        "**Theil-Sen slope of the HB posterior mean on the mouse's own fit** (1 = individual differences preserved; below 1 = compressed toward the mean):", "",
        table(hdr, body), "",
        "**Spearman rank correlation between the two** (does HB order mice the way their own fits do):", "", table(hdr, rho), "",
        "Mice entering each slope, shown for the rewarded learning rate (no exclusions apply to it): " + ", ".join(f"D = {D}: {rows_by['learn_rate_rew'][D]['n_used']}/{rows_by['learn_rate_rew'][D]['n_all']}" for D in Ds) + ". "
        "Other parameters lose mice to the exclusions, most at the unrewarded learning rate (pinned at 0) and forgetting rate (pinned at 1); the per-parameter counts are in `analysis/shrinkage.csv`.",
    ])


def region_r4() -> str:
    d = load("convergence.json")
    runs = d["numpyro_runs"]
    head = sorted([r for r in runs if r["group"] in ("one-stage-ladder@20260903-154819", "one-stage-ladder@20260903-154828")], key=lambda r: r["D"])
    a = [[str(r["D"]), f(r["group_max_rhat"], 3), f(r["group_min_ess"], 0), f(r["subject_max_rhat"], 3), f(r["subject_min_ess"], 0), str(r["divergences"])] for r in head]
    order = [("one-stage-ladder@20260903-052244", "500 / 500"), ("one-stage-ladder@20260903-154819", "2000 / 500"),
             ("one-stage-ladder@20260903-154825", "4000 / 500"), ("one-stage-ladder@20260904-082628", "2000 / 2000")]
    b = []
    for g, lab in order:
        r = next(x for x in runs if x["group"] == g and x["D"] == 29)
        b.append([lab, f(max(r["group_max_rhat"], r["subject_max_rhat"]), 3), f(min(r["group_min_ess"], r["subject_min_ess"]), 0), str(r["divergences"]),
                  "yes" if r["passes_all_gates"] else "no"])
    c = d["stan_vs_numpyro_D29"]; n, s_ = c["numpyro"], c["stan"]; ra = c["ratios"]
    cmp_rows = [["divergences", str(n["divergences"]), str(s_["divergences"])],
                ["max R-hat (population means)", f(n["max_rhat"], 4), f(s_["max_rhat"], 4)],
                ["lowest bulk ESS (population means)", f(n["min_ess_bulk"], 0), f(s_["min_ess_bulk"], 0)],
                ["ESS per retained draw", f(n["ess_per_draw"], 3), f(s_["ess_per_draw"], 3)],
                ["wall time (hours), as run", f(n["fit_seconds"] / 3600, 2), f(s_["fit_seconds"] / 3600, 2)],
                ["ESS per second", f(n["ess_per_second"], 3), f(s_["ess_per_second"], 3)],
                ["hardware", n["hardware"], s_["hardware"]]]
    return "\n".join([
        f"Gates: no divergences, worst R-hat below {d['gates']['max_rhat']}, lowest bulk ESS at least {d['gates']['min_ess_bulk']}.", "",
        "![Convergence of the NumPyro HB fits](../fig_convergence.png)", "",
        "**By training-cohort size** (warmup 2000, 500 draws, 4 chains; recomputed from each run's saved posterior):", "",
        table(["D", "group-level worst R-hat", "group-level lowest ESS", "per-mouse worst R-hat", "per-mouse lowest ESS", "divergences"], a), "",
        "**By sampler setting at D = 29** (warmup / draws per chain; worst and lowest over every group-level and per-mouse site):", "",
        table(["setting", "worst R-hat", "lowest ESS", "divergences", "passes all gates"], b), "",
        f"**Stan against NumPyro at D = 29**, {c['settings']}; diagnostics on {c['diagnostic_sites']}:", "",
        table(["", "NumPyro", "Stan"], cmp_rows), "",
        f"Relative to NumPyro, Stan's effective sample per draw is {ra['stan_ess_per_draw_over_numpyro']:.2f}x, its wall time as run is "
        f"{ra['stan_wall_over_numpyro']:.1f}x, and its effective sample per second is {ra['stan_ess_per_second_over_numpyro']:.2f}x.",
    ])


REGION = {"r1": region_r1, "r2": region_r2, "r3": region_r3, "r4": region_r4}


def update(rid: str) -> None:
    path = REPORTS / REPORT_FILE[rid]
    text = path.read_text()
    n = rid[1:]
    pat = re.compile(rf"(<!-- BEGIN result-{n} -->\n)(.*?)(<!-- END result-{n} -->)", re.S)
    if len(pat.findall(text)) != 1:
        raise SystemExit(f"{path.name}: expected exactly one BEGIN/END result-{n} marker pair, found {len(pat.findall(text))}")
    head = f"[regenerated by `analysis/update_reports.py --report {rid}` from `analysis/{SCRIPT[rid]}` output; do not edit by hand]\n\n"
    new = pat.sub(lambda m: m.group(1) + head + REGION[rid]() + "\n" + m.group(3), text)
    path.write_text(new)
    print(f"{path.name}: region regenerated ({len(new) - len(text):+d} chars)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, choices=["r1", "r2", "r3", "r4", "all"])
    a = ap.parse_args()
    for rid in (["r1", "r2", "r3", "r4"] if a.report == "all" else [a.report]):
        update(rid)


if __name__ == "__main__":
    main()
