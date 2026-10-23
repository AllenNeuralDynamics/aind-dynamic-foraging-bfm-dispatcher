#!/usr/bin/env python
"""Result 1: held-out likelihood of HB vs per-mouse classical RL vs the GRU, on the same 149 mice.

Offline: reads only frozen or committed files, never W&B.
  * hb_runs.json                         frozen HB runs (extract_hb_runs.py)
  * hb_posterior_summary.json            convergence from the saved posteriors (summarize_hb_posteriors.py)
  * ../01-gru-scaling-law/scaling_results.csv        GRU, study 01 (committed)
  * ../01-gru-scaling-law/analysis/rl_baseline.json  classical RL, study 01 (committed)

Outputs: three_arm.json (with _meta), three_arm.csv, fig_three_arm.png.

Headline HB rungs are the two launches that share one setting (warmup 2000, 500 draws): the ladder
group (D~10/29/99/300) and the full-cohort group (D=614). The probe launches (warmup 4000, 2000 draws,
diagnostic) are in the CSV but not on the curve, so every plotted HB point has one sampler setting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
STUDY01 = HERE.parent.parent / "01-gru-scaling-law"
sys.path.insert(0, str(HERE.parent.parent / "util"))
from _meta import build_meta  # noqa: E402

HEADLINE = ("one-stage-ladder@20260903-154819", "one-stage-ladder@20260903-154828")
N_SUBJECTS = {0.016: 10, 0.049: 29, 0.163: 99, 0.489: 300, 1.0: 614}
GRU_POOL = {10: 10, 29: 29, 30: 29, 99: 99, 101: 99, 300: 300, 301: 300, 614: 614}   # study 01 labels a size two ways
NOT_MIXED_RHAT = 1.1          # conventional "clearly not converged"; every rung also fails the 1.01 gate
GRU_VARIANT = "v2-sc-active"
RL = {"ctt": "compare-to-threshold", "bari": "Bari", "hattori": "Hattori2019 (same model as HB, one fit per mouse)"}

C = {"HB": "#0072B2", "GRU": "#1a1a1a", "RL": "#7f7f7f"}


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    hb = pd.DataFrame(json.loads((HERE / "hb_runs.json").read_text())["runs"])
    post = pd.DataFrame(json.loads((HERE / "hb_posterior_summary.json").read_text())["runs"])
    gru = pd.read_csv(STUDY01 / "scaling_results.csv")
    rl = json.loads((STUDY01 / "analysis" / "rl_baseline.json").read_text())
    return hb, post, gru, rl


def build() -> dict:
    hb, post, gru, rl = load()
    fin = hb[(hb.state == "finished") & (hb.kind == "numpyro")].copy()
    fin = fin.drop(columns=["max_rhat", "min_ess_bulk"])   # Stan-only columns in hb_runs; NumPyro values come from the posterior summary
    fin["D"] = fin.subject_ratio.round(3).map(N_SUBJECTS).astype(int)
    fin = fin.merge(post[["wandb_run_id", "max_rhat", "min_ess_bulk", "divergences"]].rename(columns={"divergences": "divergences_posterior"}),
                    on="wandb_run_id", how="left")
    assert fin.max_rhat.notna().all(), "a finished HB run has no posterior summary"
    assert (fin.divergences == fin.divergences_posterior).all(), "W&B and posterior divergence counts disagree"
    assert (fin.num_test_subjects == 149).all(), "HB was not scored on the 149-mouse held-out cohort"
    fin["headline"] = fin.group.isin(HEADLINE)
    fin["not_mixed"] = fin.max_rhat > NOT_MIXED_RHAT

    g = gru[gru.variant == GRU_VARIANT].copy()
    g["D_pooled"] = g.D.map(GRU_POOL)
    gru_by_d = g.groupby("D_pooled").heldout_likelihood.agg(["mean", "std", "count"]).reset_index().rename(columns={"D_pooled": "D"})

    rl_vals = {k: float(rl["models"][k]["pooled_likelihood_trial_weighted"]) for k in RL}
    for k in RL:   # same held-out cohort and the same trial-weighted aggregation as the HB metric
        assert rl["models"][k]["n_subjects"] == 149 and rl["models"][k]["n_trials"] == 1013172

    head = fin[fin.headline].sort_values("D")
    rows = []
    for _, r in head.iterrows():
        gm = gru_by_d.loc[gru_by_d.D == r.D, "mean"]
        rows.append({"D": int(r.D), "wandb_run_id": r.wandb_run_id, "hb_matched": float(r.heldout_matched),
                     "gru_v2_mean": float(gm.iloc[0]) if len(gm) else None,
                     "gap_gru_minus_hb": float(gm.iloc[0] - r.heldout_matched) if len(gm) else None,
                     "max_rhat": float(r.max_rhat), "min_ess_bulk": float(r.min_ess_bulk), "divergences": int(r.divergences),
                     "not_mixed": bool(r.not_mixed)})
    return {"hb_all": fin, "hb_head": head, "gru": g, "gru_by_d": gru_by_d, "rl": rl_vals, "rows": rows}


def figure(d: dict, path: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.set_xscale("log")
    xmax = 614
    # classical RL: fixed per-mouse fits, so horizontal references (drawn behind everything)
    for k, y in d["rl"].items():
        ax.hlines(y, 8, xmax * 1.08, color=C["RL"], lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.text(xmax * 1.16, y, RL[k].split(" (")[0], va="center", ha="left", fontsize=7, color="#4a4a4a")
    g, gb = d["gru"], d["gru_by_d"]
    ax.scatter(g.D, g.heldout_likelihood, s=9, color=C["GRU"], alpha=0.35, lw=0, zorder=3)
    ax.plot(gb.D, gb["mean"], "-o", color=C["GRU"], ms=3.5, lw=1.5, zorder=4)
    ax.text(xmax * 1.16, float(gb["mean"].iloc[-1]), "GRU", va="center", ha="left", fontsize=7, color=C["GRU"])
    h = d["hb_head"]
    ax.plot(h.D, h.heldout_matched, "-", color=C["HB"], lw=1.2, zorder=5)
    for _, r in h.iterrows():
        ax.plot(r.D, r.heldout_matched, "o", ms=5.5, mfc="white" if r.not_mixed else C["HB"], mec=C["HB"], mew=1.3, zorder=6)
    ax.text(xmax * 1.16, float(h.heldout_matched.iloc[-1]), "HB", va="center", ha="left", fontsize=7, color=C["HB"])
    ax.set_xticks([10, 30, 100, 300, 614]); ax.set_xticklabels(["10", "30", "100", "300", "614"])
    ax.minorticks_off()
    ax.set_xlim(8, xmax * 3.6)
    lo = min(h.heldout_matched.min(), min(d["rl"].values())) - 0.002
    ax.set_ylim(lo, float(gb["mean"].max()) + 0.002)
    ax.set_xlabel("Training mice (D)")
    ax.set_ylabel("Held-out choice likelihood per trial")
    ax.set_title("The hierarchical model predicts held-out mice worse than every other arm", loc="left", fontsize=8)
    ax.text(0.01, 0.02, "open marker: chains not mixed (R-hat > 1.1)", transform=ax.transAxes, fontsize=6.5, color="#4a4a4a", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    d = build()
    out = {"_meta": build_meta("analysis/three_arm.py", sorted(set(d["hb_all"].group)), HERE.parent),
           "cohort": {"n_subjects": 149, "n_trials": 1013172},
           "aggregation": "pooled trial-weighted per-trial choice likelihood (exp of mean log-likelihood over all held-out trials)",
           "rl": d["rl"], "gru_variant": GRU_VARIANT,
           "gru_by_D": d["gru_by_d"].to_dict("records"), "hb_headline": d["rows"],
           "not_mixed_rhat_threshold": NOT_MIXED_RHAT, "gru_size_pooling": GRU_POOL}
    (HERE / "three_arm.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    keep = ["group", "wandb_run_id", "D", "num_warmup", "num_samples", "headline", "heldout_matched", "heldout_k0", "heldout_k8",
            "divergences", "max_rhat", "min_ess_bulk", "not_mixed", "fit_seconds"]
    d["hb_all"].sort_values(["D", "num_samples", "num_warmup"])[keep].to_csv(HERE / "three_arm.csv", index=False)
    figure(d, HERE / "fig_three_arm.png")
    for r in d["rows"]:
        print(f"D={r['D']:>3} hb={r['hb_matched']:.5f} gru={r['gru_v2_mean']:.5f} gap={r['gap_gru_minus_hb']:+.5f} rhat={r['max_rhat']:.3f} not_mixed={r['not_mixed']}")
    print("RL:", {k: round(v, 5) for k, v in d["rl"].items()})


if __name__ == "__main__":
    main()
