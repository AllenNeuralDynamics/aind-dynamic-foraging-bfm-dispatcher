#!/usr/bin/env python
"""Result 2: per mouse, does HB beat the same mouse's own classical RL fit on held-out sessions?

Offline: reads hb_runs.json, hb_per_subject_heldout.csv, hb_posterior_summary.json, ml_params/ (frozen) and
study 01's committed rl_baseline.json (used only to check that the RL pooled values reproduce).

A mouse's HB likelihood and its RL likelihood are scored on the same held-out trials (asserted: the HB
table's n_trials equals the ML table's num_eval_trials for every mouse). Pairs are therefore mice, and the
paired difference is HB minus RL, so a negative value means HB predicts that mouse worse.

Outputs: per_mouse.json (with _meta), per_mouse.csv, fig_per_mouse.png.
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
from scipy import stats

HERE = Path(__file__).parent
STUDY01 = HERE.parent.parent / "01-gru-scaling-law"
sys.path.insert(0, str(HERE.parent.parent / "util"))
from _meta import build_meta  # noqa: E402

HEADLINE = ("one-stage-ladder@20260903-154819", "one-stage-ladder@20260903-154828")
N_SUBJECTS = {0.016: 10, 0.049: 29, 0.163: 99, 0.489: 300, 1.0: 614}
RL = ("hattori", "bari", "ctt")
NOT_MIXED_RHAT = 1.1
SCATTER_D = 29           # the headline rung whose group- and subject-level R-hat are closest to the gate
C = {"HB": "#0072B2", "RL": "#7f7f7f"}


def pooled(lik: pd.Series, n: pd.Series) -> float:
    return float(np.exp((n * np.log(lik)).sum() / n.sum()))


def build() -> dict:
    runs = pd.DataFrame(json.loads((HERE / "hb_runs.json").read_text())["runs"])
    post = pd.DataFrame(json.loads((HERE / "hb_posterior_summary.json").read_text())["runs"])[["wandb_run_id", "max_rhat"]]
    hb_tab = pd.read_csv(HERE / "hb_per_subject_heldout.csv", dtype={"subject_id": str})
    rl_json = json.loads((STUDY01 / "analysis" / "rl_baseline.json").read_text())
    ml = {k: pd.read_csv(HERE / "ml_params" / f"ml_{k}_heldout.csv", dtype={"subject_id": str}) for k in RL}
    for k, t in ml.items():   # the frozen per-mouse RL fits must reproduce study 01's pooled value
        agg = pooled(t.eval_likelihood, t.num_eval_trials)
        want = rl_json["models"][k]["pooled_likelihood_trial_weighted"]
        assert abs(agg - want) < 1e-6, f"{k}: frozen table pools to {agg} but rl_baseline.json says {want}"
    head = runs[(runs.state == "finished") & runs.group.isin(HEADLINE)].copy()
    head = head.drop(columns=["max_rhat", "min_ess_bulk"])   # Stan-only columns in hb_runs; NumPyro values come from the posterior summary
    head["D"] = head.subject_ratio.round(3).map(N_SUBJECTS).astype(int)
    head = head.merge(post, on="wandb_run_id", how="left").sort_values("D")
    rows, pairs = [], []
    for _, r in head.iterrows():
        h = hb_tab[hb_tab.wandb_run_id == r.wandb_run_id]
        assert len(h) == 149
        for k in RL:
            m = h.merge(ml[k][["subject_id", "eval_likelihood", "num_eval_trials"]], on="subject_id", validate="one_to_one")
            assert len(m) == 149 and (m.n_trials == m.num_eval_trials).all(), "HB and RL were not scored on the same trials"
            d = m.likelihood - m.eval_likelihood
            w = stats.wilcoxon(d)
            rows.append({"D": int(r.D), "wandb_run_id": r.wandb_run_id, "rl_model": k, "n_mice": len(m),
                         "hb_pooled": pooled(m.likelihood, m.n_trials), "rl_pooled": pooled(m.eval_likelihood, m.n_trials),
                         "median_diff": float(d.median()), "mean_diff": float(d.mean()),
                         "n_hb_worse": int((d < 0).sum()), "frac_hb_worse": float((d < 0).mean()),
                         "wilcoxon_p": float(w.pvalue), "max_rhat": float(r.max_rhat), "not_mixed": bool(r.max_rhat > NOT_MIXED_RHAT)})
            pairs.append(m.assign(D=int(r.D), rl_model=k, diff=d, wandb_run_id=r.wandb_run_id)[
                ["D", "rl_model", "subject_id", "n_trials", "likelihood", "eval_likelihood", "diff", "wandb_run_id"]])
    return {"summary": pd.DataFrame(rows), "pairs": pd.concat(pairs), "ml": ml}


def figure(d: dict, path: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    s, p = d["summary"], d["pairs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.38})
    sc = p[(p.D == SCATTER_D) & (p.rl_model == "hattori")]
    a.scatter(sc.eval_likelihood, sc.likelihood, s=9, color=C["HB"], alpha=0.75, lw=0)
    lo, hi = float(min(sc.eval_likelihood.min(), sc.likelihood.min())) - 0.02, float(max(sc.eval_likelihood.max(), sc.likelihood.max())) + 0.02
    a.plot([lo, hi], [lo, hi], color="#4a4a4a", lw=0.8)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    row = s[(s.D == SCATTER_D) & (s.rl_model == "hattori")].iloc[0]
    a.set_xlabel("Same mouse fit on its own (Hattori2019)")
    a.set_ylabel("Hierarchical model")
    a.set_title("Most mice are predicted worse by the pooled fit", loc="left", fontsize=8)
    a.text(0.04, 0.96, f"below the line: {int(row.n_hb_worse)} of {int(row.n_mice)} mice", transform=a.transAxes, fontsize=7, va="top")
    a.text(0.0, -0.22, f"D = {SCATTER_D} training mice", transform=a.transAxes, fontsize=6.5, color="#4a4a4a", va="top")
    # b: paired difference per rung, vs the same-model per-mouse fit
    ds = sorted(s.D.unique()); rng = np.random.default_rng(0); flags = []
    for i, D in enumerate(ds):
        pr = p[(p.D == D) & (p.rl_model == "hattori")]
        nm = bool(s[(s.D == D) & (s.rl_model == "hattori")].not_mixed.iloc[0])
        b.scatter(i + rng.uniform(-0.22, 0.22, len(pr)), pr["diff"], s=5, color=C["HB"], alpha=0.35, lw=0, zorder=2)
        b.plot([i - 0.3, i + 0.3], [pr["diff"].median()] * 2, color="#1a1a1a", lw=1.6, zorder=4)
        flags.append((i, nm))
    b.axhline(0, color="#4a4a4a", lw=0.8, zorder=1)
    b.set_xticks(range(len(ds))); b.set_xticklabels([str(x) for x in ds]); b.set_xlim(-0.6, len(ds) - 0.4)
    dd = p[p.rl_model == "hattori"]["diff"]
    pad = 0.15 * (dd.max() - dd.min())
    b.set_ylim(dd.min() - pad, dd.max() + 3 * pad)
    for i, nm in flags:
        if nm:
            b.text(i, b.get_ylim()[1] - 0.4 * pad, "not mixed", ha="center", va="top", fontsize=6.5, color="#4a4a4a")
    b.set_xlabel("Training mice (D)")
    b.set_ylabel("Per-mouse likelihood, HB minus same-mouse fit")
    b.set_title("The gap to each mouse's own fit is present at every rung", loc="left", fontsize=8)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = build()
    s = d["summary"]
    meta = build_meta("analysis/per_mouse.py", sorted(HEADLINE), HERE.parent)
    (HERE / "per_mouse.json").write_text(json.dumps({"_meta": meta, "cohort": {"n_mice": 149, "n_trials": 1013172},
        "pairing": "same held-out mice and the same scored trials (asserted); difference = HB likelihood minus the mouse's own RL fit",
        "not_mixed_rhat_threshold": NOT_MIXED_RHAT, "summary": s.to_dict("records")}, indent=2) + "\n")
    d["pairs"].to_csv(HERE / "per_mouse.csv", index=False)
    figure(d, HERE / "fig_per_mouse.png")
    print(s[["D", "rl_model", "hb_pooled", "rl_pooled", "median_diff", "n_hb_worse", "wilcoxon_p", "not_mixed"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
