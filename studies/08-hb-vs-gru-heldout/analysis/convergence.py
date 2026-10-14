#!/usr/bin/env python
"""Result 4: do the HB fits mix, what does each sampler setting buy, and what does Stan cost against NumPyro?

Offline: reads hb_posterior_summary.json (R-hat / ESS / divergences recomputed from every saved NumPyro
posterior) and hb_runs.json (W&B summary values, including the Stan run's own diagnostics).

Like-for-like rule for the Stan comparison: HBStanTrainer computes R-hat and bulk ESS on the five POPULATION
MEANS only (hb_stan_trainer.py, _diagnostics). The NumPyro counterpart is therefore the population_mean site,
not the minimum over all sites (which includes per-mouse sites and is lower). Both fits use 4 chains x 2000
draws after 2000 warmup on the same 29 mice. They ran on different hardware (NumPyro on one GPU, Stan on 8 CPUs),
so wall time compares the two as the project ran them and is not a statement about the frameworks alone.

Outputs: convergence.json (with _meta), convergence.csv, fig_convergence.png.
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
sys.path.insert(0, str(HERE.parent.parent / "util"))
from _meta import build_meta  # noqa: E402

LADDER = "one-stage-ladder@20260903-154819"
FULL = "one-stage-ladder@20260903-154828"
GATES = {"max_rhat": 1.01, "min_ess_bulk": 400}
# (group, label) of the D~29 runs that differ only in sampler setting; the diagnostic probe is excluded because it
# repeats the ladder setting exactly (identical numbers, same seed).
STAN_GROUP = "stan-arm@20260905-020928"   # the counted Stan launch; its one finished run is the D=29 fit
D29_SETTINGS = [("one-stage-ladder@20260903-052244", "500 / 500"), (LADDER, "2000 / 500"),
                ("one-stage-ladder@20260903-154825", "4000 / 500"), ("one-stage-ladder@20260904-082628", "2000 / 2000")]
C = {"group": "#0072B2", "subject": "#D55E00"}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    post = json.loads((HERE / "hb_posterior_summary.json").read_text())["runs"]
    rows = []
    for r in post:
        rows.append({"wandb_run_id": r["wandb_run_id"], "group": r["group"], "D": r["n_subjects"], "num_warmup": r["num_warmup"],
                     "num_samples": r["num_samples"], "divergences": r["divergences"],
                     "group_max_rhat": r["group_level"]["max_rhat"], "group_min_ess": r["group_level"]["min_ess_bulk"],
                     "subject_max_rhat": r["subject_level"]["max_rhat"], "subject_min_ess": r["subject_level"]["min_ess_bulk"],
                     "popmean_max_rhat": r["group_level"]["per_site"]["population_mean"]["max_rhat"],
                     "popmean_min_ess": r["group_level"]["per_site"]["population_mean"]["min_ess_bulk"],
                     "total_draws": r["chains"] * r["draws_per_chain"], "passes_all_gates": r["passes_all_gates"]})
    runs = pd.DataFrame(json.loads((HERE / "hb_runs.json").read_text())["runs"])
    return pd.DataFrame(rows), runs


def stan_vs_numpyro(post: pd.DataFrame, runs: pd.DataFrame) -> dict:
    npy_rows = post[(post.group == "one-stage-ladder@20260904-082628") & (post.D == 29)]
    assert len(npy_rows) == 1, f"expected exactly one NumPyro D=29 run in the draws group, found {len(npy_rows)}"
    npy = npy_rows.iloc[0]
    npy_fit = float(runs[runs.wandb_run_id == npy.wandb_run_id].fit_seconds.iloc[0])
    # The pinned counted group, selected explicitly: groups with counts=False are crash history and must never
    # feed a reported number, even if one of their runs were later to finish.
    st_rows = runs[(runs.group == STAN_GROUP) & (runs.kind == "stan") & (runs.state == "finished") & runs.counts]
    assert len(st_rows) == 1, f"expected exactly one finished counted Stan run in {STAN_GROUP}, found {len(st_rows)}"
    st = st_rows.iloc[0]
    assert int(st.n_subjects) == 29 and int(st.n_draws) == int(npy.total_draws) == 8000
    # Stan's R-hat and ESS come from recompute_stan_diagnostics.py, which recomputes them from the saved draws with
    # PyStan's chain axis. The values in hb_runs.json are what the trainer logged to W&B from a chain-major reshape that
    # mixed the chains; they are kept there as recorded and must not feed a reported number.
    sd = json.loads((HERE / "stan_d29_diagnostics.json").read_text())
    assert sd["wandb_run_id"] == st.wandb_run_id and sd["n_draws"] == int(st.n_draws)
    st_rhat, st_ess = sd["population_mean"]["max_rhat"], sd["population_mean"]["min_ess_bulk"]
    out = {"cohort_D": 29, "settings": "4 chains x (2000 warmup + 2000 draws), 8000 retained draws each",
           "diagnostic_sites": "population_mean (the five population means); Stan's own definition",
           "numpyro": {"wandb_run_id": npy.wandb_run_id, "fit_seconds": npy_fit, "divergences": int(npy.divergences),
                       "max_rhat": float(npy.popmean_max_rhat), "min_ess_bulk": float(npy.popmean_min_ess),
                       "ess_per_draw": float(npy.popmean_min_ess / npy.total_draws), "ess_per_second": float(npy.popmean_min_ess / npy_fit),
                       "hardware": "one GPU (Beaker)"},
           "stan": {"wandb_run_id": st.wandb_run_id, "fit_seconds": float(st.fit_seconds), "divergences": int(st.divergences),
                    "max_rhat": float(st_rhat), "min_ess_bulk": float(st_ess),
                    "ess_per_draw": float(st_ess / st.n_draws), "ess_per_second": float(st_ess / st.fit_seconds),
                    "hardware": "8 CPU cores, one chain per process (HPC)",
                    "diagnostics_source": "analysis/stan_d29_diagnostics.json, recomputed from the saved draws",
                    "wandb_logged_WRONG": {"max_rhat": float(st.max_rhat), "min_ess_bulk": float(st.min_ess_bulk),
                                           "why": "chain-major reshape of a draw-major axis mixed the chains"}}}
    out["ratios"] = {"stan_ess_per_draw_over_numpyro": out["stan"]["ess_per_draw"] / out["numpyro"]["ess_per_draw"],
                     "stan_wall_over_numpyro": out["stan"]["fit_seconds"] / out["numpyro"]["fit_seconds"],
                     "stan_ess_per_second_over_numpyro": out["stan"]["ess_per_second"] / out["numpyro"]["ess_per_second"]}
    return out


def figure(post: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"wspace": 0.45})
    h = post[post.group.isin([LADDER, FULL])].sort_values("D")
    a.axhline(GATES["max_rhat"], color="#4a4a4a", lw=0.8, ls=(0, (4, 2)), zorder=1)
    a.plot(h.D, h.group_max_rhat, "-o", color=C["group"], ms=4, lw=1.2, zorder=3)
    a.plot(h.D, h.subject_max_rhat, "-s", color=C["subject"], ms=4, lw=1.2, zorder=3)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks([10, 30, 100, 300, 614]); a.set_xticklabels(["10", "30", "100", "300", "614"]); a.minorticks_off()
    a.set_yticks([1.0, 1.1, 1.3, 1.6, 1.9]); a.set_yticklabels(["1.0", "1.1", "1.3", "1.6", "1.9"])
    a.set_xlim(8, 614 * 3.4); a.set_ylim(0.995, 2.1)
    a.text(614 * 1.3, h.group_max_rhat.iloc[-1], "group-level", fontsize=6.5, color=C["group"], va="center")
    a.text(614 * 1.3, h.subject_max_rhat.iloc[-1], "per-mouse", fontsize=6.5, color=C["subject"], va="center")
    a.text(614 * 1.3, GATES["max_rhat"] * 1.012, "gate 1.01", fontsize=6.5, color="#4a4a4a", va="bottom")
    a.set_xlabel("Training mice (D)"); a.set_ylabel("Worst R-hat")
    a.set_title("Per-mouse chains do not mix at D = 99 and above", loc="left", fontsize=8)
    # b: D~29, min bulk ESS over group and per-mouse sites by sampler setting
    s = []
    for g, lab in D29_SETTINGS:
        r = post[(post.group == g) & (post.D == 29)].iloc[0]
        s.append((lab, min(r.group_min_ess, r.subject_min_ess), int(r.divergences)))
    xs = np.arange(len(s))
    b.axhline(GATES["min_ess_bulk"], color="#4a4a4a", lw=0.8, ls=(0, (4, 2)), zorder=1)
    b.plot(xs, [v[1] for v in s], "o", color="#1a1a1a", ms=5, zorder=3)
    for x, (_, ess, dv) in zip(xs, s):
        b.text(x, ess * 0.80, f"{dv} div.", ha="center", va="top", fontsize=6.5, color="#4a4a4a")
    b.set_yscale("log"); b.set_yticks([100, 200, 400]); b.set_yticklabels(["100", "200", "400"]); b.minorticks_off()
    b.set_ylim(60, 700)
    b.set_xticks(xs); b.set_xticklabels([v[0] for v in s]); b.set_xlim(-0.4, len(s) - 0.6)
    b.text(-0.38, GATES["min_ess_bulk"] * 1.05, "gate 400", fontsize=6.5, color="#4a4a4a", ha="left", va="bottom")
    b.set_xlabel("Warmup / draws per chain"); b.set_ylabel("Lowest bulk ESS, any site")
    b.set_title("Only more draws approach the ESS gate", loc="left", fontsize=8)
    fig.text(0.01, -0.02, "Left: D = 10 to 614 at warmup 2000, 500 draws. Right: D = 29, 4 chains; div. = divergent transitions.",
             fontsize=6.5, color="#4a4a4a", ha="left", va="top")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    post, runs = load()
    cmp_ = stan_vs_numpyro(post, runs)
    meta = build_meta("analysis/convergence.py", sorted(set(post.group) | {"stan-arm@20260905-020928"}), HERE.parent)
    (HERE / "convergence.json").write_text(json.dumps({"_meta": meta, "gates": GATES, "numpyro_runs": post.to_dict("records"),
                                                       "stan_vs_numpyro_D29": cmp_}, indent=2, default=float) + "\n")
    post.sort_values(["D", "num_samples", "num_warmup"]).to_csv(HERE / "convergence.csv", index=False)
    figure(post, HERE / "fig_convergence.png")
    print(post[post.group.isin([LADDER, FULL])].sort_values("D")[["D", "group_max_rhat", "subject_max_rhat", "group_min_ess", "subject_min_ess"]].round(3).to_string(index=False))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cmp_["ratios"].items()}))
    print("numpyro popmean ESS/draw", round(cmp_["numpyro"]["ess_per_draw"], 4), "| stan", round(cmp_["stan"]["ess_per_draw"], 4))


if __name__ == "__main__":
    main()
