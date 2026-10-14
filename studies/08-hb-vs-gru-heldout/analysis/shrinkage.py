#!/usr/bin/env python
"""Result 3: how much does HB pull each mouse's parameters toward the population, parameter by parameter?

Offline: reads hb_subject_posteriors.csv (HB subject-level posterior means, natural units),
ml_params/ml_hattori_train.csv (the same mice, fit one at a time by maximum likelihood) and
hb_posterior_summary.json (for the mixing flag). HB and the per-mouse fit are the same model family.

For each headline rung and parameter, the slope of the HB posterior mean on that mouse's own fit:
slope 1 means HB reproduces the individual estimate, slope < 1 means individual differences are
compressed toward the mean. Primary statistics are the Theil-Sen slope and the Spearman correlation.
Ordinary least squares is computed only for contrast: one mouse with an inverse temperature of 27
(the per-mouse fitter's bound is wider than HB's beta_max = 10) drags an OLS slope to 0.13 where
Theil-Sen gives 1.10.

Excluded from the primary slope, and drawn open: ML fits pinned at a rate bound (exactly 0 or 1, an
optimiser artifact, not an estimate) and ML inverse temperatures above HB's beta_max (outside HB's support).
The slope over ALL mice is reported alongside.

Axis limits follow the project convention: rates [0, 1], inverse temperature [0, beta_max], bias_l +-0.5
(a reporting convention for an unbounded parameter, not a limit).

Outputs: shrinkage.json (with _meta), shrinkage.csv, fig_shrinkage.png.
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
sys.path.insert(0, str(HERE.parent.parent / "util"))
from _meta import build_meta  # noqa: E402

PARAMS = ("learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen", "softmax_inverse_temperature", "bias_l")
ML_COL = {"learn_rate_rew": "learn_rate_rew", "learn_rate_unrew": "learn_rate_unrew", "forget_rate_unchosen": "forget_rate_unchosen",
          "softmax_inverse_temperature": "softmax_inverse_temperature", "bias_l": "biasL"}
LABEL = {"learn_rate_rew": "learning rate, rewarded", "learn_rate_unrew": "learning rate, unrewarded", "forget_rate_unchosen": "forgetting rate",
         "softmax_inverse_temperature": "inverse temperature", "bias_l": "left bias"}
BETA_MAX = 10.0
XLIM = {"learn_rate_rew": (0, 1), "learn_rate_unrew": (0, 1), "forget_rate_unchosen": (0, 1),
        "softmax_inverse_temperature": (0, BETA_MAX), "bias_l": (-0.5, 0.5)}
NOT_MIXED_RHAT = 1.1
SCATTER_D = 29
MIN_N = 8
COLOR = {"learn_rate_rew": "#0072B2", "learn_rate_unrew": "#E69F00", "forget_rate_unchosen": "#009E73",
         "softmax_inverse_temperature": "#CC79A7", "bias_l": "#555555"}


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sp = pd.read_csv(HERE / "hb_subject_posteriors.csv", dtype={"subject_id": str})
    ml = pd.read_csv(HERE / "ml_params" / "ml_hattori_train.csv", dtype={"subject_id": str})
    post = json.loads((HERE / "hb_posterior_summary.json").read_text())["runs"]
    rhat = {r["wandb_run_id"]: r["max_rhat"] for r in post}
    return sp, ml, rhat


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    sp, ml, rhat = load()
    long = ml.melt(id_vars="subject_id", value_vars=list(ML_COL.values()), var_name="ml_col", value_name="ml")
    long["param"] = long.ml_col.map({v: k for k, v in ML_COL.items()})
    df = sp.merge(long[["subject_id", "param", "ml"]], on=["subject_id", "param"], how="left", validate="many_to_one")
    assert df.ml.notna().all(), "an HB training mouse has no ML fit"
    rate = df.param.isin(PARAMS[:3])
    df["pinned"] = rate & ((df.ml <= 1e-9) | (df.ml >= 1 - 1e-9))
    df["off_support"] = (df.param == "softmax_inverse_temperature") & (df.ml > BETA_MAX)
    df["excluded"] = df.pinned | df.off_support
    df["max_rhat"] = df.wandb_run_id.map(rhat)
    df["not_mixed"] = df.max_rhat > NOT_MIXED_RHAT
    rows = []
    for (rid, D, p), g in df.groupby(["wandb_run_id", "D", "param"]):
        use = g[~g.excluded]
        row = {"wandb_run_id": rid, "D": int(D), "param": p, "n_all": len(g), "n_used": len(use), "n_pinned": int(g.pinned.sum()),
               "n_off_support": int(g.off_support.sum()), "max_rhat": float(g.max_rhat.iloc[0]), "not_mixed": bool(g.not_mixed.iloc[0])}
        if len(use) >= MIN_N:
            ts = stats.theilslopes(use.hb_mean, use.ml)
            row.update(theilsen_slope=float(ts[0]), spearman=float(stats.spearmanr(use.ml, use.hb_mean)[0]))
        else:
            row.update(theilsen_slope=np.nan, spearman=np.nan)
        row["theilsen_slope_all"] = float(stats.theilslopes(g.hb_mean, g.ml)[0]) if len(g) >= MIN_N else np.nan
        row["ols_slope_all"] = float(stats.linregress(g.ml, g.hb_mean).slope) if len(g) >= MIN_N else np.nan
        row["sd_ratio_used"] = float(use.hb_mean.std() / use.ml.std()) if len(use) >= MIN_N else np.nan
        rows.append(row)
    return df, pd.DataFrame(rows).sort_values(["param", "D"])


def figure(df: pd.DataFrame, s: pd.DataFrame, path: Path) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8,
                         "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 5.0), gridspec_kw={"wspace": 0.42, "hspace": 0.55})
    sc = df[df.D == SCATTER_D]
    for ax, p in zip(axs.flat[:5], PARAMS):
        g, lim = sc[sc.param == p], XLIM[p]
        ok, ex = g[~g.excluded], g[g.excluded]
        ax.plot(lim, lim, color="#4a4a4a", lw=0.8, zorder=1)
        ax.scatter(ok.ml, ok.hb_mean, s=14, color=COLOR[p], lw=0, zorder=3)
        inside = ex[(ex.ml >= lim[0]) & (ex.ml <= lim[1])]
        ax.scatter(inside.ml, inside.hb_mean, s=16, facecolors="white", edgecolors=COLOR[p], linewidths=1.0, zorder=3)
        ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
        row = s[(s.D == SCATTER_D) & (s.param == p)].iloc[0]
        ax.set_title(LABEL[p], loc="left", fontsize=8)
        ax.text(0.04, 0.96, f"slope {row.theilsen_slope:.2f}", transform=ax.transAxes, fontsize=7, va="top")
        off = int(((g.ml < lim[0]) | (g.ml > lim[1])).sum())
        if off:
            ax.text(0.96, 0.04, f"{off} off scale", transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom", color="#4a4a4a")
        ax.set_xlabel("Mouse fit alone"); ax.set_ylabel("HB posterior mean")
    ax = axs.flat[5]
    ax.axhline(1, color="#4a4a4a", lw=0.8, zorder=1)
    ends = []
    for p in PARAMS:
        g = s[s.param == p].sort_values("D")
        ax.plot(g.D, g.theilsen_slope, "-", color=COLOR[p], lw=1.2, zorder=3)
        for _, r in g.iterrows():
            ax.plot(r.D, r.theilsen_slope, "o", ms=4.2, mfc="white" if r.not_mixed else COLOR[p], mec=COLOR[p], mew=1.0, zorder=4)
        ends.append([float(g.theilsen_slope.iloc[-1]), p])
    # direct labels at the right end, spread so none overlap (minimum gap 0.09 in slope units)
    ends.sort()
    ys = [e[0] for e in ends]
    for _ in range(50):
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < 0.09:
                mid = (ys[i] + ys[i - 1]) / 2
                ys[i - 1], ys[i] = mid - 0.0451, mid + 0.0451
    short = {"learn_rate_rew": "rewarded lr", "learn_rate_unrew": "unrewarded lr", "forget_rate_unchosen": "forgetting",
             "softmax_inverse_temperature": "inv. temp.", "bias_l": "left bias"}
    for (_, p), y in zip(ends, ys):
        ax.text(614 * 1.3, y, short[p], fontsize=6.3, color=COLOR[p], va="center")
    ax.set_xscale("log"); ax.set_xticks([10, 30, 100, 300, 614]); ax.set_xticklabels(["10", "30", "100", "300", "614"]); ax.minorticks_off()
    ax.set_xlim(8, 614 * 4.2); ax.set_ylim(0, 1.45)
    ax.set_xlabel("Training mice (D)"); ax.set_ylabel("Slope (1 = no shrinkage)")
    ax.set_title("Compression differs by parameter, no monotone trend in D", loc="left", fontsize=8)
    fig.text(0.01, -0.01, f"Scatters: D = {SCATTER_D} training mice; open markers = mouse fit pinned at a bound. "
             "Right panel: open markers = chains not mixed (R-hat > 1.1).", fontsize=6.5, color="#4a4a4a", ha="left", va="top")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df, s = build()
    post_groups = sorted({"one-stage-ladder@20260903-154819", "one-stage-ladder@20260903-154828"})
    meta = build_meta("analysis/shrinkage.py", post_groups, HERE.parent)
    (HERE / "shrinkage.json").write_text(json.dumps({"_meta": meta, "exclusion": "ML fits pinned at a rate bound (exactly 0 or 1) or with inverse temperature above beta_max=10",
        "min_n_for_slope": MIN_N, "not_mixed_rhat_threshold": NOT_MIXED_RHAT, "rows": s.to_dict("records")}, indent=2, default=float) + "\n")
    s.to_csv(HERE / "shrinkage.csv", index=False)
    figure(df, s, HERE / "fig_shrinkage.png")
    pv = s.pivot(index="param", columns="D", values="theilsen_slope").round(3).reindex(PARAMS)
    print(pv.to_string())
    print("\nn used / n all at D=614:", {r.param: f"{r.n_used}/{r.n_all}" for r in s[s.D == 614].itertuples()})
    print("n used at D=10:", {r.param: r.n_used for r in s[s.D == 10].itertuples()})


if __name__ == "__main__":
    main()
