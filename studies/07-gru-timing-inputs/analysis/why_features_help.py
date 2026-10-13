#!/usr/bin/env python
"""Two logistic regressions: why previous-trial RT and lick counts predict the next choice.

This is the behavioural *mechanism* behind r1's result that adding previous-trial
reaction time and lick counts improves held-out choice prediction. It does NOT
read the model at all — it asks, in the data, whether each feature carries
information about whether the mouse repeats or switches its choice on the next
trial, and whether that information is reward-gated.

Two regressions on responded trials (per-session previous-trial shift), stay =
(this choice == previous choice):

    Reg 1  stay ~ prev_reward * z(prev log RT)
    Reg 2  stay ~ prev_reward * z(prev total licks)

The reward interaction is the point: it separates a reward-INDEPENDENT signal
(engagement) from a reward-GATED one (post-outcome abandonment). Verdict
(HELD-OUT mice with computable RT, 125 of 157 -- the other 32 are early old-schema
sessions lacking the reaction_time field; n~1.57M; slope/+1SD, per-mouse
cluster-bootstrap 95% CI):
  RT    unrewarded -0.29 [-0.34,-0.25] (z~-13), rewarded -0.23 [-0.29,-0.16]
        (z~-6.9) -> slower responding precedes switching REGARDLESS of outcome
        (a global engagement state).
  licks unrewarded -0.28 [-0.33,-0.22] (z~-9.3), rewarded +0.21 [+0.11,+0.31]
        (z~+3.9) -> after NO reward more licking predicts abandoning that side;
        after reward it mildly confirms it (REWARD-GATED). So the two blocks carry
        different, choice-relevant state that (prev choice, prev reward) alone do
        not -- which is why they help. Naive (trial-independent) SEs are
        anti-conservative by ~an order of magnitude here; mouse-clustering is
        essential.

    python analysis/why_features_help.py     # needs DB access + statsmodels + scikit-learn + matplotlib

Outputs fig_why_features_help.png + why_features_help.csv at the study root.
Cohort = the HELD-OUT mice (157), pinned in provenance/heldout_subjects.txt -- the
population the study-07 GRU was evaluated on but never trained on, so the mechanism
is shown where the held-out likelihood gain it explains is scored. The coupling is
population-general: the same trend holds (slightly noisier) on the train mice, which
is precisely why the model carries RT/lick information to unseen animals.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent

SNAPSHOT = "20260603"
LICK_WINDOW_S = 2.0
BLUE, ORANGE = "#2166ac", "#d6604d"


def _duckdb():
    import duckdb
    ext = os.path.join(tempfile.gettempdir(), "duckdb_ext")
    os.makedirs(ext, exist_ok=True)
    duckdb.execute(f"SET extension_directory='{ext}'")
    try:
        duckdb.execute("LOAD httpfs;")
    except Exception:
        duckdb.execute("INSTALL httpfs;")
        duckdb.execute("LOAD httpfs;")
    duckdb.execute("SET s3_region='us-west-2'; SET s3_endpoint='s3.us-west-2.amazonaws.com';")
    return duckdb


def build_sequence(db, duckdb, subjects, allowed_sessions):
    """Responded-trial frame with per-session previous-trial features + stay.

    Scoped to ``allowed_sessions``. ``main`` passes ALL sessions of the cohort
    subjects (all-stage), matching the population the GRU trained on
    (``data.mature_only=false``); pass a mature/curricula subset to match the r1
    probe instead.
    """
    tsrc = db.read_trials(subjects, snapshot=SNAPSHOT)
    esrc = db.read_events(subjects, snapshot=SNAPSHOT)
    feat = duckdb.sql(f"""
      WITH t AS (SELECT tr.session_id, tr.trial, tr.animal_response, tr.earned_reward,
          COALESCE(tr.goCue_start_time_in_session, tr.goCue_start_time) AS gocue,
          COALESCE(tr.reaction_time, tr.choice_time_in_session - tr.goCue_start_time_in_session) AS rt
        FROM {tsrc} tr),
      lk AS (SELECT session_id, event, timestamps FROM {esrc}
             WHERE event IN ('left_lick_time','right_lick_time'))
      SELECT t.session_id, t.trial, ANY_VALUE(t.animal_response) AS animal_response,
        ANY_VALUE(t.earned_reward) AS earned_reward, ANY_VALUE(t.rt) AS rt,
        COUNT(*) FILTER (WHERE lk.event IN ('left_lick_time','right_lick_time')) AS n_licks
      FROM t LEFT JOIN lk ON lk.session_id=t.session_id
         AND lk.timestamps>=t.gocue AND lk.timestamps<t.gocue+{float(LICK_WINDOW_S)}
      GROUP BY t.session_id, t.trial
    """).df()
    feat = feat[feat["session_id"].isin(allowed_sessions)]
    seq = feat[feat["animal_response"] < 2].sort_values(["session_id", "trial"]).copy()
    seq["subject_id"] = seq["session_id"].astype(str).str.split("_").str[0]
    seq["reward"] = seq["earned_reward"].astype(float)
    g = seq.groupby("session_id", sort=False)
    seq["prev_choice"] = g["animal_response"].shift(1)
    seq["prev_reward"] = g["reward"].shift(1)
    seq["prev_logrt"] = np.log(np.clip(g["rt"].shift(1), 1e-3, 10))
    seq["prev_licks"] = g["n_licks"].shift(1)
    seq = seq.dropna(subset=["prev_choice", "prev_reward", "prev_logrt", "prev_licks"]).copy()
    seq["stay"] = (seq["animal_response"].values == seq["prev_choice"].values).astype(int)
    for col in ("prev_logrt", "prev_licks"):
        seq["z_" + col] = (seq[col] - seq[col].mean()) / seq[col].std()
    return seq


def fit(seq):
    import statsmodels.formula.api as smf
    m_rt = smf.logit("stay ~ prev_reward * z_prev_logrt", data=seq).fit(disp=0)
    m_lk = smf.logit("stay ~ prev_reward * z_prev_licks", data=seq).fit(disp=0)
    return m_rt, m_lk


def cluster_bootstrap_bins(seq, feat, rew, *, nq=6, nboot=2000, seed=0):
    """Per-mouse cluster-bootstrap 95% CI on the binned P(stay).

    Trials within a mouse are not independent, so a naive binomial SEM understates
    uncertainty several-fold; resampling the mice (not the trials) with replacement
    gives an honest interval. Fast: aggregate per-subject x bin once, then each
    resample is a matrix product over the ~60 subjects.
    """
    rng = np.random.default_rng(seed)
    zc = "z_" + feat
    mask = seq.prev_reward.values == rew
    z = seq[zc].values[mask]; st = seq.stay.values[mask]
    sb = seq["subject_id"].astype(str).values[mask]
    edges = np.quantile(z, np.linspace(0, 1, nq + 1)); edges[0] -= 1e-9; edges[-1] += 1e-9
    b = np.clip(np.digitize(z, edges) - 1, 0, nq - 1)
    su = np.unique(sb); si = {s: i for i, s in enumerate(su)}
    scode = np.array([si[s] for s in sb])
    K = np.zeros((len(su), nq)); N = np.zeros((len(su), nq))
    np.add.at(K, (scode, b), st); np.add.at(N, (scode, b), 1.0)
    xpos = np.array([z[b == k].mean() if (b == k).any() else np.nan for k in range(nq)])
    boot = np.empty((nboot, nq))
    for it in range(nboot):
        m = np.bincount(rng.integers(0, len(su), len(su)), minlength=len(su)).astype(float)
        num = m @ K; den = m @ N
        boot[it] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return dict(x=xpos, mean=np.nanmean(boot, 0),
                lo=np.nanpercentile(boot, 2.5, 0), hi=np.nanpercentile(boot, 97.5, 0))


def cluster_bootstrap_slopes(seq, feat, *, nboot=300, seed=0):
    """Per-mouse cluster-bootstrap 95% CI on the per-reward-state logit slope/+1SD.

    Refits the logistic on each mouse-resample (sklearn, fast). Naive model SEs
    assume independent trials and are anti-conservative here.
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed + 1)
    subs = seq["subject_id"].astype(str).values
    uniq = np.unique(subs); S = len(uniq)
    sub_idx = {s: np.where(subs == s)[0] for s in uniq}
    zc = "z_" + feat

    def design(idx):
        z = seq[zc].values[idx]; pr = seq.prev_reward.values[idx]
        return np.column_stack([pr, z, pr * z]), seq.stay.values[idx]
    X0, y0 = design(np.arange(len(seq)))
    c0 = LogisticRegression(max_iter=300, C=1e9).fit(X0, y0).coef_[0]
    un = np.empty(nboot); rw = np.empty(nboot)
    for it in range(nboot):
        idx = np.concatenate([sub_idx[s] for s in uniq[rng.integers(0, S, S)]])
        X, y = design(idx)
        c = LogisticRegression(max_iter=300, C=1e9).fit(X, y).coef_[0]
        un[it] = c[1]; rw[it] = c[1] + c[2]
    return dict(un=c0[1], un_ci=(np.percentile(un, 2.5), np.percentile(un, 97.5)), un_z=c0[1] / un.std(),
                rw=c0[1] + c0[2], rw_ci=(np.percentile(rw, 2.5), np.percentile(rw, 97.5)),
                rw_z=(c0[1] + c0[2]) / rw.std())


def _panel(ax, seq, m, feat, xlabel, title, nq=6):
    zc = f"z_{feat}"
    for rew, c, lab in [(0.0, BLUE, "prev unrewarded"), (1.0, ORANGE, "prev rewarded")]:
        d = cluster_bootstrap_bins(seq, feat, rew, nq=nq)
        ok = np.isfinite(d["x"]) & np.isfinite(d["mean"])
        x = d["x"][ok]; mn = d["mean"][ok]
        yerr = np.vstack([mn - d["lo"][ok], d["hi"][ok] - mn])
        ax.errorbar(x, mn, yerr=yerr, fmt="o", color=c, ms=6, capsize=3, lw=0,
                    elinewidth=1.3, ecolor=c, zorder=3)
        xs = np.linspace(seq[zc].quantile(.01), seq[zc].quantile(.99), 100)
        ax.plot(xs, m.predict(pd.DataFrame({"prev_reward": rew, zc: xs})),
                color=c, lw=2.2, zorder=2, label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel("P(repeat same choice)")
    ax.set_title(title); ax.legend(frameon=False, fontsize=10, loc="lower left")


def figure(seq, m_rt, m_lk):
    import matplotlib.pyplot as plt
    try:
        import sys
        sys.path.insert(0, str(STUDY.parent / "util"))
        from plot_style import apply_presentation_style
        apply_presentation_style()
    except Exception:
        pass

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    _panel(axs[0], seq, m_rt, "prev_logrt", "previous-trial reaction time  (z of log RT)",
           "Reg 1 \u2014 slow responses precede switching,\nin BOTH reward states (engagement)")
    _panel(axs[1], seq, m_lk, "prev_licks", "previous-trial total lick count  (z)",
           "Reg 2 \u2014 after NO reward more licking precedes a switch;\nafter reward a mild stay (reward-gated)")
    fig.suptitle("Why previous-trial RT and lick counts predict the next choice "
                 f"(logistic; {seq['subject_id'].nunique()} held-out mice, n={len(seq):,} trials; "
                 "error bars = per-mouse cluster bootstrap 95% CI)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(STUDY / "fig_why_features_help.png", dpi=160, bbox_inches="tight")


def main():
    import aind_dynamic_foraging_database as db
    duckdb = _duckdb()
    db.use_snapshot(SNAPSHOT)
    sdb = db.session_db()
    session_df = duckdb.sql(
        f"SELECT _session_id, subject_id, session_date, task, current_stage_actual "
        f"FROM read_parquet('{sdb}')").df()
    # HELD-OUT cohort: the mice the GRU was evaluated on but never trained on, so the
    # behavioural mechanism is shown on the same population as the held-out likelihood
    # gain it explains. Pinned in provenance/heldout_subjects.txt (157 mice = eligible
    # cohort minus the union of every study-07 GRU training cohort). ALL-STAGE sessions.
    prov = HERE / "provenance" / "heldout_subjects.txt"
    subjects = [ln.strip() for ln in prov.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    sd = session_df.copy(); sd["subject_id"] = sd["subject_id"].astype(str)
    allowed = set(sd.loc[sd["subject_id"].isin(subjects), "_session_id"])
    seq = build_sequence(db, duckdb, subjects, allowed)
    m_rt, m_lk = fit(seq)
    # Per-mouse cluster-bootstrap CIs on the per-reward-state slope/+1SD are the
    # headline uncertainty (naive model SEs assume independent trials).
    rows = []
    for feat, lab in [("prev_logrt", "RT (log)"), ("prev_licks", "total licks")]:
        s = cluster_bootstrap_slopes(seq, feat)
        rows.append({"regression": lab, "quantity": "slope/+1SD, prev UNREWARDED",
                     "estimate": s["un"], "ci_lo": s["un_ci"][0], "ci_hi": s["un_ci"][1],
                     "clustered_z": s["un_z"]})
        rows.append({"regression": lab, "quantity": "slope/+1SD, prev REWARDED",
                     "estimate": s["rw"], "ci_lo": s["rw_ci"][0], "ci_hi": s["rw_ci"][1],
                     "clustered_z": s["rw_z"]})
    pd.DataFrame(rows).to_csv(STUDY / "why_features_help.csv", index=False)
    figure(seq, m_rt, m_lk)
    print(f"why_features_help: n={len(seq)} trials, {seq['subject_id'].nunique()} mice with "
          f"computable RT (of {len(subjects)} held-out pinned); wrote fig + csv")


if __name__ == "__main__":
    main()
