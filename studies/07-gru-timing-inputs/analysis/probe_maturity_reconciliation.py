#!/usr/bin/env python
"""Reconcile the logistic probe's maturity scope to the GRU's training population.

Context. The committed probe ``code/analysis/calibrate_timing_features.py`` in the
wrapper restricts to mature (STAGE_FINAL/GRADUATED) sessions in the three
curricula, whereas the study-07 GRU runs use ``data.mature_only: false``
(all-stage trials, set by the dispatcher's ``mice_snapshot_scaling.yaml`` that the
sweeps compose). This one-off asks whether that maturity mismatch changes the
probe's incremental Δ, and whether it explains the probe-vs-GRU baseline offset.

It re-runs the probe's EXACT logistic fit (copied from calibrate_timing_features.py)
on the same seeded 60-subject train cohort, varying only the session filter:

    A  mature + curricula        (the committed probe scope; reproduces
                                  timing_calibration.csv byte-for-byte)
    B  all-stage + curricula     (drop the maturity filter only)
    C  all-stage, all-task       (subject-scoped, closest to the GRU loader)

Verdict (all on the identical 60-subject cohort, snapshot 20260603):
    Δ(+RT&licks): A +0.00764 -> B +0.00834 -> C +0.00853  (unchanged, slightly
      larger all-stage -> the effect does NOT depend on the maturity scope)
    baseline:     A 0.7433 -> B 0.7294 ~= GRU OFF baseline 0.7285  (the offset was
      the maturity scope; all-stage probe and GRU measure the same thing)

    python analysis/probe_maturity_reconciliation.py   # needs WANDB-free DB access

Requires the AIND foraging DB (duckdb + aind_dynamic_foraging_database) and scipy.
Outputs fig_probe_maturity_reconciliation.png + probe_maturity_reconciliation.csv
at the study root.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent

CURRICULA = ["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"]
MATURE_STAGES = ("STAGE_FINAL", "GRADUATED")
SNAPSHOT = "20260603"
N_SUBJECTS = 60
LICK_WINDOW_S = 2.0
SEED = 0


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


# --- exact copy of calibrate_timing_features.select_cohort_subjects ----------
def select_cohort_subjects(session_df, *, min_sessions=10, heldout_every_n=5):
    df = session_df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    n_per = df.groupby("subject_id").size()
    kept = n_per[n_per >= min_sessions].index
    df_kept = df[df["subject_id"].isin(kept)]
    df_task = df_kept[df_kept["task"].notna()]
    tc = (df_task.groupby(["subject_id", "task"]).size().rename("n").reset_index()
          .sort_values(["subject_id", "n", "task"], ascending=[True, False, True]))
    subj_cur = tc.groupby("subject_id", sort=False).first()["task"].to_dict()
    train, heldout = [], []
    for cur in CURRICULA:
        subs = [s for s, c in subj_cur.items() if c == cur]
        ranked = sorted(subs, key=lambda s: (-int(n_per[s]), s))
        heldout += [s for i, s in enumerate(ranked, 1) if i % heldout_every_n == 0]
        train += [s for i, s in enumerate(ranked, 1) if i % heldout_every_n != 0]
    return {"train": sorted(set(train)), "heldout": sorted(set(heldout))}


# --- exact copy of the probe's fit + prep -----------------------------------
def _fit_eval(design, y, tr, ev):
    from scipy.optimize import minimize
    M = np.column_stack([design[k] for k in design])

    def nll(w, Mx, yx):
        z = Mx @ w
        return np.logaddexp(0, z).sum() - (yx * z).sum()
    w = minimize(nll, np.zeros(M.shape[1]), args=(M[tr], y[tr]), method="L-BFGS-B").x
    ze = M[ev] @ w
    ll = -(np.logaddexp(0, ze).sum() - (y[ev] * ze).sum())
    return float(np.exp(ll / ev.sum()))


def run_calibration(seq, n_lags=3):
    ch1 = seq["p1_animal_response"].values * 2 - 1
    y = seq["animal_response"].values
    sids = seq["session_id"].astype("category").cat.codes.values
    tr, ev = sids % 2 == 0, sids % 2 == 1

    def z(a):
        a = np.asarray(a, float)
        s = a[tr].std()
        return (a - a[tr].mean()) / (s if s > 0 else 1.0)
    base = {"bias": np.ones(len(seq))}
    for L in range(1, n_lags + 1):
        ch = seq[f"p{L}_animal_response"].values * 2 - 1
        base[f"ch{L}"] = ch
        base[f"rewch{L}"] = ch * seq[f"p{L}_rewarded"].values
    Lc, Rc = seq["p1_n_lick_left"].values, seq["p1_n_lick_right"].values
    lrt = z(np.log(np.clip(seq["p1_rt"].values, 1e-3, 10)))
    enc_rt = {"lrt": lrt, "lrt_x_ch": lrt * ch1}
    enc_lr = {"nL": z(np.log1p(Lc)), "nL_x_ch": z(np.log1p(Lc)) * ch1,
              "nR": z(np.log1p(Rc)), "nR_x_ch": z(np.log1p(Rc)) * ch1}
    sets = {"baseline": {}, "rt": enc_rt, "lick": enc_lr, "both": {**enc_rt, **enc_lr}}
    base_lik = _fit_eval(base, y, tr, ev)
    out = {"baseline": base_lik}
    for name, extra in sets.items():
        if name == "baseline":
            continue
        out[name] = _fit_eval({**base, **extra}, y, tr, ev) - base_lik
    return out, int(ev.sum())


def prepare_sequence(feat, n_lags=3):
    seq = feat[feat["animal_response"] < 2].sort_values(["session_id", "trial"]).copy()
    seq["rewarded"] = seq["earned_reward"].astype(float)
    g = seq.groupby("session_id", sort=False)
    seq["p1_rt"] = g["reaction_time"].shift(1)
    seq["p1_n_lick_left"] = g["n_lick_left"].shift(1)
    seq["p1_n_lick_right"] = g["n_lick_right"].shift(1)
    for L in range(1, n_lags + 1):
        seq[f"p{L}_animal_response"] = g["animal_response"].shift(L)
        seq[f"p{L}_rewarded"] = g["rewarded"].shift(L)
    need = ["p1_rt", "p1_n_lick_left", "p1_n_lick_right",
            f"p{n_lags}_animal_response", f"p{n_lags}_rewarded"]
    return seq.dropna(subset=need).copy()


def compute_timing(db, duckdb, subjects):
    tsrc = db.read_trials(subjects, snapshot=SNAPSHOT)
    esrc = db.read_events(subjects, snapshot=SNAPSHOT)
    timing = duckdb.sql(f"""
      WITH t AS (SELECT tr.session_id AS ses_idx, tr.trial,
          COALESCE(tr.goCue_start_time_in_session, tr.goCue_start_time) AS gocue,
          COALESCE(tr.reaction_time, tr.choice_time_in_session - tr.goCue_start_time_in_session) AS reaction_time
        FROM {tsrc} tr),
      lk AS (SELECT session_id, event, timestamps FROM {esrc}
             WHERE event IN ('left_lick_time','right_lick_time'))
      SELECT t.ses_idx, t.trial, ANY_VALUE(t.reaction_time) AS reaction_time,
        COUNT(*) FILTER (WHERE lk.event='left_lick_time')  AS n_lick_left,
        COUNT(*) FILTER (WHERE lk.event='right_lick_time') AS n_lick_right
      FROM t LEFT JOIN lk ON lk.session_id=t.ses_idx
         AND lk.timestamps>=t.gocue AND lk.timestamps<t.gocue+{float(LICK_WINDOW_S)}
      GROUP BY t.ses_idx, t.trial
    """).df()
    timing["trial"] = timing["trial"].astype(int)
    return timing.rename(columns={"ses_idx": "session_id"})


def gru_reference():
    """GRU study-07 held-out reference (D>=100) from the committed producer CSV."""
    csv = STUDY / "analysis" / "timing_scaling.csv"
    g = pd.read_csv(csv)
    big = g[g.D >= 100]
    return float(big.OFF_mean.mean()), float(big.net.mean())


def main():
    import aind_dynamic_foraging_database as db
    duckdb = _duckdb()
    db.use_snapshot(SNAPSHOT)
    sdb = db.session_db()
    session_df = duckdb.sql(
        f"SELECT _session_id, subject_id, session_date, task, current_stage_actual "
        f"FROM read_parquet('{sdb}')").df()
    cohort = select_cohort_subjects(session_df)["train"]
    rng = np.random.default_rng(SEED)
    subjects = sorted(rng.choice(sorted(cohort), min(N_SUBJECTS, len(cohort)),
                                 replace=False).tolist())
    timing = compute_timing(db, duckdb, subjects)

    sel_all = db.select_sessions(
        subjects=subjects,
        columns=["_session_id", "subject_id", "task", "current_stage_actual"])
    variants = {
        "A mature+curricula": sel_all[sel_all["current_stage_actual"].isin(MATURE_STAGES)
                                      & sel_all["task"].isin(CURRICULA)],
        "B all-stage+curricula": sel_all[sel_all["task"].isin(CURRICULA)],
        "C all-stage all-task": sel_all,
    }
    rows = []
    for name, sel in variants.items():
        trials = db.fetch_trials(sel, columns=["animal_response", "earned_reward"])
        feat = trials.merge(timing, on=["session_id", "trial"], how="left")
        feat["n_lick_left"] = feat["n_lick_left"].fillna(0)
        feat["n_lick_right"] = feat["n_lick_right"].fillna(0)
        seq = prepare_sequence(feat)
        out, nev = run_calibration(seq)
        rows.append({"variant": name, "n_sessions": int(sel.shape[0]),
                     "n_model_trials": int(len(seq)), "n_eval_trials": nev,
                     "baseline": out["baseline"], "d_rt": out["rt"],
                     "d_lick": out["lick"], "d_both": out["both"]})
    df = pd.DataFrame(rows)
    df.to_csv(STUDY / "probe_maturity_reconciliation.csv", index=False)

    gru_off, gru_net = gru_reference()
    _figure(df, gru_off, gru_net)
    print(df.to_string(index=False))
    print(f"GRU (D>=100): OFF baseline={gru_off:.5f}  net={gru_net:.5f}")


def _figure(df, gru_off, gru_net):
    import matplotlib.pyplot as plt
    try:
        import sys
        sys.path.insert(0, str(STUDY.parent / "util"))
        from plot_style import apply_presentation_style
        apply_presentation_style()
    except Exception:
        pass
    BLUE, ORANGE, GREEN, PURPLE, GREY = "#2166ac", "#d6604d", "#4a7c59", "#762a83", "#9e9e9e"
    short = ["mature\n+curricula\n(committed)", "all-stage\n+curricula", "all-stage\nall-task\n(GRU-like)"]
    x = np.arange(len(df))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    ax.bar(x, df.baseline, color=[GREY, BLUE, BLUE], width=.6, zorder=3)
    for xi, v in zip(x, df.baseline):
        ax.text(xi, v + 0.0006, f"{v:.4f}", ha="center", fontsize=10)
    ax.axhline(gru_off, color=ORANGE, lw=2, ls="--", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylim(0.70, 0.755)
    ax.set_ylabel("baseline norm. likelihood")
    ax.set_title(f"Matching maturity aligns the baseline:\nall-stage probe \u2248 GRU OFF baseline "
                 f"({gru_off:.4f}, dashed)")
    ax.set_xlim(-0.6, len(df) + 0.4)
    ax = axs[1]
    w = 0.26
    ax.bar(x - w, df.d_rt * 1000, w, color=GREEN, label="+ RT only", zorder=3)
    ax.bar(x, df.d_lick * 1000, w, color=PURPLE, label="+ licks only", zorder=3)
    ax.bar(x + w, df.d_both * 1000, w, color=BLUE, label="+ both", zorder=3)
    ax.axhline(gru_net * 1000, color=ORANGE, lw=2, ls="--", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_ylabel("\u0394 over baseline (\u00d71000)")
    ax.set_title(f"The probe \u0394 is unchanged (slightly larger)\nall-stage; GRU realizes ~90% "
                 f"(+{gru_net * 1000:.2f}, dashed)")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_ylim(0, 9.6)
    fig.suptitle("Reconciling the logistic probe to the GRU's all-stage population "
                 "(60-subject train cohort, snapshot 20260603)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(STUDY / "fig_probe_maturity_reconciliation.png", dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
