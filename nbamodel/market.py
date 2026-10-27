"""Sportsbook market as a probability benchmark.

No-vig method (baseline): proportional normalisation. For decimal prices d_h, d_a the raw
implied probabilities are q = 1/d; their sum minus 1 is the bookmaker margin (overround); the
no-vig probabilities are q_h / (q_h + q_a) and q_a / (q_h + q_a). It ignores favourite-longshot
bias, so it slightly overstates longshots.

Alignment rule: a model probability is only ever compared with the price the book showed at or
before the moment that probability was produced (the prediction's own timestamp), never with a
later price. Live predictions store those prices in prediction_odds when they are made.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import odds_store
from .evaluation import compare_brier, compare_log_loss, full_score
from .odds import overround, remove_vig_two_way

MODEL_COLS = ("p_elo", "p_logistic", "p_logistic_raw", "p_gbm", "p_gbm_raw", "p_ensemble", "p_blend", "p_home")
# minutes before tip-off of the prediction (and so of the price it is compared with)
HORIZONS = {"<=30m": (0, 30), "30m-2h": (30, 120), "2h-12h": (120, 720), ">12h": (720, 10 ** 6)}


def novig_frame(home_decimal, away_decimal) -> pd.DataFrame:
    """Implied, margin and no-vig probabilities for a two-way market (documented above)."""
    h, a = np.asarray(home_decimal, float), np.asarray(away_decimal, float)
    nv_h, nv_a = remove_vig_two_way(h, a)
    return pd.DataFrame({"implied_home": 1 / h, "implied_away": 1 / a, "overround": overround(h, a),
                         "novig_home": nv_h, "novig_away": nv_a})


def attach_market_asof(preds: pd.DataFrame, snap: pd.DataFrame, book: str, max_age_minutes: float):
    """Fill each prediction's market columns from the latest pregame quote observed at or before
    its prediction_timestamp (no older than max_age). Returns (preds, all-book quotes used)."""
    out = preds.copy()
    for c in ("home_odds", "away_odds", "odds_timestamp", "odds_book", "market_home_prob_novig",
              "market_overround", "edge_home", "ev_home", "ev_away"):
        out[c] = np.nan if c not in ("odds_timestamp", "odds_book") else None
    if out.empty or snap.empty:
        return out, snap.iloc[0:0]
    asof = pd.Series(pd.to_datetime(out["prediction_timestamp"], utc=True).to_numpy(), index=out["game_key"])
    asof = asof[~asof.index.duplicated()]
    quotes = odds_store.quotes_asof(snap, asof, pd.Timedelta(minutes=max_age_minutes))
    chosen = odds_store.best_price(quotes) if book == "best" else quotes[quotes["book"] == book]
    if chosen.empty:
        return out, quotes
    m = out[["game_key"]].merge(chosen[["game_key", "home_decimal", "away_decimal", "observed_at", "book"]],
                                on="game_key", how="left")
    priced = m["home_decimal"].notna().to_numpy()
    out.loc[priced, "home_odds"] = m.loc[priced, "home_decimal"].to_numpy()
    out.loc[priced, "away_odds"] = m.loc[priced, "away_decimal"].to_numpy()
    out.loc[priced, "odds_timestamp"] = m.loc[priced, "observed_at"].map(lambda t: t.isoformat()).to_numpy()
    out.loc[priced, "odds_book"] = m.loc[priced, "book"].to_numpy()
    if priced.any():
        h, a = out.loc[priced, "home_odds"].astype(float), out.loc[priced, "away_odds"].astype(float)
        nv = novig_frame(h, a)
        out.loc[priced, "market_home_prob_novig"] = nv["novig_home"].to_numpy()
        out.loc[priced, "market_overround"] = nv["overround"].to_numpy()
        out.loc[priced, "edge_home"] = out.loc[priced, "p_home"] - nv["novig_home"].to_numpy()
        out.loc[priced, "ev_home"] = out.loc[priced, "p_home"] * h - 1
        out.loc[priced, "ev_away"] = out.loc[priced, "p_away"] * a - 1
    return out, quotes


# ------------------------------------------------------------------ benchmark on live predictions
def live_comparison_frame(con: sqlite3.Connection, book: str, model_version: str | None = None) -> pd.DataFrame:
    """Every scored pregame prediction with the `book` price recorded when it was made, plus its
    horizon (minutes before tip-off). One row per (game, model version, horizon): the latest
    prediction inside each horizon bucket."""
    q = """SELECT p.*, e.home_win, e.late, po.home_decimal AS mkt_home_decimal, po.away_decimal AS mkt_away_decimal,
                  po.observed_at AS mkt_observed_at, po.book AS mkt_book
           FROM predictions p JOIN evaluations e ON e.prediction_id = p.id
           JOIN prediction_odds po ON po.prediction_id = p.id
           WHERE e.late = 0"""
    df = pd.read_sql(q, con)
    if book != "best":
        df = df[df["mkt_book"] == book]
    else:  # best per side across books recorded for the same prediction
        if not df.empty:
            agg = df.groupby("id").agg(mkt_home_decimal=("mkt_home_decimal", "max"),
                                       mkt_away_decimal=("mkt_away_decimal", "max"),
                                       mkt_observed_at=("mkt_observed_at", "max"))
            df = df.drop(columns=["mkt_home_decimal", "mkt_away_decimal", "mkt_observed_at", "mkt_book"]
                         ).drop_duplicates("id").merge(agg, left_on="id", right_index=True).assign(mkt_book="best")
    if model_version:
        df = df[df["model_version"] == model_version]
    if df.empty:
        return df
    pred_ts = pd.to_datetime(df["prediction_ts"], utc=True)
    obs = pd.to_datetime(df["mkt_observed_at"], utc=True)
    df = df[obs <= pred_ts]  # alignment guard: never a price published after the prediction
    tip = pd.to_datetime(df["tipoff_utc"], utc=True)
    df = df.assign(minutes_before_tip=((tip - pd.to_datetime(df["prediction_ts"], utc=True)).dt.total_seconds() / 60))
    df["horizon"] = None
    for name, (lo, hi) in HORIZONS.items():
        df.loc[(df["minutes_before_tip"] > lo) & (df["minutes_before_tip"] <= hi), "horizon"] = name
    df = df[df["horizon"].notna()]
    df = df.sort_values("prediction_ts").groupby(["game_key", "model_version", "horizon"], as_index=False).tail(1)
    nv = novig_frame(df["mkt_home_decimal"], df["mkt_away_decimal"])
    df["p_market"] = nv["novig_home"].to_numpy()
    df["market_overround"] = nv["overround"].to_numpy()
    df["date"] = pd.to_datetime(df["game_date"])
    return df.reset_index(drop=True)


def benchmark(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, min_n: int = 2) -> dict:
    """Each model vs the no-vig market on identical rows (rows where every compared column exists)."""
    if df is None or df.empty:
        return {"n": 0, "status": "no_aligned_games"}
    cols = [c for c in MODEL_COLS if c in df.columns and df[c].notna().any()]
    cols = [c for c in cols if df[c].notna().all()] or cols[:1]
    d = df.dropna(subset=cols + ["p_market", "home_win"])
    if len(d) < min_n:
        return {"n": int(len(d)), "status": "insufficient_games"}
    y = d["home_win"].to_numpy(float)
    res = {"n": int(len(d)), "n_dates": int(d["date"].nunique()), "status": "ok",
           "date_range": [str(d["date"].min().date()), str(d["date"].max().date())],
           "mean_overround": float(d["market_overround"].mean()) if "market_overround" in d else None,
           "market": full_score(y, d["p_market"], d["date"], n_boot, seed), "models": {}}
    for c in cols:
        s = full_score(y, d[c], d["date"], n_boot, seed)
        s["log_loss_minus_market"] = compare_log_loss(y, d[c], d["p_market"], d["date"], n_boot, seed)
        s["brier_minus_market"] = compare_brier(y, d[c], d["p_market"], d["date"], n_boot, seed)
        res["models"][c] = s
    res["preliminary"] = len(d) < 500
    return res


def live_benchmark(con: sqlite3.Connection, book: str, n_boot: int = 1000, seed: int = 0) -> dict:
    df = live_comparison_frame(con, book)
    out = {"book": book, "by_horizon": {}}
    for h in HORIZONS:
        sub = df[df["horizon"] == h] if not df.empty else df
        out["by_horizon"][h] = benchmark(sub, n_boot, seed)
    return out


def snapshot_market_metrics(con: sqlite3.Connection, book: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Append market log loss / Brier (decision horizon) to metrics_history as model 'market:<book>'."""
    df = live_comparison_frame(con, book)
    if df.empty:
        return pd.DataFrame()
    d = df[df["horizon"] == "30m-2h"].drop_duplicates("game_key")
    if d.empty:
        return pd.DataFrame()
    from .evaluation import expected_calibration_error, per_game_brier, per_game_log_loss
    y, p = d["home_win"].to_numpy(float), d["p_market"].to_numpy(float)
    row = pd.DataFrame([{"computed_at": (now or pd.Timestamp.now(tz="UTC")).isoformat(),
                         "model_version": f"market:{book}", "window": "all", "n": len(d),
                         "log_loss": float(per_game_log_loss(y, p).mean()), "brier": float(per_game_brier(y, p).mean()),
                         "accuracy": float(((p >= 0.5) == (y == 1)).mean()), "ece": expected_calibration_error(y, p),
                         "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}])
    row.to_sql("metrics_history", con, if_exists="append", index=False)
    return row


def upcoming_market_view(con: sqlite3.Connection, book: str) -> pd.DataFrame:
    """Latest unresolved prediction per game with the price recorded alongside it."""
    p = pd.read_sql("""SELECT p.*, po.book AS mkt_book, po.home_decimal AS mkt_home_decimal,
                              po.away_decimal AS mkt_away_decimal, po.home_american AS mkt_home_american,
                              po.away_american AS mkt_away_american, po.observed_at AS mkt_observed_at
                       FROM predictions p LEFT JOIN results r USING (game_key)
                       LEFT JOIN prediction_odds po ON po.prediction_id = p.id AND po.book = ?
                       WHERE r.game_key IS NULL""", con, params=[book])
    if p.empty:
        return p
    p = p.sort_values("prediction_ts").groupby(["game_key", "model_version"], as_index=False).tail(1)
    priced = p["mkt_home_decimal"].notna()
    for c in ("p_market", "edge_home", "edge_away", "exp_roi_home", "exp_roi_away"):
        p[c] = np.nan
    if priced.any():
        nv = novig_frame(p.loc[priced, "mkt_home_decimal"], p.loc[priced, "mkt_away_decimal"])
        p.loc[priced, "p_market"] = nv["novig_home"].to_numpy()
        p.loc[priced, "edge_home"] = p.loc[priced, "p_home"] - p.loc[priced, "p_market"]
        p.loc[priced, "edge_away"] = -p.loc[priced, "edge_home"]
        p.loc[priced, "exp_roi_home"] = p.loc[priced, "p_home"] * p.loc[priced, "mkt_home_decimal"] - 1
        p.loc[priced, "exp_roi_away"] = p.loc[priced, "p_away"] * p.loc[priced, "mkt_away_decimal"] - 1
    return p.sort_values(["game_date", "tipoff_utc"]).reset_index(drop=True)
