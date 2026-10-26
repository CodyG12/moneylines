"""Permanent prediction log and result feedback (SQLite, standard library only).

Predictions are append-only: a stored probability is never updated. Results live in
their own table and are matched to predictions by `game_key` after the game. A
prediction counts toward live metrics only if it was made before tip-off (or before
the game date when the tip-off is unknown); later ones are kept but flagged `late`.

Model weights are never adjusted from individual results. Feedback flows through:
  reconcile()          record results, score each prediction, snapshot metrics, Elo ratings
  detect_degradation() recent log loss vs the model's walk-forward reference, with a
                       significance threshold, so one bad night does not trigger anything
  retrain_due()        scheduled (cadence) or degradation-triggered retraining decision
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import expected_calibration_error, per_game_brier, per_game_log_loss

DEFAULT_DB = os.environ.get("MONEYLINES_DB", "data/predictions.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_key TEXT NOT NULL, game_date TEXT NOT NULL, season INTEGER, home_team TEXT, away_team TEXT,
    tipoff_utc TEXT, prediction_ts TEXT NOT NULL, model_version TEXT NOT NULL,
    p_home REAL NOT NULL, p_away REAL NOT NULL,
    fair_home_american INTEGER, fair_away_american INTEGER, fair_home_decimal REAL, fair_away_decimal REAL,
    market_home_odds REAL, market_away_odds REAL, market_odds_ts TEXT, market_book TEXT, market_home_novig REAL,
    model_spread REAL, low_history INTEGER, features_asof TEXT, feature_snapshot TEXT, key_factors TEXT,
    UNIQUE (game_key, model_version, prediction_ts)
);
CREATE TABLE IF NOT EXISTS results (
    game_key TEXT PRIMARY KEY, home_score INTEGER, away_score INTEGER, home_win INTEGER, recorded_at TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id), game_key TEXT, home_win INTEGER,
    late INTEGER, log_loss REAL, brier REAL, abs_error REAL, correct INTEGER, evaluated_at TEXT
);
CREATE TABLE IF NOT EXISTS metrics_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, computed_at TEXT, model_version TEXT, window TEXT, n INTEGER,
    log_loss REAL, brier REAL, accuracy REAL, ece REAL, mean_predicted REAL, observed_rate REAL
);
CREATE TABLE IF NOT EXISTS elo_ratings (
    snapshot_ts TEXT, team TEXT, rating REAL, PRIMARY KEY (snapshot_ts, team)
);
CREATE TABLE IF NOT EXISTS training_runs (
    model_version TEXT PRIMARY KEY, trained_at TEXT, reason TEXT, train_start TEXT, train_end TEXT,
    n_rows INTEGER, metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT, model_version TEXT, message TEXT,
    details_json TEXT
);
"""

METRIC_WINDOWS = {"all": None, "last_100": 100, "last_300": 300}


def connect(path=DEFAULT_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    return con


def _none(v):
    if v is None or (np.ndim(v) == 0 and pd.isna(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def log_predictions(con: sqlite3.Connection, preds: pd.DataFrame) -> int:
    """Append predictions (production.predict_upcoming output). Re-logging the same
    (game, model version, timestamp) is ignored; nothing is overwritten."""
    rows = []
    for r in preds.to_dict("records"):
        rows.append(tuple(_none(v) for v in (
            r["game_key"], str(pd.Timestamp(r["date"]).date()), r["season"], r["home_team"], r["away_team"],
            r.get("tipoff_utc"), r["prediction_timestamp"], r["model_version"], r["p_home"], r["p_away"],
            r["fair_home_american"], r["fair_away_american"], r["fair_home_decimal"], r["fair_away_decimal"],
            r.get("home_odds"), r.get("away_odds"),
            str(r["odds_timestamp"]) if pd.notna(r.get("odds_timestamp")) else None,
            r.get("odds_book"), r.get("market_home_prob_novig"), r.get("model_spread"),
            int(bool(r.get("low_history"))), r.get("features_asof"), r.get("feature_snapshot"),
            r.get("key_factors"))))
    before = con.total_changes
    con.executemany("""INSERT OR IGNORE INTO predictions (game_key, game_date, season, home_team, away_team,
        tipoff_utc, prediction_ts, model_version, p_home, p_away, fair_home_american, fair_away_american,
        fair_home_decimal, fair_away_decimal, market_home_odds, market_away_odds, market_odds_ts, market_book,
        market_home_novig, model_spread, low_history, features_asof, feature_snapshot, key_factors)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    con.commit()
    return con.total_changes - before


def record_training_run(con, meta: dict, reason: str) -> None:
    t = meta["training"]
    con.execute("INSERT OR REPLACE INTO training_runs VALUES (?,?,?,?,?,?,?)",
                (meta["version"], meta["trained_at"], reason, t["start"], t["end"], t["n_rows"],
                 json.dumps(meta, default=str)))
    con.commit()


def _is_late(pred_ts: pd.Series, tipoff: pd.Series, game_date: pd.Series) -> pd.Series:
    ts = pd.to_datetime(pred_ts, utc=True, errors="coerce")
    tip = pd.to_datetime(tipoff, utc=True, errors="coerce")
    day_start = pd.to_datetime(game_date).dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    deadline = tip.where(tip.notna(), day_start + pd.Timedelta(hours=12))
    return ts >= deadline


def reconcile(con: sqlite3.Connection, games: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Record final results for predicted games and score every not-yet-scored prediction."""
    now = (now or pd.Timestamp.now(tz="UTC")).isoformat()
    played = games[games["played"]]
    keys = set(pd.read_sql("SELECT DISTINCT game_key FROM predictions", con)["game_key"])
    res = played[played["game_key"].isin(keys)]
    con.executemany("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?)",
                    [(r.game_key, int(r.home_score), int(r.away_score), int(r.home_win), now)
                     for r in res.itertuples()])
    pending = pd.read_sql("""SELECT p.id, p.game_key, p.p_home, p.prediction_ts, p.tipoff_utc, p.game_date,
                             r.home_win FROM predictions p JOIN results r USING (game_key)
                             LEFT JOIN evaluations e ON e.prediction_id = p.id WHERE e.prediction_id IS NULL""", con)
    if not pending.empty:
        y, p = pending["home_win"].to_numpy(float), pending["p_home"].to_numpy(float)
        late = _is_late(pending["prediction_ts"], pending["tipoff_utc"], pending["game_date"])
        con.executemany("INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?)", [
            (int(i), k, int(yy), int(lt), float(ll), float(br), float(abs(pp - yy)), int((pp >= 0.5) == (yy == 1)), now)
            for i, k, yy, pp, lt, ll, br in zip(pending["id"], pending["game_key"], y, p, late,
                                                per_game_log_loss(y, p), per_game_brier(y, p))])
    con.commit()
    return {"results_recorded": int(len(res)), "predictions_scored": int(len(pending)),
            "late_predictions": int(late.sum()) if not pending.empty else 0}


def official_evaluations(con: sqlite3.Connection, model_version: str | None = None) -> pd.DataFrame:
    """One scored prediction per (game, model version): the latest one made before tip-off."""
    q = """SELECT p.*, e.home_win, e.log_loss, e.brier, e.abs_error, e.correct, e.late, r.home_score, r.away_score
           FROM predictions p JOIN evaluations e ON e.prediction_id = p.id JOIN results r USING (game_key)
           WHERE e.late = 0"""
    df = pd.read_sql(q, con)
    if model_version:
        df = df[df["model_version"] == model_version]
    if df.empty:
        return df
    df = df.sort_values("prediction_ts").groupby(["game_key", "model_version"], as_index=False).tail(1)
    return df.sort_values(["game_date", "game_key"]).reset_index(drop=True)


def snapshot_metrics(con: sqlite3.Connection, now: pd.Timestamp | None = None) -> pd.DataFrame:
    now = (now or pd.Timestamp.now(tz="UTC")).isoformat()
    ev = official_evaluations(con)
    rows = []
    for version, sub in ev.groupby("model_version") if not ev.empty else []:
        for name, n in METRIC_WINDOWS.items():
            w = sub if n is None else sub.tail(n)
            if n is not None and len(sub) < n:
                continue
            y, p = w["home_win"].to_numpy(float), w["p_home"].to_numpy(float)
            rows.append({"computed_at": now, "model_version": version, "window": name, "n": len(w),
                         "log_loss": float(w["log_loss"].mean()), "brier": float(w["brier"].mean()),
                         "accuracy": float(w["correct"].mean()), "ece": expected_calibration_error(y, p),
                         "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_sql("metrics_history", con, if_exists="append", index=False)
    return out


def snapshot_elo(con: sqlite3.Connection, ratings: dict, now: pd.Timestamp | None = None) -> None:
    ts = (now or pd.Timestamp.now(tz="UTC")).isoformat()
    con.executemany("INSERT OR REPLACE INTO elo_ratings VALUES (?,?,?)",
                    [(ts, t, float(r)) for t, r in sorted(ratings.items())])
    con.commit()


def detect_degradation(con: sqlite3.Connection, model_version: str, reference: dict, min_n: int = 100,
                       window: int = 300, z: float = 2.0, now: pd.Timestamp | None = None) -> dict | None:
    """Flag when recent live log loss is significantly worse than the walk-forward reference.

    Test: mean log loss over the last `window` official predictions exceeds the reference mean
    by more than z standard errors (SE from the reference per-game SD). Needs `min_n` games.
    """
    ev = official_evaluations(con, model_version)
    if len(ev) < min_n or not reference.get("log_loss"):
        return None
    recent = ev.tail(window)
    sd = reference.get("log_loss_sd") or float(recent["log_loss"].std(ddof=1))
    se = sd / np.sqrt(len(recent))
    diff = float(recent["log_loss"].mean() - reference["log_loss"])
    flagged = diff > z * se
    out = {"model_version": model_version, "n": int(len(recent)), "recent_log_loss": float(recent["log_loss"].mean()),
           "reference_log_loss": reference["log_loss"], "diff": diff, "threshold": z * se, "degraded": bool(flagged)}
    if flagged:
        ts = (now or pd.Timestamp.now(tz="UTC")).isoformat()
        con.execute("INSERT INTO alerts (created_at, kind, model_version, message, details_json) VALUES (?,?,?,?,?)",
                    (ts, "degradation", model_version,
                     f"recent log loss {out['recent_log_loss']:.4f} vs reference {reference['log_loss']:.4f} "
                     f"over {len(recent)} games (> {z} SE)", json.dumps(out)))
        con.commit()
    return out


def retrain_due(con: sqlite3.Connection, cadence_days: int | None, degradation: dict | None,
                new_results: int, now: pd.Timestamp | None = None) -> str | None:
    """Reason to retrain now, or None. Scheduled retrains wait for new completed games."""
    now = now or pd.Timestamp.now(tz="UTC")
    if degradation and degradation.get("degraded"):
        return "degradation"
    if cadence_days is None:
        return None
    last = con.execute("SELECT MAX(trained_at) FROM training_runs").fetchone()[0]
    if last is None:
        return "no_training_run_recorded"
    if (now - pd.Timestamp(last)).days >= cadence_days and new_results > 0:
        return f"scheduled_every_{cadence_days}_days"
    return None
