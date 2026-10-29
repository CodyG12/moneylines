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

# Phase 3. Every table here is append-only: rows are inserted once and never updated, so a stored
# price, probability or decision always shows what was known when it was written.
PHASE3_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL, source TEXT NOT NULL, provider_event_id TEXT NOT NULL,
    game_key TEXT, match_status TEXT NOT NULL, flipped INTEGER NOT NULL DEFAULT 0,
    home_team TEXT, away_team TEXT, provider_home_name TEXT, provider_away_name TEXT,
    book TEXT NOT NULL, market TEXT NOT NULL DEFAULT 'h2h',
    home_american REAL, away_american REAL, home_decimal REAL NOT NULL, away_decimal REAL NOT NULL,
    commence_time TEXT, tipoff_utc TEXT,
    observed_at TEXT NOT NULL, provider_ts TEXT, retrieved_at TEXT NOT NULL,
    is_pregame INTEGER NOT NULL, run_id INTEGER,
    UNIQUE (provider, provider_event_id, book, market, observed_at)
);
CREATE INDEX IF NOT EXISTS odds_snapshots_game ON odds_snapshots (game_key, book, observed_at);
CREATE TABLE IF NOT EXISTS odds_collection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, finished_at TEXT, provider TEXT, mode TEXT,
    status TEXT, reason TEXT, n_events INTEGER, n_rows INTEGER, n_inserted INTEGER, n_duplicates INTEGER,
    n_invalid INTEGER, n_unmatched INTEGER, n_live INTEGER, credits_used INTEGER, credits_remaining INTEGER,
    error TEXT, targets_json TEXT
);
CREATE TABLE IF NOT EXISTS prediction_odds (
    prediction_id INTEGER NOT NULL REFERENCES predictions(id), book TEXT NOT NULL, snapshot_id INTEGER,
    observed_at TEXT NOT NULL, home_decimal REAL NOT NULL, away_decimal REAL NOT NULL,
    home_american REAL, away_american REAL, PRIMARY KEY (prediction_id, book)
);
CREATE TABLE IF NOT EXISTS paper_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy TEXT NOT NULL, prediction_id INTEGER NOT NULL,
    game_key TEXT NOT NULL, game_date TEXT, tipoff_utc TEXT, model_version TEXT NOT NULL, p_col TEXT,
    decided_at TEXT NOT NULL, decision TEXT NOT NULL, book TEXT, side TEXT, p_side REAL,
    decimal_odds REAL, american_odds REAL, odds_observed_at TEXT, market_p_side_novig REAL, edge REAL,
    est_ev REAL, stake_flat REAL, stake_kelly REAL, bankroll_at_decision REAL, details_json TEXT,
    UNIQUE (strategy, game_key)
);
CREATE TABLE IF NOT EXISTS paper_results (
    decision_id INTEGER PRIMARY KEY REFERENCES paper_decisions(id), home_win INTEGER, won INTEGER,
    pnl_flat REAL, pnl_kelly REAL, resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS clv_results (
    decision_id INTEGER PRIMARY KEY REFERENCES paper_decisions(id), status TEXT NOT NULL, book TEXT,
    comparison TEXT, decision_decimal REAL, close_decimal REAL, close_observed_at TEXT, price_clv REAL,
    open_novig REAL, close_novig REAL, novig_prob_move REAL, computed_at TEXT
);
"""

# Official injury reports (nbamodel/injuries.py): every URL tried, and a change log of player
# statuses keyed by report publication time. Append-only.
INJURY_SCHEMA = """
CREATE TABLE IF NOT EXISTS injury_report_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT NOT NULL UNIQUE, source TEXT NOT NULL, nominal_et TEXT,
    published_at TEXT, retrieved_at TEXT NOT NULL, http_status INTEGER, sha256 TEXT, n_rows INTEGER,
    parse_status TEXT NOT NULL, error TEXT, game_dates TEXT
);
CREATE TABLE IF NOT EXISTS injury_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER NOT NULL REFERENCES injury_report_files(id),
    published_at TEXT NOT NULL, retrieved_at TEXT NOT NULL, source TEXT NOT NULL,
    game_date TEXT NOT NULL, game_time_et TEXT, matchup TEXT, home_team TEXT, away_team TEXT, game_key TEXT,
    team TEXT, player_name TEXT, player_key TEXT NOT NULL, player_id TEXT, status TEXT, reason TEXT,
    UNIQUE (file_id, game_date, matchup, team, player_key)
);
CREATE INDEX IF NOT EXISTS injury_reports_date ON injury_reports (game_date, published_at);
CREATE INDEX IF NOT EXISTS injury_reports_game ON injury_reports (game_key, team, published_at);
"""

# Player box scores (nbamodel/espn.py), one row per player per game. Post-game facts: features
# only read games that finished before the prediction time.
PLAYER_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_games (
    espn_event_id TEXT NOT NULL, player_id TEXT NOT NULL, game_key TEXT, game_date TEXT NOT NULL,
    team TEXT NOT NULL, opponent TEXT, home INTEGER, player_name TEXT, starter INTEGER, did_not_play INTEGER,
    dnp_reason TEXT, minutes REAL, pts INTEGER, fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER,
    ftm INTEGER, fta INTEGER, oreb INTEGER, dreb INTEGER, reb INTEGER, ast INTEGER, stl INTEGER, blk INTEGER,
    tov INTEGER, pf INTEGER, plus_minus INTEGER, source TEXT NOT NULL DEFAULT 'espn', fetched_at TEXT,
    PRIMARY KEY (espn_event_id, player_id)
);
CREATE INDEX IF NOT EXISTS player_games_player ON player_games (player_id, game_date);
CREATE INDEX IF NOT EXISTS player_games_team ON player_games (team, game_date);
"""


def log_player_games(con: sqlite3.Connection, rows: pd.DataFrame) -> int:
    """Append player box rows; a (game, player) already stored is left as it was."""
    if rows.empty:
        return 0
    cols = [c for c in rows.columns if c != "source"]
    before = con.total_changes
    con.executemany(f"INSERT OR IGNORE INTO player_games ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [tuple(_none(v) for v in r) for r in rows[cols].itertuples(index=False)])
    con.commit()
    return con.total_changes - before


# Component probabilities logged next to the published one, so every model can be benchmarked
# against the market on exactly the same games and timestamps. Added to old databases in place.
COMPONENT_COLUMNS = ("p_elo", "p_logistic", "p_logistic_raw", "p_gbm", "p_gbm_raw", "p_ensemble", "p_blend")

METRIC_WINDOWS = {"all": None, "last_100": 100, "last_300": 300}


def connect(path=DEFAULT_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    con.executescript(PHASE3_SCHEMA)
    con.executescript(INJURY_SCHEMA)
    con.executescript(PLAYER_SCHEMA)
    if "game_dates" not in {r[1] for r in con.execute("PRAGMA table_info(injury_report_files)")}:
        con.execute("ALTER TABLE injury_report_files ADD COLUMN game_dates TEXT")
    have = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
    for col in COMPONENT_COLUMNS:
        if col not in have:
            con.execute(f"ALTER TABLE predictions ADD COLUMN {col} REAL")
    con.commit()
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
            r.get("key_factors"), *(r.get(c) for c in COMPONENT_COLUMNS))))
    before = con.total_changes
    con.executemany(f"""INSERT OR IGNORE INTO predictions (game_key, game_date, season, home_team, away_team,
        tipoff_utc, prediction_ts, model_version, p_home, p_away, fair_home_american, fair_away_american,
        fair_home_decimal, fair_away_decimal, market_home_odds, market_away_odds, market_odds_ts, market_book,
        market_home_novig, model_spread, low_history, features_asof, feature_snapshot, key_factors,
        {", ".join(COMPONENT_COLUMNS)})
        VALUES ({",".join("?" * (24 + len(COMPONENT_COLUMNS)))})""", rows)
    con.commit()
    return con.total_changes - before


def log_prediction_odds(con: sqlite3.Connection, preds: pd.DataFrame, quotes: pd.DataFrame) -> int:
    """Record every book's price that was available when each prediction was made.

    `quotes` has one row per (game_key, book) from odds_store.quotes_asof. Rows are keyed to the
    stored prediction (game_key, model_version, prediction_ts) and never updated afterwards."""
    if quotes is None or quotes.empty or preds.empty:
        return 0
    ids = pd.read_sql("SELECT id, game_key, model_version, prediction_ts FROM predictions", con)
    keyed = preds[["game_key", "model_version", "prediction_timestamp"]].rename(
        columns={"prediction_timestamp": "prediction_ts"}).merge(ids, on=["game_key", "model_version", "prediction_ts"])
    q = quotes.merge(keyed[["game_key", "id"]], on="game_key")
    before = con.total_changes
    con.executemany("INSERT OR IGNORE INTO prediction_odds VALUES (?,?,?,?,?,?,?,?)", [
        (int(r["id"]), r["book"], _none(r.get("snapshot_id")), str(r["observed_at"]), float(r["home_decimal"]),
         float(r["away_decimal"]), _none(r.get("home_american")), _none(r.get("away_american")))
        for r in q.to_dict("records")])
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
    before = con.total_changes
    con.executemany("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?)",
                    [(r.game_key, int(r.home_score), int(r.away_score), int(r.home_win), now)
                     for r in res.itertuples()])
    new_results = con.total_changes - before
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
    return {"results_recorded": int(new_results), "predictions_scored": int(len(pending)),
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
