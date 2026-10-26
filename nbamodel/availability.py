"""Ingestion adapter for timestamped player-availability (injury report) data.

Not used by the default feature set: this project has no reliable, timestamped
historical injury source (the NBA's official reports are published as PDFs and
nba_api does not archive them). This adapter defines the format such data must
have and computes features that respect report timing, so a real source can be
plugged in later without fabricating anything.

Expected CSV columns:
  report_ts   ISO time the report was published (required; rows without it are rejected)
  date        game date (YYYY-MM-DD, US/Eastern)
  team        team code matching the games file
  player      player name or id
  status      out | doubtful | questionable | probable | available
  impact      optional non-negative player weight (e.g. prior-season minutes share);
              computed by the provider from data available before report_ts

Features, per side, from each player's latest report at or before the game's
prediction_timestamp (later reports are ignored, however relevant):
  <side>_n_out       players listed out or doubtful
  <side>_impact_out  sum of their impact weights (0 when impact is missing)
"""
from __future__ import annotations

import pandas as pd

from .data import DataError

STATUSES = {"out", "doubtful", "questionable", "probable", "available"}
UNAVAILABLE = {"out", "doubtful"}


def prepare_reports(raw: pd.DataFrame) -> pd.DataFrame:
    missing = {"report_ts", "date", "team", "player", "status"} - set(raw.columns)
    if missing:
        raise DataError(f"availability data missing columns: {sorted(missing)}")
    df = raw.copy()
    df["report_ts"] = pd.to_datetime(df["report_ts"], utc=True, errors="coerce")
    if df["report_ts"].isna().any():
        raise DataError("availability rows without a parseable report_ts cannot be used safely")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    bad = set(df["status"]) - STATUSES
    if bad:
        raise DataError(f"unknown availability statuses: {sorted(bad)}")
    df["impact"] = pd.to_numeric(df["impact"], errors="coerce").fillna(0.0) if "impact" in df else 0.0
    if (df["impact"] < 0).any():
        raise DataError("impact must be non-negative")
    return df


def availability_features(rows: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """rows: dataset rows with date, home_team, away_team, prediction_timestamp."""
    reps = prepare_reports(reports)
    out = pd.DataFrame(index=rows.index)
    for side in ("home", "away"):
        key = rows[["date", f"{side}_team", "prediction_timestamp"]].rename(columns={f"{side}_team": "team"})
        key = key.reset_index().rename(columns={"index": "_row"})
        m = key.merge(reps, on=["date", "team"], how="inner")
        m = m[m["report_ts"] <= pd.to_datetime(m["prediction_timestamp"], utc=True)]
        latest = m.sort_values("report_ts").groupby(["_row", "player"]).tail(1)
        latest = latest[latest["status"].isin(UNAVAILABLE)]
        agg = latest.groupby("_row").agg(n_out=("player", "size"), impact_out=("impact", "sum"))
        out[f"{side}_n_out"] = agg["n_out"].reindex(out.index).fillna(0).to_numpy()
        out[f"{side}_impact_out"] = agg["impact_out"].reindex(out.index).fillna(0.0).to_numpy()
    out["n_out_diff"] = out["home_n_out"] - out["away_n_out"]
    out["impact_out_diff"] = out["home_impact_out"] - out["away_impact_out"]
    return out
