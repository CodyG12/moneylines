"""Loading and validating game CSVs.

Required columns: date, home_team, away_team, home_score, away_score
Optional columns:
  season                       integer season start year (2023 == 2023-24).
                               Inferred from the date when absent.
  home_odds, away_odds         decimal moneyline odds
  home_moneyline, away_moneyline  American odds (used if decimal absent)
  odds_timestamp               when the quoted odds were observed (ISO 8601)
  neutral_site                 1/true for games with no real home team (default false)

Rows with both scores blank are treated as scheduled (unplayed) games.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .odds import american_to_decimal

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score"]


class DataError(ValueError):
    """Raised when an input file cannot be used safely."""


@dataclass
class DataReport:
    """Data-quality flags surfaced to the user rather than silently fixed."""

    n_rows: int = 0
    n_played: int = 0
    n_scheduled: int = 0
    season_inferred: bool = False
    has_odds: bool = False
    odds_timing_verified: bool = False
    n_odds_missing: int = 0
    n_odds_after_game_dropped: int = 0
    n_unplayed_in_past: int = 0
    warnings: list = field(default_factory=list)

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)


def infer_season(dates: pd.Series) -> pd.Series:
    """NBA seasons start in Oct; Aug-Dec games belong to that year's season.

    Caveat: the 2019-20 bubble ran into Oct 2020 and 2020-21 started Dec 2020.
    Supply an explicit `season` column for those years.
    """
    return np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1).astype(int)


def load_games(path) -> tuple[pd.DataFrame, DataReport]:
    try:
        raw = pd.read_csv(path, comment="#")
    except FileNotFoundError as exc:
        raise DataError(f"games file not found: {path}") from exc
    return prepare_games(raw)


def prepare_games(raw: pd.DataFrame) -> tuple[pd.DataFrame, DataReport]:
    report = DataReport()
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataError(f"missing required columns: {missing}")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = raw.loc[df["date"].isna(), "date"].head(3).tolist()
        raise DataError(f"unparseable dates, e.g. {bad}")
    df["date"] = df["date"].dt.normalize()

    for col in ("home_team", "away_team"):
        df[col] = df[col].astype(str).str.strip().str.upper()
    if (df["home_team"] == df["away_team"]).any():
        raise DataError("found games where home_team == away_team")

    for col in ("home_score", "away_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    home_na, away_na = df["home_score"].isna(), df["away_score"].isna()
    if (home_na != away_na).any():
        raise DataError("found rows with only one score filled in")
    df["played"] = ~home_na
    if (df.loc[df["played"], "home_score"] == df.loc[df["played"], "away_score"]).any():
        raise DataError("found tied final scores; NBA games cannot end tied")
    df["home_win"] = np.where(df["played"], (df["home_score"] > df["away_score"]).astype(float), np.nan)

    if df.duplicated(["date", "home_team", "away_team"]).any():
        raise DataError("duplicate (date, home_team, away_team) rows")
    appearances = pd.concat([df[["date", "home_team"]].set_axis(["date", "team"], axis=1),
                             df[["date", "away_team"]].set_axis(["date", "team"], axis=1)])
    if appearances.duplicated().any():
        dup = appearances[appearances.duplicated()].head(3).values.tolist()
        raise DataError(f"a team is listed in two games on the same date, e.g. {dup}")

    if "neutral_site" in df.columns:
        df["neutral_site"] = df["neutral_site"].fillna(False).astype(str).str.lower().isin(["1", "1.0", "true"])
    else:
        df["neutral_site"] = False

    if "season" in df.columns and df["season"].notna().all():
        df["season"] = df["season"].astype(int)
    else:
        df["season"] = infer_season(df["date"])
        report.season_inferred = True

    _prepare_odds(df, report)

    df = df.sort_values(["date", "home_team"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "game_id", np.arange(len(df)))

    report.n_rows = len(df)
    report.n_played = int(df["played"].sum())
    report.n_scheduled = report.n_rows - report.n_played
    if report.n_played:
        last_played = df.loc[df["played"], "date"].max()
        stale = (~df["played"]) & (df["date"] < last_played)
        report.n_unplayed_in_past = int(stale.sum())
        if report.n_unplayed_in_past:
            report.warn(f"{report.n_unplayed_in_past} games before the last played date have no score; "
                        "treating as missing results (excluded from training and evaluation)")
    return df, report


def _prepare_odds(df: pd.DataFrame, report: DataReport) -> None:
    if {"home_odds", "away_odds"} <= set(df.columns):
        for col in ("home_odds", "away_odds"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    elif {"home_moneyline", "away_moneyline"} <= set(df.columns):
        for side in ("home", "away"):
            ml = pd.to_numeric(df[f"{side}_moneyline"], errors="coerce")
            dec = pd.Series(np.nan, index=df.index)
            ok = ml.notna()
            dec[ok] = american_to_decimal(ml[ok])
            df[f"{side}_odds"] = dec
    else:
        df["home_odds"] = np.nan
        df["away_odds"] = np.nan
        return

    report.has_odds = True
    bad = (df["home_odds"] <= 1.0) | (df["away_odds"] <= 1.0)
    if bad.any():
        raise DataError("decimal odds must be > 1.0")
    incomplete = df["home_odds"].isna() | df["away_odds"].isna()
    df.loc[incomplete, ["home_odds", "away_odds"]] = np.nan
    report.n_odds_missing = int(incomplete.sum())

    if "odds_timestamp" in df.columns:
        ts = pd.to_datetime(df["odds_timestamp"], errors="coerce", utc=True)
        # Tip-off times are not in the data, so the strictest check available is
        # that the quote was observed no later than the game's calendar date.
        after = ts.notna() & (ts.dt.tz_convert(None).dt.normalize() > df["date"])
        if after.any():
            report.n_odds_after_game_dropped = int(after.sum())
            df.loc[after, ["home_odds", "away_odds"]] = np.nan
            report.warn(f"dropped odds for {int(after.sum())} games whose odds_timestamp is after game date")
        report.odds_timing_verified = bool(ts.notna().all())
    if not report.odds_timing_verified:
        report.warn("odds have no (complete) odds_timestamp: quotes may be closing lines or later; "
                    "paper-trading results do not represent prices you could have actually bet")
