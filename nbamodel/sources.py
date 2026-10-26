"""Converters from provider formats into the canonical games CSV schema."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Team box-score totals kept from LeagueGameFinder (enough to estimate possessions).
BOX_STATS = ["fgm", "fga", "fg3m", "fta", "oreb", "dreb", "tov"]


def normalize_game_id(ids: pd.Series) -> pd.Series:
    """NBA game ids are 10-digit strings ("0022401187"); CSV round trips drop the zeros."""
    return ids.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def team_logs_to_games(logs: pd.DataFrame) -> pd.DataFrame:
    """Convert nba_api LeagueGameFinder team rows (two per game) into one row per game.

    MATCHUP is "AAA vs. BBB" for the home team and "AAA @ BBB" for the away team.
    Neutral-site games (international games, NBA Cup knockouts in Las Vegas) come
    back with "@" on *both* rows, e.g. "MIA @ WAS" twice. Those are kept with
    neutral_site=True, using the listed (nominal) home team as home_team.
    """
    needed = {"SEASON_ID", "GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP", "PTS"}
    missing = needed - set(logs.columns)
    if missing:
        raise ValueError(f"team logs missing columns: {sorted(missing)}")
    logs = logs.copy()
    logs["_home_row"] = logs["MATCHUP"].str.contains(" vs. ", regex=False)
    logs["_away_row"] = logs["MATCHUP"].str.contains(" @ ", regex=False)
    logs["_nominal_home"] = logs["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[-1]
    per_game = logs.groupby("GAME_ID").agg(rows=("GAME_ID", "size"), homes=("_home_row", "sum"),
                                           aways=("_away_row", "sum"))

    regular = per_game[(per_game["rows"] == 2) & (per_game["homes"] == 1) & (per_game["aways"] == 1)].index
    neutral = per_game[(per_game["rows"] == 2) & (per_game["aways"] == 2)].index
    skipped = len(per_game) - len(regular) - len(neutral)
    if skipped:
        log.warning("%d games had an unexpected row layout and were skipped", skipped)

    # For a neutral game the row whose team equals the nominal home plays the "home" role.
    logs["_is_home"] = logs["_home_row"] | (
        logs["GAME_ID"].isin(neutral) & (logs["TEAM_ABBREVIATION"] == logs["_nominal_home"]))
    keep = logs[logs["GAME_ID"].isin(regular.union(neutral))]
    home, away = keep[keep["_is_home"]], keep[~keep["_is_home"]]
    box = [c.upper() for c in BOX_STATS if c.upper() in logs.columns]
    merged = home.merge(away[["GAME_ID", "TEAM_ABBREVIATION", "PTS"] + box], on="GAME_ID", suffixes=("_h", "_a"))
    if len(merged) != len(regular) + len(neutral):
        log.warning("%d games lost while pairing home/away rows",
                    len(regular) + len(neutral) - len(merged))

    games = pd.DataFrame({
        "date": pd.to_datetime(merged["GAME_DATE"]).dt.date.astype(str),
        "season": merged["SEASON_ID"].astype(str).str[1:].astype(int),
        "home_team": merged["TEAM_ABBREVIATION_h"],
        "away_team": merged["TEAM_ABBREVIATION_a"],
        "home_score": merged["PTS_h"],
        "away_score": merged["PTS_a"],
        "neutral_site": merged["GAME_ID"].isin(neutral).astype(int),
        "game_id_source": merged["GAME_ID"],
    })
    for col in box:
        games[f"home_{col.lower()}"] = merged[f"{col}_h"].to_numpy()
        games[f"away_{col.lower()}"] = merged[f"{col}_a"].to_numpy()
    return games.sort_values(["date", "home_team"]).reset_index(drop=True)


def merge_schedule_info(games: pd.DataFrame, schedule_games: pd.DataFrame) -> pd.DataFrame:
    """Add tip-off times and neutral-site flags from `schedule_to_games` output, matched on game id.

    The schedule's isNeutral flag is reliable for older seasons where LeagueGameFinder
    reports neutral games as ordinary home/away games; a game is neutral if either says so.
    """
    out = games.copy()
    sched = schedule_games.assign(game_id_source=normalize_game_id(schedule_games["game_id_source"]))
    sched = sched.drop_duplicates("game_id_source").set_index("game_id_source")
    key = normalize_game_id(out["game_id_source"])
    out["game_id_source"] = key
    out["tipoff_utc"] = key.map(sched["tipoff_utc"])
    out["neutral_site"] = (out["neutral_site"].astype(int) | key.map(sched["neutral_site"]).fillna(0).astype(int))
    return out


def merge_box_scores(games: pd.DataFrame, box_games: pd.DataFrame) -> pd.DataFrame:
    """Fill box-score columns in `games` (e.g. schedule rows) from `team_logs_to_games` output."""
    out = games.copy()
    cols = [c for c in box_games.columns if c.startswith(("home_", "away_")) and c[5:] in BOX_STATS]
    if not cols:
        return out
    box = box_games.assign(game_id_source=normalize_game_id(box_games["game_id_source"]))
    box = box.drop_duplicates("game_id_source").set_index("game_id_source")[cols]
    key = normalize_game_id(out["game_id_source"])
    out["game_id_source"] = key
    for c in cols:
        out[c] = key.map(box[c])
    return out


def schedule_to_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Convert nba_api ScheduleLeagueV2 rows into canonical games rows.

    Keeps regular-season games only (gameId prefix "002"): preseason ("001") and
    the NBA Cup final ("006", not counted in standings) are dropped, as are games
    whose teams are not yet known. Final games keep their scores; everything
    else (scheduled, live, postponed) gets blank scores.
    """
    needed = {"gameId", "gameDateEst", "gameDateTimeUTC", "gameStatus", "isNeutral", "seasonYear",
              "homeTeam_teamTricode", "awayTeam_teamTricode", "homeTeam_score", "awayTeam_score"}
    missing = needed - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule missing columns: {sorted(missing)}")
    df = schedule[schedule["gameId"].astype(str).str.startswith("002")
                  & schedule["homeTeam_teamTricode"].notna() & schedule["awayTeam_teamTricode"].notna()].copy()
    final = df["gameStatus"].astype(int) == 3
    games = pd.DataFrame({
        "date": df["gameDateEst"].str[:10],
        "season": df["seasonYear"].astype(str).str[:4].astype(int),
        "home_team": df["homeTeam_teamTricode"],
        "away_team": df["awayTeam_teamTricode"],
        "home_score": df["homeTeam_score"].where(final).astype("Int64"),
        "away_score": df["awayTeam_score"].where(final).astype("Int64"),
        "neutral_site": df["isNeutral"].astype(str).str.lower().isin(["true", "1"]).astype(int),
        "tipoff_utc": df["gameDateTimeUTC"],
        "game_id_source": df["gameId"],
    })
    return games.sort_values(["date", "home_team"]).reset_index(drop=True)
