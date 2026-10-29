"""Results, schedule updates and player box scores from ESPN's public API.

stats.nba.com does not answer GitHub's runners and often times out from home connections, so
the hosted collector takes results from ESPN instead (it answers both; tested 2026-10-29):
  scoreboard  https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
  summary     https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=ID

This API is public but unofficial and undocumented; it can change without notice. Every update is
validated: a game's player points must add up to its team score, and team abbreviations must map
to the codes in the games file. ESPN results are written onto the existing schedule rows
(matched by date and teams), so NBA game ids, and every `game_key` that predictions, odds and
paper decisions reference, stay the same. A game only ESPN knows about (for example an NBA Cup
knockout game scheduled mid-season) gets `game_id_source` "espn:<event id>".

Player box scores are post-game facts. Features may only use them for games that finished
before the prediction time; they never stand in for a pregame injury report.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SOURCE = "espn"
ET = "US/Eastern"
ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
TEAMS = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
         "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}
REGULAR_SEASON = 2
PLAYER_COLUMNS = ["game_key", "espn_event_id", "game_date", "team", "opponent", "home", "player_id", "player_name",
                  "starter", "did_not_play", "dnp_reason", "minutes", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm",
                  "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "plus_minus", "fetched_at"]


class EspnError(RuntimeError):
    pass


def code(abbr: str) -> str:
    return ABBR.get(abbr, abbr)


# ESPN's edge cache sometimes serves an empty 200 for a URL (seen 2026-10-29 for whole dates);
# the same request with different (harmless) query parameters is served fresh.
QUERY_VARIANTS = ("lang=en&region=us", "limit=100", "lang=en&region=us&limit=200")


def _get(url: str, timeout: float = 30.0, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        full = f"{url}&{QUERY_VARIANTS[attempt % len(QUERY_VARIANTS)]}"
        try:
            with urllib.request.urlopen(urllib.request.Request(full, headers={"Accept": "application/json"}),
                                        timeout=timeout) as resp:
                body = resp.read()
            if not body:
                raise ValueError("empty response")
            return json.loads(body)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 8))
    raise EspnError(f"ESPN request failed: {url}: {last}")


# ------------------------------------------------------------------ parsing
def parse_scoreboard(data: dict) -> pd.DataFrame:
    """One row per event: espn_event_id, date (US/Eastern), tipoff_utc, home/away codes, final,
    scores, neutral_site, season_type."""
    rows = []
    for e in data.get("events", []):
        c = e["competitions"][0]
        side = {t["homeAway"]: t for t in c["competitors"]}
        if set(side) != {"home", "away"}:
            continue
        tip = pd.Timestamp(c.get("date") or e["date"])
        tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
        final = bool(c["status"]["type"].get("completed")) and c["status"]["type"].get("name") == "STATUS_FINAL"
        rows.append({
            "espn_event_id": str(e["id"]), "date": tip.tz_convert(ET).date().isoformat(),
            "tipoff_utc": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": code(side["home"]["team"]["abbreviation"]), "away_team": code(side["away"]["team"]["abbreviation"]),
            "final": final,
            "home_score": int(side["home"]["score"]) if final else None,
            "away_score": int(side["away"]["score"]) if final else None,
            "neutral_site": int(bool(c.get("neutralSite"))),
            "season_type": (e.get("season") or {}).get("type"),
            "season": (e.get("season") or {}).get("year"),
        })
    return pd.DataFrame(rows)


def _made_att(v: str) -> tuple:
    a, b = (v or "0-0").split("-")
    return int(a), int(b)


def _int(v):
    try:
        return int(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def parse_summary(data: dict, game_key: str | None = None, fetched_at: str | None = None) -> tuple:
    """(team box {"home": {...}, "away": {...}} in the games-file BOX_STATS names, player rows)."""
    comp = data["header"]["competitions"][0]
    sides = {t["id"]: t["homeAway"] for t in comp["competitors"]}
    abbr = {t["homeAway"]: code(t["team"]["abbreviation"]) for t in comp["competitors"]}
    event_id = str(data["header"]["id"])
    tip = pd.Timestamp(comp["date"])
    tip = tip.tz_localize("UTC") if tip.tzinfo is None else tip.tz_convert("UTC")
    day = tip.tz_convert(ET).date().isoformat()
    team_box = {}
    for t in data["boxscore"].get("teams", []):
        s = {x["name"]: x["displayValue"] for x in t["statistics"]}
        fgm, fga = _made_att(s.get("fieldGoalsMade-fieldGoalsAttempted"))
        fg3m, _ = _made_att(s.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
        _, fta = _made_att(s.get("freeThrowsMade-freeThrowsAttempted"))
        team_box[t["homeAway"]] = {"fgm": fgm, "fga": fga, "fg3m": fg3m, "fta": fta,
                                   "oreb": _int(s.get("offensiveRebounds")), "dreb": _int(s.get("defensiveRebounds")),
                                   "tov": _int(s.get("turnovers"))}   # player turnovers, as stats.nba.com counts them
    rows = []
    for team in data["boxscore"].get("players", []):
        ha = sides.get(team["team"]["id"]) or ("home" if code(team["team"]["abbreviation"]) == abbr["home"] else "away")
        for block in team.get("statistics", []):
            keys = block.get("keys") or []
            for a in block.get("athletes", []):
                st = dict(zip(keys, a.get("stats") or []))
                fgm, fga = _made_att(st.get("fieldGoalsMade-fieldGoalsAttempted"))
                fg3m, fg3a = _made_att(st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                ftm, fta = _made_att(st.get("freeThrowsMade-freeThrowsAttempted"))
                dnp = bool(a.get("didNotPlay")) or not st
                rows.append({
                    "game_key": game_key, "espn_event_id": event_id, "game_date": day, "team": abbr[ha],
                    "opponent": abbr["away" if ha == "home" else "home"], "home": int(ha == "home"),
                    "player_id": f"espn:{a['athlete']['id']}", "player_name": a["athlete"]["displayName"],
                    "starter": int(bool(a.get("starter"))), "did_not_play": int(dnp),
                    "dnp_reason": a.get("reason") if dnp else None,
                    "minutes": 0.0 if dnp else float(_int(st.get("minutes")) or 0),
                    "pts": 0 if dnp else _int(st.get("points")), "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a,
                    "ftm": ftm, "fta": fta, "oreb": _int(st.get("offensiveRebounds")),
                    "dreb": _int(st.get("defensiveRebounds")), "reb": _int(st.get("rebounds")),
                    "ast": _int(st.get("assists")), "stl": _int(st.get("steals")), "blk": _int(st.get("blocks")),
                    "tov": _int(st.get("turnovers")), "pf": _int(st.get("fouls")),
                    "plus_minus": 0 if dnp else _int(st.get("plusMinus")), "fetched_at": fetched_at})
    return team_box, pd.DataFrame(rows, columns=PLAYER_COLUMNS)


def validate_game(final_scores: dict, players: pd.DataFrame) -> list:
    """Problems that make a game's ESPN data unusable."""
    issues = []
    for side, score in final_scores.items():
        team = players[players["home"] == int(side == "home")]
        if team.empty:
            issues.append(f"no {side} players")
        elif int(team["pts"].fillna(0).sum()) != int(score):
            issues.append(f"{side} player points {int(team['pts'].fillna(0).sum())} != team score {score}")
    unknown = set(players["team"]) - TEAMS
    if unknown:
        issues.append(f"unknown team codes {sorted(unknown)}")
    return issues


# ------------------------------------------------------------------ fetching
def scoreboard(day, timeout: float = 30.0) -> pd.DataFrame:
    return parse_scoreboard(_get(f"{BASE_URL}/scoreboard?dates={pd.Timestamp(day):%Y%m%d}", timeout))


def summary(event_id: str, game_key: str | None = None, timeout: float = 30.0) -> tuple:
    return parse_summary(_get(f"{BASE_URL}/summary?event={event_id}", timeout), game_key,
                         pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))


def apply_results(raw: pd.DataFrame, board: pd.DataFrame, boxes: dict) -> tuple:
    """Write ESPN finals, box stats and tip-off times onto the raw games rows.

    raw: the games CSV as read (date, home_team, away_team, home_score, ..., game_id_source).
    board: parse_scoreboard rows; boxes: espn_event_id -> team box. Returns (updated raw, summary).
    Only regular-season events are used. Scores already recorded are never changed: a mismatch
    with ESPN is reported, not overwritten."""
    out = raw.copy()
    board = board[board["season_type"] == REGULAR_SEASON]
    key = out["date"].astype(str).str[:10] + "|" + out["home_team"] + "|" + out["away_team"]
    index = dict(zip(key, out.index))
    stats = {"updated": 0, "tipoff_changes": 0, "added": 0, "conflicts": [], "unmatched": []}
    new_rows = []
    for g in board.itertuples(index=False):
        i = index.get(f"{g.date}|{g.home_team}|{g.away_team}")
        if i is None:
            if g.home_team in TEAMS and g.away_team in TEAMS:
                new_rows.append({"date": g.date, "season": int(g.season) - 1 if g.season else None,
                                 "home_team": g.home_team, "away_team": g.away_team, "home_score": g.home_score,
                                 "away_score": g.away_score, "neutral_site": g.neutral_site,
                                 "game_id_source": f"espn:{g.espn_event_id}", "tipoff_utc": g.tipoff_utc})
                stats["added"] += 1
            else:
                stats["unmatched"].append(g.espn_event_id)
            continue
        if pd.isna(out.at[i, "tipoff_utc"]) or str(out.at[i, "tipoff_utc"]) != g.tipoff_utc:
            if not g.final:
                stats["tipoff_changes"] += int(pd.notna(out.at[i, "tipoff_utc"]))
                out.at[i, "tipoff_utc"] = g.tipoff_utc
        if not g.final:
            continue
        have = out.at[i, "home_score"]
        if pd.notna(have):
            if (int(have), int(out.at[i, "away_score"])) != (int(g.home_score), int(g.away_score)):
                stats["conflicts"].append(f"{g.date} {g.away_team}@{g.home_team}: file {int(have)}-"
                                          f"{int(out.at[i, 'away_score'])}, ESPN {int(g.home_score)}-{int(g.away_score)}")
            continue
        box = boxes.get(g.espn_event_id)
        if box is None:   # final but its box score failed validation or was not fetched: wait
            continue
        out.at[i, "home_score"], out.at[i, "away_score"] = int(g.home_score), int(g.away_score)
        for side in ("home", "away"):
            for stat, v in box[side].items():
                if f"{side}_{stat}" in out.columns:
                    out.at[i, f"{side}_{stat}"] = v
        out.at[i, "fetched_at"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
        stats["updated"] += 1
    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    return out, stats


def update(raw: pd.DataFrame, con, start, end, pause: float = 0.3) -> tuple:
    """Fetch ESPN for US/Eastern dates [start, end]: score newly final games (validated), refresh
    tip-off times, add games only ESPN lists, and store player box scores for every final game
    that has none yet. Returns (updated raw games, summary)."""
    from . import store

    boards = []
    for day in pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D"):
        boards.append(scoreboard(day))
        time.sleep(pause)
    board = pd.concat(boards, ignore_index=True) if boards else parse_scoreboard({})
    if board.empty:
        return raw, {"updated": 0, "tipoff_changes": 0, "added": 0, "conflicts": [], "unmatched": [],
                     "players": 0, "invalid": []}
    board = board[board["season_type"] == REGULAR_SEASON]
    have = {r[0] for r in con.execute("SELECT DISTINCT espn_event_id FROM player_games")}
    ids = dict(zip(raw["date"].astype(str).str[:10] + "|" + raw["home_team"] + "|" + raw["away_team"],
                   raw["game_id_source"]))
    boxes, invalid, n_players = {}, [], 0
    for g in board[board["final"]].itertuples(index=False):
        k = f"{g.date}|{g.home_team}|{g.away_team}"
        row_scored = k in ids and raw.loc[raw["game_id_source"] == ids[k], "home_score"].notna().any()
        if row_scored and g.espn_event_id in have:
            continue
        gid = ids.get(k) or f"espn:{g.espn_event_id}"
        box, players = summary(g.espn_event_id, game_key=_key(gid))
        time.sleep(pause)
        problems = validate_game({"home": g.home_score, "away": g.away_score}, players)
        if problems:
            invalid.append(f"{g.date} {g.away_team}@{g.home_team}: {'; '.join(problems)}")
            continue
        boxes[g.espn_event_id] = box
        n_players += store.log_player_games(con, players)
    out, stats = apply_results(raw, board, boxes)
    return out, {**stats, "players": n_players, "invalid": invalid}


def _key(game_id_source: str) -> str:
    from .sources import normalize_game_id
    return str(normalize_game_id(pd.Series([game_id_source])).iloc[0])
