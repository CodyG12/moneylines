"""Player availability features: how much stronger or weaker a team is tonight than its recent
form, given the official injury report at prediction time.

Method (no fixed star penalties; every number is estimated from earlier games):

1. Player ratings: game-level regularized plus-minus. For each game,
       home margin = sum_home (min_i / 48) r_i - sum_away (min_j / 48) r_j + home court
   fitted by ridge regression over a trailing window of box scores (`fit_ratings`). The ridge
   shrinks each player toward the typical rating of players with his minutes per game (a prior
   estimated in the same window), so rookies, small samples and players who just changed teams
   get cautious ratings, and a rating follows the player, not the team.
   Ratings are refitted weekly (Mondays) from games before the cutoff only.
2. Replacement level: the minutes-weighted mean rating of low-minute players in the same window.
   Minutes a missing player leaves behind are assumed to go to replacement-level players.
3. Minutes: `healthy` = mean minutes over the player's last N games played; `recent` = his mean
   minutes over his current team's last N games (0 when he did not play). The team features
   (rolling ratings, Elo) already reflect `recent`.
4. Availability: P(plays | report status) is measured from earlier seasons (`play_probabilities`),
   by matching each player's last pregame status to whether he played. Players not on the report
   get the measured rate for unlisted rotation players, not 1.
5. Feature = change versus recent form, in points of margin:
       impact = sum_i (r_i - r_rep) * (P_i * healthy_i - recent_i) / 48
   A player out for the whole recent window adds ~0 (already in the team numbers); a star
   returning adds a positive amount. Questionable players contribute their expected value, and
   their spread enters `impact_sd` = sqrt(sum_i P_i (1 - P_i) ((r_i - r_rep) healthy_i / 48)^2).

Games with no injury report published before the prediction time get NaN features
(`report_available` = 0): absence of a report is not evidence that everyone is healthy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .injuries import NOT_SUBMITTED, STATUSES, player_key

UNLISTED = "unlisted"


@dataclass(frozen=True)
class ImpactConfig:
    rating_window_days: int = 730       # trailing box-score window for ratings
    ridge_lambda: float = 30.0          # shrinkage toward league average, in game-equivalents
    recent_games: int = 10              # matches DatasetConfig.window (the team features' window)
    healthy_games: int = 10             # last N games the player actually played
    rotation_minutes: float = 20.0      # "rotation player": healthy minutes at or above this
    replacement_mpg: float = 12.0       # replacement level = players below this many minutes a game
    # Used only until play_probabilities() has data to measure; conservative round numbers.
    default_play_prob: tuple = (("out", 0.0), ("doubtful", 0.1), ("questionable", 0.5), ("probable", 0.9),
                                ("available", 0.97), (UNLISTED, 0.97), (NOT_SUBMITTED, 0.9))


# ------------------------------------------------------------------ ratings
def team_game_margins(pg: pd.DataFrame) -> pd.DataFrame:
    """One row per game: espn_event_id, game_date, home team, home margin (from player points)."""
    played = pg[pg["did_not_play"] == 0]
    pts = played.groupby(["espn_event_id", "home"])["pts"].sum().unstack("home")
    dates = pg.groupby("espn_event_id")["game_date"].first()
    out = pd.DataFrame({"margin": pts.get(1) - pts.get(0), "game_date": dates}).dropna(subset=["margin"])
    return out.reset_index()


def fit_ratings(pg: pd.DataFrame, cutoff, cfg: ImpactConfig = ImpactConfig()) -> tuple:
    """(ratings: Series player_id -> points of margin per 48 minutes vs league average,
    replacement level) from games in [cutoff - window, cutoff). Nothing on or after `cutoff`."""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(days=cfg.rating_window_days)
    d = pd.to_datetime(pg["game_date"])
    win = pg[(d >= start) & (d < cutoff) & (pg["did_not_play"] == 0) & (pg["minutes"] > 0)]
    if win.empty:
        return pd.Series(dtype=float), 0.0
    games = team_game_margins(win)
    win = win[win["espn_event_id"].isin(games["espn_event_id"])]
    players = pd.Index(sorted(win["player_id"].unique()))
    gidx = pd.Index(games["espn_event_id"])
    rows = gidx.get_indexer(win["espn_event_id"])
    cols = players.get_indexer(win["player_id"])
    sign = np.where(win["home"].to_numpy() == 1, 1.0, -1.0)
    X = np.zeros((len(gidx), len(players) + 1))
    np.add.at(X, (rows, cols), sign * win["minutes"].to_numpy() / 48.0)
    X[:, -1] = 1.0                                           # home court (not shrunk)
    y = games["margin"].to_numpy(dtype=float)
    pen = np.full(len(players) + 1, cfg.ridge_lambda)
    pen[-1] = 1e-6
    A = X.T @ X + np.diag(pen)
    mpg = win.groupby("player_id")["minutes"].mean().reindex(players)
    tot = win.groupby("player_id")["minutes"].sum().reindex(players)
    # Pass 1 shrinks toward 0. Pass 2 shrinks toward a minutes-informed prior (the minutes-weighted
    # fit of pass-1 ratings on minutes per game): with little data a player is assumed to be like
    # other players who get his minutes, not like the league average, so deep-bench players (the
    # replacement pool) are not pulled up to average.
    first = np.linalg.solve(A, X.T @ y)[:-1]
    slope, intercept = np.polyfit(mpg.to_numpy(), first, 1, w=np.sqrt(tot.to_numpy()))
    prior = np.append(intercept + slope * mpg.to_numpy(), 0.0)
    beta = np.linalg.solve(A, X.T @ y + pen * prior)
    ratings = pd.Series(beta[:-1], index=players)
    low = mpg[mpg < cfg.replacement_mpg].index
    rep = float(np.average(ratings[low], weights=tot[low])) if len(low) else 0.0
    return ratings, rep


def rating_cutoff(day) -> pd.Timestamp:
    """Weekly refit: a game on day d uses ratings fitted on games before the Monday of its week.
    (Monthly would ignore up to four weeks of the newest games, worst early in the season.)"""
    d = pd.Timestamp(day).normalize()
    return d - pd.Timedelta(days=d.weekday())


# ------------------------------------------------------------------ minutes
def minutes_asof(pg: pd.DataFrame, asof_date, cfg: ImpactConfig = ImpactConfig()) -> pd.DataFrame:
    """Per player, from box scores strictly before `asof_date`: current team (latest game),
    healthy minutes (mean over his last N games played), recent minutes (mean over his current
    team's last N games, 0 when he did not play) and the player key used to match reports."""
    asof_date = pd.Timestamp(asof_date)
    past = pg[pd.to_datetime(pg["game_date"]) < asof_date]
    if past.empty:
        return pd.DataFrame(columns=["player_id", "team", "player_key", "healthy", "recent"])
    past = past.sort_values(["game_date", "espn_event_id"])
    latest = past.groupby("player_id").tail(1).set_index("player_id")
    played = past[(past["did_not_play"] == 0) & (past["minutes"] > 0)]
    healthy = played.groupby("player_id").tail(cfg.healthy_games).groupby("player_id")["minutes"].mean()
    team_games = past[["team", "game_date", "espn_event_id"]].drop_duplicates()
    last = team_games.sort_values(["game_date", "espn_event_id"]).groupby("team").tail(cfg.recent_games)
    recent_rows = past.merge(last[["team", "espn_event_id"]], on=["team", "espn_event_id"])
    n_games = last.groupby("team").size()
    out = pd.DataFrame({"team": latest["team"], "player_name": latest["player_name"]})
    sums = recent_rows.groupby(["player_id", "team"])["minutes"].sum()
    out["recent"] = [sums.get((pid, t), 0.0) / max(int(n_games.get(t, 0)), 1) for pid, t in zip(out.index, out["team"])]
    out["healthy"] = healthy.reindex(out.index).fillna(0.0)
    out["player_key"] = out["player_name"].map(player_key)
    return out.reset_index().rename(columns={"index": "player_id"})


# ------------------------------------------------------------------ availability probabilities
def play_probabilities(final_statuses: pd.DataFrame, pg: pd.DataFrame, cfg: ImpactConfig = ImpactConfig(),
                       min_n: int = 50) -> dict:
    """P(plays | last pregame status), measured on past games.

    final_statuses: one row per (game_key, team, player_key, status) as of each game's
    prediction time. A player "played" if his box row shows minutes > 0. `unlisted` is measured on
    rotation players (at least rotation_minutes in their previous game, so nothing later is used)
    who were not on the report. Statuses with fewer than `min_n` matched cases keep the default.
    Limitation: a player missing from the box score entirely (inactive, not dressed) cannot be
    seen, so the unlisted rate can be slightly high."""
    probs = dict(cfg.default_play_prob)
    box = pg.assign(player_key=pg["player_name"].map(player_key), played=(pg["minutes"] > 0) & (pg["did_not_play"] == 0))
    box = box.sort_values(["player_id", "game_date", "espn_event_id"])
    box["prev_minutes"] = box.groupby("player_id")["minutes"].shift(1)
    m = final_statuses.merge(box[["game_key", "team", "player_key", "played"]], on=["game_key", "team", "player_key"])
    for status, grp in m.groupby("status"):
        if status in STATUSES and len(grp) >= min_n:
            probs[status] = float(grp["played"].mean())
    listed = set(zip(final_statuses["game_key"], final_statuses["team"], final_statuses["player_key"]))
    covered = set(final_statuses["game_key"])
    if covered:
        rot = box[box["game_key"].isin(covered)]
        rot = rot[[(g, t, k) not in listed for g, t, k in zip(rot["game_key"], rot["team"], rot["player_key"])]]
        rot = rot[rot["prev_minutes"] >= cfg.rotation_minutes]
        if len(rot) >= min_n:
            probs[UNLISTED] = float(rot["played"].mean())
    return probs


# ------------------------------------------------------------------ team features
FEATURES = ["impact", "impact_sd", "n_rotation_out", "n_rotation_uncertain", "report_available"]


def team_availability(team: str, statuses: pd.DataFrame | None, minutes: pd.DataFrame, ratings: pd.Series,
                      replacement: float, probs: dict, cfg: ImpactConfig = ImpactConfig()) -> dict:
    """Features for one team at one prediction time.

    statuses: this team's rows from injuries.statuses_asof (None when no report was published
    before the prediction time). Returns the FEATURES plus `unmatched` (listed players without a
    box-score history: rookies, two-way call-ups, name variants)."""
    if statuses is None:
        return {**{f: np.nan for f in FEATURES}, "report_available": 0, "unmatched": []}
    roster = minutes[minutes["team"] == team].copy()
    st = statuses[statuses["status"] != NOT_SUBMITTED]
    status_by_key = dict(zip(st["player_key"], st["status"]))
    roster["status"] = roster["player_key"].map(status_by_key).fillna(UNLISTED)
    if (statuses["status"] == NOT_SUBMITTED).any() and st.empty:
        roster["status"] = NOT_SUBMITTED                       # team has not reported yet
    unmatched = sorted(set(status_by_key) - set(roster["player_key"]))
    roster["p"] = roster["status"].map(lambda s: probs.get(s, probs.get(UNLISTED)))
    value = (ratings.reindex(roster["player_id"]).fillna(0.0).to_numpy() - replacement)
    expected = roster["p"].to_numpy() * roster["healthy"].to_numpy()
    impact = float(np.sum(value * (expected - roster["recent"].to_numpy())) / 48.0)
    sd = float(np.sqrt(np.sum(roster["p"] * (1 - roster["p"]) * (value * roster["healthy"] / 48.0) ** 2)))
    rot = roster["healthy"] >= cfg.rotation_minutes
    return {"impact": impact, "impact_sd": sd, "n_rotation_out": int((rot & (roster["p"] < 0.5)).sum()),
            "n_rotation_uncertain": int((rot & roster["p"].between(0.05, 0.95, inclusive="neither")).sum()),
            "report_available": 1, "unmatched": unmatched}


# ------------------------------------------------------------------ dataset rows
def statuses_at(changes: pd.DataFrame, game_date: str, matchup: str, ts: str) -> pd.DataFrame:
    """Latest status per player for one game from the change log, as of `ts` (ISO UTC)."""
    c = changes[(changes["game_date"] == game_date) & (changes["matchup"] == matchup) & (changes["published_at"] <= ts)]
    last = c.groupby(["team", "player_key"], dropna=False).tail(1)
    return last[last["status"] != "not_listed"]


def final_statuses(games: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Each listed player's status at each game's prediction time (game_key, team, player_key,
    status): the input play_probabilities() measures outcomes against."""
    out = []
    for g in games.itertuples(index=False):
        st = statuses_at(changes, g.day, g.matchup, g.ts)
        if not st.empty:
            out.append(st[["team", "player_key", "status"]].assign(game_key=g.game_key))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["team", "player_key", "status", "game_key"])


def availability_features(rows: pd.DataFrame, changes: pd.DataFrame, first_report: pd.Series, pg: pd.DataFrame,
                          cfg: ImpactConfig = ImpactConfig()) -> pd.DataFrame:
    """Per dataset row (game_key, date, season, home_team, away_team, prediction_timestamp):
    home_/away_ FEATURES plus injury_impact_diff (home minus away) and injury_impact_sd.

    Everything is as of the row's prediction timestamp: reports published by then, box scores
    from earlier dates, ratings from the weekly cutoff before the game, and play probabilities
    measured on earlier seasons only. Rows on dates with no report before the prediction time
    get NaN with report_available = 0."""
    r = rows.copy()
    r["day"] = pd.to_datetime(r["date"]).dt.date.astype(str)
    r["matchup"] = r["away_team"] + "@" + r["home_team"]
    r["ts"] = pd.to_datetime(r["prediction_timestamp"], utc=True).map(lambda t: t.isoformat())
    pg = pg.assign(game_key=pg["game_key"].astype(str))
    out = pd.DataFrame(index=r.index)
    ratings_cache, minutes_cache, probs_cache = {}, {}, {}
    for season in sorted(r["season"].unique()):
        if season not in probs_cache:  # measured on earlier seasons only
            past = r[(r["season"] < season) & r["day"].isin(first_report.index)]
            if past.empty:
                probs_cache[season] = dict(cfg.default_play_prob)
            else:
                fs = final_statuses(past, changes)
                probs_cache[season] = play_probabilities(fs, pg[pg["game_key"].isin(set(past["game_key"]))], cfg)
    for i, g in r.iterrows():
        first = first_report.get(g["day"])
        cutoff = rating_cutoff(g["day"])
        if cutoff not in ratings_cache:
            ratings_cache[cutoff] = fit_ratings(pg, cutoff, cfg)
        if g["day"] not in minutes_cache:
            minutes_cache[g["day"]] = minutes_asof(pg, g["day"], cfg)
        ratings, rep = ratings_cache[cutoff]
        st = statuses_at(changes, g["day"], g["matchup"], g["ts"]) if first is not None and first <= g["ts"] else None
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            f = team_availability(team, None if st is None else st[st["team"] == team], minutes_cache[g["day"]],
                                  ratings, rep, probs_cache[g["season"]], cfg)
            for k in FEATURES:
                out.loc[i, f"{side}_{k}"] = f[k]
            out.loc[i, f"{side}_unmatched"] = len(f["unmatched"])
    out["injury_impact_diff"] = out["home_impact"] - out["away_impact"]
    out["injury_impact_sd"] = np.sqrt(out["home_impact_sd"] ** 2 + out["away_impact_sd"] ** 2)
    return out
