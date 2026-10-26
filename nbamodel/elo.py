"""Online Elo ratings with home-court advantage, margin-of-victory scaling,
and between-season regression to the mean.

Leakage rule: every game on a given date is rated using ratings as of the
*start* of that date. Same-day results are never used, because NBA games on
the same date overlap and their results are not known before tip-off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    home_advantage: float = 100.0
    initial_rating: float = 1500.0
    season_carryover: float = 0.75
    use_margin: bool = True


def expected_score(rating_diff):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(rating_diff, dtype=float) / 400.0))


def margin_multiplier(margin: float, winner_rating_diff: float) -> float:
    """FiveThirtyEight-style MOV multiplier; damps blowouts by heavy favourites."""
    return ((abs(margin) + 3.0) ** 0.8) / max(7.5 + 0.006 * winner_rating_diff, 1.0)


def compute_elo(games: pd.DataFrame, config: EloConfig = EloConfig()):
    """Return (per-game pre-game ratings frame, final ratings dict).

    `games` must be sorted by date with a RangeIndex (see data.prepare_games).
    """
    ratings: dict = {}
    team_season: dict = {}
    n = len(games)
    home_pre = np.empty(n)
    away_pre = np.empty(n)

    home = games["home_team"].to_numpy()
    away = games["away_team"].to_numpy()
    season = games["season"].to_numpy()
    played = games["played"].to_numpy()
    margin = (games["home_score"] - games["away_score"]).to_numpy()
    hca = np.where(games["neutral_site"].to_numpy(), 0.0, config.home_advantage)

    groups = games.groupby("date", sort=True).indices
    for date in sorted(groups):
        idx = groups[date]
        for i in idx:
            for team in (home[i], away[i]):
                if team not in ratings:
                    ratings[team] = config.initial_rating
                elif team_season[team] != season[i]:
                    ratings[team] = (config.season_carryover * ratings[team]
                                     + (1 - config.season_carryover) * config.initial_rating)
                team_season[team] = season[i]
            home_pre[i] = ratings[home[i]]
            away_pre[i] = ratings[away[i]]

        for i in idx:
            if not played[i]:
                continue
            diff = home_pre[i] + hca[i] - away_pre[i]
            outcome = 1.0 if margin[i] > 0 else 0.0
            mult = 1.0
            if config.use_margin:
                mult = margin_multiplier(margin[i], diff if outcome else -diff)
            delta = config.k * mult * (outcome - float(expected_score(diff)))
            ratings[home[i]] += delta
            ratings[away[i]] -= delta

    out = pd.DataFrame({"home_elo_pre": home_pre, "away_elo_pre": away_pre}, index=games.index)
    out["elo_home_prob"] = expected_score(home_pre + hca - away_pre)
    return out, ratings
