"""Pregame feature construction.

Every feature for a game on date D uses only games completed on dates < D.
Rest days use the schedule (known in advance), not results.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from .elo import EloConfig, compute_elo

FEATURES = [
    "elo_diff",
    "form_margin_diff",
    "form_winpct_diff",
    "rest_diff",
    "home_b2b",
    "away_b2b",
]

REST_CAP = 7  # days; also used for a team's first game of a season


class _TeamState:
    __slots__ = ("season", "recent", "last_date", "n_games_total")

    def __init__(self, window: int):
        self.season = None
        self.recent = deque(maxlen=window)
        self.last_date = None
        self.n_games_total = 0


def build_features(games: pd.DataFrame, elo_config: EloConfig = EloConfig(), window: int = 10) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")
    elo, _ = compute_elo(games, elo_config)
    out = games.join(elo)

    n = len(games)
    cols = {name: np.empty(n) for name in (
        "home_form_margin", "away_form_margin", "home_form_winpct", "away_form_winpct",
        "home_form_n", "away_form_n", "home_days_rest", "away_days_rest", "home_history", "away_history")}

    states: dict = {}
    home = games["home_team"].to_numpy()
    away = games["away_team"].to_numpy()
    season = games["season"].to_numpy()
    played = games["played"].to_numpy()
    margin = (games["home_score"] - games["away_score"]).to_numpy()

    groups = games.groupby("date", sort=True).indices
    for date in sorted(groups):
        idx = groups[date]
        # 1) Read features from state as of the start of this date.
        for i in idx:
            for side, team in (("home", home[i]), ("away", away[i])):
                st = states.setdefault(team, _TeamState(window))
                new_season = st.season != season[i]
                if new_season:
                    st.recent.clear()
                    st.season = season[i]
                if st.recent:
                    margins = [m for m, _ in st.recent]
                    wins = [w for _, w in st.recent]
                    cols[f"{side}_form_margin"][i] = float(np.mean(margins))
                    cols[f"{side}_form_winpct"][i] = float(np.mean(wins))
                else:  # no in-season history yet: neutral values, flagged via *_form_n
                    cols[f"{side}_form_margin"][i] = 0.0
                    cols[f"{side}_form_winpct"][i] = 0.5
                cols[f"{side}_form_n"][i] = len(st.recent)
                if st.last_date is None or new_season:
                    rest = REST_CAP
                else:
                    rest = min((pd.Timestamp(date) - st.last_date).days, REST_CAP)
                cols[f"{side}_days_rest"][i] = rest
                cols[f"{side}_history"][i] = st.n_games_total
        # 2) Only then fold in this date's results.
        for i in idx:
            for team, sign in ((home[i], 1.0), (away[i], -1.0)):
                st = states[team]
                st.last_date = pd.Timestamp(date)
                if played[i]:
                    m = sign * margin[i]
                    st.recent.append((m, 1.0 if m > 0 else 0.0))
                    st.n_games_total += 1

    for name, values in cols.items():
        out[name] = values
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    out["form_margin_diff"] = out["home_form_margin"] - out["away_form_margin"]
    out["form_winpct_diff"] = out["home_form_winpct"] - out["away_form_winpct"]
    out["rest_diff"] = out["home_days_rest"] - out["away_days_rest"]
    out["home_b2b"] = (out["home_days_rest"] == 1).astype(float)
    out["away_b2b"] = (out["away_days_rest"] == 1).astype(float)
    return out
