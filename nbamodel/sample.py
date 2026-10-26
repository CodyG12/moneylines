"""Deterministic SYNTHETIC game generator for tests and demos.

These games are simulated from made-up team strengths. They are not real
NBA results and say nothing about how the model performs on real data.
Team codes are SYN01..SYNnn so they cannot be mistaken for real teams.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOME_EDGE_POINTS = 2.5
MARGIN_SD = 12.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def generate_synthetic_games(n_teams: int = 30, seasons=(2022, 2023), games_per_season: int = 600,
                             games_per_date: int = 10, n_upcoming_dates: int = 2, with_odds: bool = True,
                             margin: float = 0.045, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"SYN{i + 1:02d}" for i in range(n_teams)]
    strength = rng.normal(0, 5, n_teams)
    rows = []
    for s_i, season in enumerate(seasons):
        if s_i:
            strength = 0.7 * strength + rng.normal(0, 3, n_teams)
        day = pd.Timestamp(f"{season}-10-20")
        n_dates = math.ceil(games_per_season / games_per_date)
        is_last = s_i == len(seasons) - 1
        for d in range(n_dates + (n_upcoming_dates if is_last else 0)):
            upcoming = is_last and d >= n_dates
            order = rng.permutation(n_teams)[: 2 * games_per_date]
            for g in range(games_per_date):
                h, a = order[2 * g], order[2 * g + 1]
                expected = strength[h] - strength[a] + HOME_EDGE_POINTS
                row = {"date": day.date().isoformat(), "season": season,
                       "home_team": teams[h], "away_team": teams[a],
                       "home_score": None, "away_score": None}
                if not upcoming:
                    m = int(round(expected + rng.normal(0, MARGIN_SD)))
                    if m == 0:
                        m = 1 if rng.random() < 0.5 else -1  # overtime
                    total = int(round(rng.normal(226, 14)))
                    row["home_score"] = (total + m) // 2 + (total + m) % 2
                    row["away_score"] = row["home_score"] - m
                if with_odds:
                    p_true = _norm_cdf(expected / MARGIN_SD)
                    logit = math.log(p_true / (1 - p_true)) + rng.normal(0, 0.15)
                    q = min(max(1 / (1 + math.exp(-logit)), 0.05), 0.95)
                    row["home_odds"] = round(1 / (q * (1 + margin)), 2)
                    row["away_odds"] = round(1 / ((1 - q) * (1 + margin)), 2)
                    row["odds_timestamp"] = f"{day.date().isoformat()}T15:00:00Z"
                rows.append(row)
            day += pd.Timedelta(days=1)
    df = pd.DataFrame(rows)
    for col in ("home_score", "away_score"):
        df[col] = df[col].astype("Int64")
    return df
