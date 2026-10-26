import pandas as pd
import pytest

from nbamodel.data import prepare_games
from nbamodel.sample import generate_synthetic_games


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return generate_synthetic_games(seasons=(2022, 2023), games_per_season=300, seed=11)


@pytest.fixture
def synthetic_games(synthetic_raw):
    games, report = prepare_games(synthetic_raw)
    return games, report


def tiny(rows):
    """Build a canonical games frame from (date, home, away, hs, as) tuples."""
    raw = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    return prepare_games(raw)[0]


@pytest.fixture(scope="session")
def multi_season_raw() -> pd.DataFrame:
    """Four SYNTHETIC seasons with fake box-score totals and tip-off times (for Phase 2 tests)."""
    import numpy as np
    raw = generate_synthetic_games(seasons=(2019, 2020, 2021, 2022), games_per_season=300, seed=5)
    rng = np.random.default_rng(1)
    played = raw["home_score"].notna()
    for side in ("home", "away"):
        for col, mean in (("fga", 88), ("oreb", 10), ("tov", 13), ("fta", 22), ("fgm", 41), ("fg3m", 12),
                          ("dreb", 34)):
            raw[f"{side}_{col}"] = np.where(played, rng.normal(mean, 4, len(raw)).round(), np.nan)
    raw["tipoff_utc"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%dT23:30:00Z")
    raw["odds_timestamp"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%dT20:00:00Z")
    return raw


@pytest.fixture(scope="session")
def multi_season_games(multi_season_raw):
    return prepare_games(multi_season_raw)[0]
