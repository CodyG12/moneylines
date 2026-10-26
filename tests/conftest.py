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
