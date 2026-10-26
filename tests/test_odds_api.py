import json
from pathlib import Path

import pandas as pd
import pytest

from nbamodel import odds_api
from nbamodel.odds_api import TEAM_CODES, load_api_key, parse_events
from nbamodel.odds_join import attach_odds

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "odds_api_events.json").read_text())


def test_team_codes_cover_30_teams():
    assert len(set(TEAM_CODES.values())) == 30


def test_parse_events_maps_teams_and_sides():
    rows = parse_events(FIXTURE, "2026-10-26T18:00:00Z")
    assert len(rows) == 3  # unknown-team event skipped
    fd = rows[(rows.event_id == "ev1") & (rows.book == "fanduel")].iloc[0]
    # outcome order in the payload must not matter
    assert (fd.home_team, fd.away_team, fd.home_price, fd.away_price) == ("CHA", "WAS", 1.57, 2.50)


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nODDS_API_KEY='abc123'\n")
    assert load_api_key(env) == "abc123"
    with pytest.raises(odds_api.OddsApiError):
        load_api_key(tmp_path / "missing.env")


def _snapshots():
    early = parse_events(FIXTURE, "2026-10-26T18:00:00Z")
    late = parse_events(FIXTURE, "2026-10-26T22:30:00Z")
    late.loc[late.book == "draftkings", ["home_price", "away_price"]] = [1.70, 2.20]
    live = parse_events(FIXTURE, "2026-10-26T23:30:00Z")  # in-game for ev1
    live[["home_price", "away_price"]] = [1.05, 9.0]
    return pd.concat([early, late, live], ignore_index=True)


GAMES = pd.DataFrame({
    "date": ["2026-10-26", "2026-10-26"], "home_team": ["CHA", "DEN"], "away_team": ["WAS", "GSW"],
    "home_score": [None, None], "away_score": [None, None],
    "tipoff_utc": ["2026-10-26T23:00:00Z", "2026-10-27T02:00:00Z"]})


def test_attach_uses_latest_quote_before_lead_cutoff():
    out = attach_odds(GAMES, _snapshots(), book="draftkings", lead_minutes=60)
    # CHA tips 23:00 -> cutoff 22:00 -> only the 18:00 snapshot qualifies
    assert (out.loc[0, "home_odds"], out.loc[0, "away_odds"]) == (1.60, 2.40)
    assert out.loc[0, "odds_timestamp"] == pd.Timestamp("2026-10-26T18:00:00Z")
    # DEN tips 02:00 (21:00 ET on the 26th) -> 22:30 snapshot qualifies; 23:30 too, as pregame
    assert out.loc[1, "odds_timestamp"] == pd.Timestamp("2026-10-26T23:30:00Z")


def test_attach_never_uses_in_game_quotes():
    out = attach_odds(GAMES, _snapshots(), book="draftkings", lead_minutes=0)
    assert out.loc[0, "odds_timestamp"] == pd.Timestamp("2026-10-26T22:30:00Z")
    assert out.loc[0, "home_odds"] == 1.70


def test_attach_best_price_and_flipped_home_away():
    flipped_games = GAMES.assign(home_team=["WAS", "DEN"], away_team=["CHA", "GSW"])
    out = attach_odds(flipped_games, _snapshots().query("observed_at == '2026-10-26T18:00:00Z'"),
                      book="best", lead_minutes=60)
    # best across DK/FD at 18:00, with sides swapped to match the NBA's home team
    assert (out.loc[0, "home_odds"], out.loc[0, "away_odds"]) == (2.50, 1.60)
    assert out.loc[0, "odds_book"] == "best"


def test_attach_then_prepare_validates(monkeypatch):
    from nbamodel.data import prepare_games
    games, report = prepare_games(attach_odds(GAMES, _snapshots(), lead_minutes=60))
    assert report.odds_timing_verified and report.n_odds_after_game_dropped == 0
    assert games["home_odds"].notna().all()
