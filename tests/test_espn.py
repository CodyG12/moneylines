"""ESPN results and player box scores: parsing (from trimmed real responses), validation and the
rules for writing onto the games file. No network access."""
import json
from pathlib import Path

import pandas as pd
import pytest

from nbamodel import espn, store

FIX = Path(__file__).parent / "fixtures"
BOARD = json.loads((FIX / "espn_scoreboard_20261027.json").read_text())
SUMMARY = json.loads((FIX / "espn_summary_401909877.json").read_text())


def test_scoreboard_maps_teams_dates_and_finals():
    b = espn.parse_scoreboard(BOARD)
    first = b.iloc[0]
    assert (first["home_team"], first["away_team"], first["home_score"], first["away_score"]) == ("BOS", "BKN", 125, 124)
    assert first["tipoff_utc"] == "2026-10-27T23:30:00Z" and first["date"] == "2026-10-27"
    late = b[b["home_team"] == "LAL"].iloc[0]
    assert late["tipoff_utc"] == "2026-10-28T03:00:00Z" and late["date"] == "2026-10-27"   # US/Eastern game date
    assert b["final"].all() and set(b["season_type"]) == {espn.REGULAR_SEASON}
    assert espn.code("GS") == "GSW" and espn.code("UTAH") == "UTA" and espn.code("BOS") == "BOS"


def test_summary_team_box_and_players():
    box, players = espn.parse_summary(SUMMARY, game_key="0022600128", fetched_at="2026-10-29T20:00:00+00:00")
    assert box["home"] == {"fgm": 42, "fga": 93, "fg3m": 11, "fta": 38, "oreb": 11, "dreb": 36, "tov": 14}
    assert box["away"]["tov"] == 15      # player turnovers (team turnovers excluded), as stats.nba.com counts
    assert players.groupby("team")["pts"].sum().to_dict() == {"BKN": 124, "BOS": 125}
    randle = players[players["player_name"] == "Julius Randle"].iloc[0]
    assert (randle["team"], randle["opponent"], randle["home"], randle["starter"]) == ("BKN", "BOS", 0, 1)
    assert (randle["minutes"], randle["pts"], randle["fga"], randle["plus_minus"]) == (44.0, 24, 23, 8)
    dnp = players[players["did_not_play"] == 1]
    assert len(dnp) == 5 and (dnp["minutes"] == 0).all() and dnp["dnp_reason"].notna().all()
    assert players["player_id"].str.startswith("espn:").all() and (players["game_key"] == "0022600128").all()


def test_validation_flags_points_that_do_not_add_up():
    _, players = espn.parse_summary(SUMMARY)
    assert espn.validate_game({"home": 125, "away": 124}, players) == []
    assert espn.validate_game({"home": 126, "away": 124}, players) == ["home player points 125 != team score 126"]


def _raw():
    return pd.DataFrame({
        "date": ["2026-10-27", "2026-10-27", "2026-10-30"], "season": [2026] * 3,
        "home_team": ["BOS", "NYK", "MIA"], "away_team": ["BKN", "DET", "ORL"],
        "home_score": [None, 111.0, None], "away_score": [None, 99.0, None], "neutral_site": [0, 0, 0],
        "game_id_source": ["0022600128", "0022600129", "0022600150"],
        "home_fgm": [None, 40.0, None], "away_fgm": [None, 38.0, None],
        "tipoff_utc": ["2026-10-27T23:30:00Z", "2026-10-28T00:00:00Z", "2026-10-30T23:30:00Z"], "fetched_at": [None] * 3})


def _board(rows):
    cols = ["espn_event_id", "date", "tipoff_utc", "home_team", "away_team", "final", "home_score", "away_score",
            "neutral_site", "season_type", "season"]
    return pd.DataFrame(rows, columns=cols)


def test_apply_results_fills_scores_and_never_overwrites():
    board = _board([
        ["e1", "2026-10-27", "2026-10-27T23:30:00Z", "BOS", "BKN", True, 125, 124, 0, 2, 2027],
        ["e2", "2026-10-27", "2026-10-28T00:00:00Z", "NYK", "DET", True, 111, 97, 0, 2, 2027],  # file says 111-99
        ["e3", "2026-10-30", "2026-10-31T00:00:00Z", "MIA", "ORL", False, None, None, 0, 2, 2027],  # moved
        ["e4", "2026-12-16", "2026-12-17T01:00:00Z", "LAL", "OKC", False, None, None, 1, 2, 2027],  # only on ESPN
        ["e5", "2026-10-27", "2026-10-28T02:00:00Z", "DEN", "GSW", True, 100, 90, 0, 1, 2027],   # preseason
    ])
    box = {"e1": {"home": {"fgm": 42}, "away": {"fgm": 45}}}
    out, st = espn.apply_results(_raw(), board, box)
    bos = out[out["home_team"] == "BOS"].iloc[0]
    assert (bos["home_score"], bos["away_score"], bos["home_fgm"], bos["away_fgm"]) == (125, 124, 42, 45)
    nyk = out[out["home_team"] == "NYK"].iloc[0]
    assert (nyk["home_score"], nyk["away_score"]) == (111, 99)
    assert st["conflicts"] == ["2026-10-27 DET@NYK: file 111-99, ESPN 111-97"]
    assert out[out["home_team"] == "MIA"].iloc[0]["tipoff_utc"] == "2026-10-31T00:00:00Z" and st["tipoff_changes"] == 1
    added = out[out["game_id_source"] == "espn:e4"].iloc[0]
    assert (added["season"], added["neutral_site"], st["added"]) == (2026, 1, 1)
    assert "DEN" not in set(out["home_team"]) and st["updated"] == 1


def test_final_game_without_a_valid_box_waits():
    board = _board([["e1", "2026-10-27", "2026-10-27T23:30:00Z", "BOS", "BKN", True, 125, 124, 0, 2, 2027]])
    out, st = espn.apply_results(_raw(), board, {})
    assert pd.isna(out[out["home_team"] == "BOS"].iloc[0]["home_score"]) and st["updated"] == 0


def test_update_scores_validates_and_stores_players_once(tmp_path, monkeypatch):
    con = store.connect(tmp_path / "p.db")
    calls = []
    bos_only = {"events": BOARD["events"][:1]}
    monkeypatch.setattr(espn, "scoreboard", lambda day, timeout=30.0: espn.parse_scoreboard(bos_only)
                        if pd.Timestamp(day) == pd.Timestamp("2026-10-27") else espn.parse_scoreboard({}))

    def fake_summary(event_id, game_key=None, timeout=30.0, variant=0):
        calls.append(event_id)
        if event_id != "401909877":
            raise AssertionError("only the BOS game should need a summary")
        return espn.parse_summary(SUMMARY, game_key, "2026-10-29T20:00:00+00:00")

    monkeypatch.setattr(espn, "summary", fake_summary)
    out, st = espn.update(_raw().iloc[[0]], con, "2026-10-27", "2026-10-27", pause=0)
    assert st["updated"] == 1 and st["players"] == 27 and st["invalid"] == [] and calls == ["401909877"]
    assert con.execute("SELECT COUNT(*) FROM player_games WHERE game_key = '0022600128'").fetchone()[0] == 27
    # second pass: the game is scored and its players stored, so nothing is fetched again
    calls.clear()
    monkeypatch.setattr(espn, "summary", lambda *a, **k: pytest.fail("no summary expected"))
    out2, st2 = espn.update(out[out["home_team"] == "BOS"], con, "2026-10-27", "2026-10-27", pause=0)
    assert st2["updated"] == 0 and st2["players"] == 0


def test_blank_cached_copy_is_refetched_before_giving_up(monkeypatch):
    good = espn.parse_summary(SUMMARY, "k")
    blank = (good[0], good[1].assign(pts=0))
    served = iter([blank, good])
    monkeypatch.setattr(espn, "summary", lambda *a, **k: next(served))
    g = espn.parse_scoreboard({"events": BOARD["events"][:1]}).iloc[0]
    _, players, problems = espn.validated_summary(g, "k", pause=0)
    assert problems == [] and players["pts"].sum() == 249
    monkeypatch.setattr(espn, "summary", lambda *a, **k: blank)
    assert espn.validated_summary(g, "k", pause=0)[2] != []
