"""Official injury reports: parsing (from word fixtures captured from real PDFs), matching,
timestamps and the append-only change log. No network access."""
import json
from pathlib import Path

import pandas as pd
import pytest

from nbamodel import injuries as inj, store

T = pd.Timestamp
FIX = Path(__file__).parent / "fixtures"
NEW = FIX / "injury_report_2026-10-27_05_00PM.words.json"
OLD = FIX / "injury_report_2023-01-10_05PM.words.json"


def _parsed(path):
    return inj.parse_words(json.loads(path.read_text())["pages"])


def test_new_format_report_parses_with_publication_time_and_context():
    pub, rows = _parsed(NEW)
    assert pub == T("2026-10-27T21:00:00Z")                 # 05:00 PM US/Eastern (EDT)
    assert set(rows["status"]) <= set(inj.STATUSES) | {inj.NOT_SUBMITTED}
    assert rows["team"].notna().all() and rows["game_date"].notna().all()
    murray = rows[rows["player_name"] == "Keegan Murray"].iloc[0]
    assert (murray["matchup"], murray["team"], murray["status"]) == ("SAC@SAS", "SAC", "out")
    assert murray["reason"] == "Injury/Illness - Left Left Ankle Capsulitis"    # wrapped above and below
    # page 2 has no column header and continues the SAC group from page 1
    raynaud = rows[rows["player_name"] == "Maxime Raynaud"].iloc[0]
    assert (raynaud["page"], raynaud["team"], raynaud["reason"]) == (1, "SAC", "Injury/Illness - Right Patella Subluxation")
    det = rows[(rows["matchup"] == "CHA@DET") & (rows["team"] == "DET")]
    assert list(det["status"]) == [inj.NOT_SUBMITTED] and det["player_name"].isna().all()


def test_old_format_report_names_and_hourly_header():
    pub, rows = _parsed(OLD)
    assert pub == T("2023-01-10T22:30:00Z")                 # the file is named 05PM; the header says 05:30 PM
    first = rows.iloc[0]
    assert (first["game_date"], first["game_time"], first["matchup"]) == ("2023-01-10", "07:00", "DET@PHI")
    assert first["player_name"] == "Marvin Bagley III" and first["reason"].startswith("Injury/Illness - Right Hand")
    assert (rows.loc[rows["player_name"] == "Joel Embiid", "status"] == "questionable").all()


def test_names_teams_and_urls():
    assert inj.display_name("Bagley III, Marvin") == "Marvin Bagley III"
    assert inj.display_name("Ajayi,Michael") == "Michael Ajayi"
    assert inj.player_key("Dončić, Luka") == inj.player_key("Luka Doncic") == "lukadoncic"
    assert inj.player_key("Marvin Bagley III") == "marvinbagley"
    assert inj.player_key("Jaren Jackson Jr.") == "jarenjackson"
    assert inj.team_code("BrooklynNets") == inj.team_code("Brooklyn Nets") == "BKN"
    assert inj.team_code("LA Clippers") == "LAC" and inj.team_code("Gotham Knights") is None
    hourly = inj.report_url(T("2023-01-10 17:00"), quarter_hourly=False)
    quarter = inj.report_url(T("2026-10-27 17:15"), quarter_hourly=True)
    assert hourly.endswith("Injury-Report_2023-01-10_05PM.pdf")
    assert quarter.endswith("Injury-Report_2026-10-27_05_15PM.pdf")
    assert inj.nominal_time(hourly) == T("2023-01-10 17:00") and inj.nominal_time(quarter) == T("2026-10-27 17:15")
    assert inj.season_label(T("2026-10-27")) == "2026-27" and inj.season_label(T("2027-03-01")) == "2026-27"


# ------------------------------------------------------------------ change log
GAMES = pd.DataFrame({"game_key": ["g1"], "date": [T("2026-11-02")], "home_team": ["BOS"], "away_team": ["NYK"]})


def _report(pub_et, rows):
    df = pd.DataFrame([{"game_date": "2026-11-02", "game_time": "07:30", "matchup": "NYK@BOS", "away_team": "NYK",
                        "home_team": "BOS", "page": 0, **r} for r in rows])
    return T(pub_et).tz_localize(inj.ET).tz_convert("UTC"), df


def _ingest(con, nominal, rows, pub=None):
    url = inj.report_url(T(nominal), quarter_hourly=True)
    parsed = _report(pub or nominal, rows)
    return inj.ingest(con, url, 200, b"%PDF fake " + url.encode(), T("2026-11-02T23:59Z"), GAMES, parsed=parsed)


@pytest.fixture()
def con(tmp_path):
    return store.connect(tmp_path / "inj.db")


def test_change_log_records_changes_and_drops_only(con):
    p = {"team": "BOS", "player_name": "Jayson Tatum", "status": "questionable", "reason": "Ankle"}
    q = {"team": "NYK", "player_name": "Jalen Brunson", "status": "out", "reason": "Calf"}
    assert _ingest(con, "2026-11-02 13:00", [p, q])["n_changes"] == 2
    assert _ingest(con, "2026-11-02 13:15", [p, q])["n_changes"] == 0           # unchanged: nothing new
    r = _ingest(con, "2026-11-02 17:00", [{**p, "status": "available"}])      # Brunson dropped from the list
    assert r["n_changes"] == 2
    log = inj.status_changes(con, "g1")
    assert list(log["status"]) == ["questionable", "out", "available", inj.NOT_LISTED]
    assert (log["published_at"].iloc[-1]) == "2026-11-02T22:00:00+00:00"      # 5 PM EST, stored in UTC
    files = pd.read_sql("SELECT * FROM injury_report_files ORDER BY id", con)
    assert len(files) == 3 and (files["parse_status"] == "ok").all()


def test_statuses_asof_never_sees_later_reports(con):
    p = {"team": "BOS", "player_name": "Jayson Tatum", "status": "questionable", "reason": "Ankle"}
    _ingest(con, "2026-11-02 13:00", [p])
    _ingest(con, "2026-11-02 17:00", [p])                     # unchanged: no change rows, but fresher
    _ingest(con, "2026-11-02 18:30", [{**p, "status": "out"}])
    before = inj.statuses_asof(con, T("2026-11-02T22:00Z"))                    # 5 PM ET, before the 6:30 report
    assert list(before["status"]) == ["questionable"]
    assert list(inj.statuses_asof(con, T("2026-11-02T23:45Z"))["status"]) == ["out"]
    assert inj.statuses_asof(con, T("2026-11-02T17:00Z")).empty               # before the first report
    cov = inj.report_coverage(con, T("2026-11-02T22:00Z"), "2026-11-02")
    assert cov["report_published_at"] == "2026-11-02T22:00:00+00:00" and cov["report_age_minutes"] == 0


def test_unlisted_players_are_unknown_not_available(con):
    _ingest(con, "2026-11-02 13:00", [{"team": "BOS", "player_name": "Jayson Tatum", "status": "out", "reason": "x"}])
    st = inj.statuses_asof(con, T("2026-11-02T23:00Z"))
    assert "jalenbrunson" not in set(st["player_key"])   # never listed: absent, the reader does not invent a status


def test_not_submitted_clears_when_the_team_submits(con):
    _ingest(con, "2026-11-02 13:00", [{"team": "BOS", "player_name": None, "status": inj.NOT_SUBMITTED, "reason": None},
                                      {"team": "NYK", "player_name": "Jalen Brunson", "status": "out", "reason": "x"}])
    assert inj.report_coverage(con, T("2026-11-02T18:30Z"), "2026-11-02")["not_submitted"] == ["BOS"]
    _ingest(con, "2026-11-02 14:00", [{"team": "BOS", "player_name": "Jayson Tatum", "status": "probable", "reason": "y"},
                                      {"team": "NYK", "player_name": "Jalen Brunson", "status": "out", "reason": "x"}])
    cov = inj.report_coverage(con, T("2026-11-02T19:30Z"), "2026-11-02")
    assert cov["not_submitted"] == []
    assert set(inj.statuses_asof(con, T("2026-11-02T19:30Z"))["player_key"]) == {"jaysontatum", "jalenbrunson"}


def test_out_of_order_missing_and_duplicate_files_add_nothing(con):
    p = {"team": "BOS", "player_name": "Jayson Tatum", "status": "out", "reason": "x"}
    _ingest(con, "2026-11-02 17:00", [p])
    assert _ingest(con, "2026-11-02 13:00", [{**p, "status": "available"}])["status"] == "out_of_order"
    assert _ingest(con, "2026-11-02 17:00", [p])["status"] == "duplicate"
    url = inj.report_url(T("2026-11-02 18:00"), quarter_hourly=True)
    assert inj.ingest(con, url, 403, b"", T("2026-11-02T23:00Z"), GAMES)["status"] == "missing"
    assert url in inj.known_urls(con)
    assert len(pd.read_sql("SELECT * FROM injury_reports", con)) == 1


def test_unparseable_pdf_is_recorded_as_error(con):
    url = inj.report_url(T("2026-11-02 13:00"), quarter_hourly=True)
    r = inj.ingest(con, url, 200, b"not a pdf", T("2026-11-02T18:00Z"), GAMES)
    assert r["status"] == "error"
    assert pd.read_sql("SELECT parse_status FROM injury_report_files", con)["parse_status"].tolist() == ["error"]


def test_game_key_matching(con):
    _ingest(con, "2026-11-02 13:00", [{"team": "BOS", "player_name": "Jayson Tatum", "status": "out", "reason": "x"}])
    assert pd.read_sql("SELECT game_key FROM injury_reports", con)["game_key"].tolist() == ["g1"]


def test_backfilled_report_counts_only_for_dates_it_is_newer_for(con):
    nov3 = pd.DataFrame({"game_key": ["g2"], "date": [T("2026-11-03")], "home_team": ["MIA"], "away_team": ["ORL"]})
    games = pd.concat([GAMES, nov3])
    live = pd.DataFrame([{"game_date": "2026-11-03", "game_time": "07:30", "matchup": "ORL@MIA", "away_team": "ORL",
                          "home_team": "MIA", "team": "MIA", "player_name": "Tyler Herro", "status": "out",
                          "reason": "x", "page": 0}])
    inj.ingest(con, inj.report_url(T("2026-11-03 12:00"), True), 200, b"a", T("2026-11-03T17:00Z"), games,
               parsed=(T("2026-11-03T17:00Z"), live))
    # an older report (day before) listing both nights, fetched later by the backfill
    older = pd.concat([_report("2026-11-02 17:00", [{"team": "BOS", "player_name": "Jayson Tatum", "status": "out",
                                                     "reason": "x"}])[1],
                       live.assign(status="questionable")])
    r = inj.ingest(con, inj.report_url(T("2026-11-02 17:00"), True), 200, b"b", T("2026-11-03T18:00Z"), games,
                   parsed=(T("2026-11-02T22:00Z"), older))
    assert r["status"] == "ok" and r["n_changes"] == 1                       # Nov 2 rows only
    assert inj.statuses_asof(con, T("2026-11-03T18:00Z"), ["2026-11-03"])["status"].tolist() == ["out"]
    assert inj.statuses_asof(con, T("2026-11-02T23:00Z"), ["2026-11-02"])["player_key"].tolist() == ["jaysontatum"]
