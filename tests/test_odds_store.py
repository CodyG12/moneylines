import pandas as pd
import pytest

from nbamodel import odds_store, store
from tests.odds_fixtures import GAMES, frame, seeded_db, snap_row


def test_validation_converts_american_and_rejects_bad_rows():
    rows = frame(snap_row("2026-10-26T18:00:00Z"),
                 snap_row("2026-10-26T18:00:00Z", book="b2", h=-50, a=130),              # |american| < 100
                 snap_row("2026-10-26T18:00:00Z", book="b3", h=-1000, a=-1000),          # 82% margin
                 snap_row("2026-10-26T18:00:00Z", book="b4", provider_ts="2026-10-26T19:00:00Z"),  # from the future
                 snap_row("2026-10-26T18:00:00Z", book="b5", h=None, a=130),             # missing price
                 snap_row("not a time", book="b6"))
    valid, invalid = odds_store.validate_snapshots(rows)
    assert len(valid) == 1
    assert valid.iloc[0]["home_decimal"] == pytest.approx(1 + 100 / 150)
    assert valid.iloc[0]["away_decimal"] == pytest.approx(2.30)
    assert set(invalid["invalid_reason"]) == {"home_american_out_of_range", "implausible_overround",
                                              "provider_ts_after_observation", "missing_home_price", "bad_observed_at"}


def test_event_mapping_matched_flipped_unmatched_ambiguous():
    rows = frame(snap_row("2026-10-26T18:00:00Z"),
                 # provider lists DEN game the other way round: prices must be swapped to NBA orientation
                 snap_row("2026-10-26T18:00:00Z", event="ev2", home="GSW", away="DEN", h=200, a=-240,
                          commence="2026-10-27T02:05:00Z"),
                 snap_row("2026-10-26T18:00:00Z", event="ev3", home="BOS", away="NYK"),
                 # right teams but a different day: outside the 18h window
                 snap_row("2026-10-26T18:00:00Z", event="ev4", commence="2026-10-29T23:00:00Z"))
    valid, _ = odds_store.validate_snapshots(rows)
    m = odds_store.map_to_games(valid, GAMES).set_index("provider_event_id")
    assert m.loc["ev1", "game_key"] == "g_cha" and m.loc["ev1", "flipped"] == 0
    assert m.loc["ev2", "game_key"] == "g_den" and m.loc["ev2", "flipped"] == 1
    assert (m.loc["ev2", "home_team"], m.loc["ev2", "home_american"], m.loc["ev2", "away_american"]) == ("DEN", -240, 200)
    assert m.loc["ev3", "match_status"] == "unmatched" and m.loc["ev4", "match_status"] == "unmatched"
    twice = pd.concat([GAMES, GAMES.assign(game_key="g_cha2", tipoff_utc="2026-10-27T01:00:00Z")])
    amb = odds_store.map_to_games(valid.iloc[[0]], twice)
    assert amb.iloc[0]["match_status"] == "ambiguous" and amb.iloc[0]["game_key"] is None


def test_history_kept_duplicates_ignored_and_ingame_flagged(tmp_path):
    con = store.connect(tmp_path / "p.db")
    early = frame(snap_row("2026-10-26T18:00:00Z"), snap_row("2026-10-26T18:00:00Z", book="fanduel", h=-145, a=125))
    c1 = odds_store.store_snapshots(con, early, GAMES)
    assert c1["n_inserted"] == 2 and c1["n_duplicates"] == 0
    c2 = odds_store.store_snapshots(con, early, GAMES)              # same observation again
    assert c2["n_inserted"] == 0 and c2["n_duplicates"] == 2
    later = frame(snap_row("2026-10-26T22:50:00Z", h=-170, a=150),   # new price, earlier one kept
                  snap_row("2026-10-26T23:10:00Z", h=-900, a=600))   # in-game
    c3 = odds_store.store_snapshots(con, later, GAMES)
    assert c3["n_inserted"] == 2 and c3["n_live"] == 1
    assert con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0] == 4
    dk = odds_store.load_snapshots(con, book="draftkings")
    assert list(dk["home_american"]) == [-150, -170]                 # ordered by time; in-game excluded
    assert dk["observed_at"].is_monotonic_increasing
    assert len(odds_store.load_snapshots(con, pregame_only=False)) == 4


def test_quotes_asof_never_uses_later_prices_and_drops_stale(tmp_path):
    con = seeded_db(tmp_path / "p.db", frame(
        snap_row("2026-10-26T12:00:00Z", h=-130, a=110), snap_row("2026-10-26T20:00:00Z", h=-150, a=130),
        snap_row("2026-10-26T22:00:00Z", h=-170, a=150), snap_row("2026-10-26T22:40:00Z", h=-180, a=160),
        snap_row("2026-10-26T23:05:00Z", h=-900, a=600)))
    snap = odds_store.load_snapshots(con)
    asof = pd.Series([pd.Timestamp("2026-10-26T22:10:00Z")], index=["g_cha"])
    q = odds_store.quotes_asof(snap, asof)
    assert len(q) == 1 and q.iloc[0]["home_american"] == -170          # 22:40 is after the prediction
    stale = odds_store.quotes_asof(snap, pd.Series([pd.Timestamp("2026-10-26T19:00:00Z")], index=["g_cha"]),
                                   max_age=pd.Timedelta(minutes=120))
    assert stale.empty                                                # 12:00 quote is 7 hours old
    close = odds_store.closing_quotes(snap, ["g_cha"], "draftkings")
    assert close.iloc[0]["home_american"] == -180                      # last PREGAME quote, not in-game


def test_remap_matches_snapshots_stored_before_schedule_had_the_game(tmp_path):
    con = store.connect(tmp_path / "p.db")
    odds_store.store_snapshots(con, frame(snap_row("2026-10-26T18:00:00Z", event="ev2", home="GSW", away="DEN",
                                                   h=200, a=-240, commence="2026-10-27T02:00:00Z")),
                               GAMES.iloc[[0]])
    assert odds_store.load_snapshots(con).empty
    assert odds_store.remap_unmatched(con, GAMES) == 1
    s = odds_store.load_snapshots(con)
    assert (s.iloc[0]["game_key"], s.iloc[0]["home_team"], s.iloc[0]["home_american"]) == ("g_den", "DEN", -240)
    assert odds_store.remap_unmatched(con, GAMES) == 0


def test_best_price_takes_best_per_side():
    q = pd.DataFrame({"game_key": ["g", "g"], "book": ["a", "b"], "home_decimal": [1.60, 1.65],
                      "away_decimal": [2.45, 2.35], "observed_at": pd.to_datetime(["2026-10-26T18:00Z"] * 2)})
    b = odds_store.best_price(q).iloc[0]
    assert (b["home_decimal"], b["away_decimal"], b["book"]) == (1.65, 2.45, "best")
