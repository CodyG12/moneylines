import numpy as np
import pandas as pd
import pytest

from nbamodel import clv, market, odds_store, store
from tests.odds_fixtures import GAMES, frame, pred, seeded_db, snap_row


def test_novig_known_values():
    nv = market.novig_frame([1 + 100 / 110], [1 + 100 / 110]).iloc[0]       # -110 / -110
    assert nv["implied_home"] == pytest.approx(110 / 210)
    assert nv["overround"] == pytest.approx(2 * 110 / 210 - 1)              # 4.76% margin
    assert nv["novig_home"] == pytest.approx(0.5)
    nv = market.novig_frame([1.5], [2.8]).iloc[0]
    q_h, q_a = 1 / 1.5, 1 / 2.8
    assert nv["novig_home"] == pytest.approx(q_h / (q_h + q_a))
    assert nv["novig_home"] + nv["novig_away"] == pytest.approx(1.0)


def test_attach_market_asof_uses_prediction_time_price(tmp_path):
    con = seeded_db(tmp_path / "p.db", frame(snap_row("2026-10-26T20:00:00Z", h=-150, a=130),
                                              snap_row("2026-10-26T22:30:00Z", h=-200, a=170),
                                              snap_row("2026-10-26T20:00:00Z", book="fanduel", h=-140, a=120)))
    preds = pd.DataFrame([pred("g_cha", "2026-10-26T22:00:00+00:00", 0.62, "2026-10-26T23:00:00Z")])
    out, quotes = market.attach_market_asof(preds, odds_store.load_snapshots(con), "draftkings", 120)
    r = out.iloc[0]
    assert r["home_odds"] == pytest.approx(1 + 100 / 150)                   # 20:00 price, not 22:30
    assert r["odds_timestamp"].startswith("2026-10-26T20:00")
    assert set(quotes["book"]) == {"draftkings", "fanduel"}
    assert r["edge_home"] == pytest.approx(0.62 - r["market_home_prob_novig"])
    best, _ = market.attach_market_asof(preds, odds_store.load_snapshots(con), "best", 120)
    assert best.iloc[0]["home_odds"] == pytest.approx(1 + 100 / 140) and best.iloc[0]["odds_book"] == "best"


def _scored_db(tmp_path):
    con = seeded_db(tmp_path / "p.db", frame(snap_row("2026-10-26T21:50:00Z", h=-150, a=130),
                                              snap_row("2026-10-26T22:50:00Z", h=-160, a=140),
                                              snap_row("2026-10-26T21:50:00Z", event="ev2", home="DEN", away="GSW",
                                                       h=-240, a=200, commence="2026-10-27T02:00:00Z")))
    snap = odds_store.load_snapshots(con)
    preds = pd.DataFrame([pred("g_cha", "2026-10-26T22:00:00+00:00", 0.62, "2026-10-26T23:00:00Z"),
                          pred("g_den", "2026-10-26T22:00:00+00:00", 0.70, "2026-10-27T02:00:00Z", home="DEN",
                               away="GSW")])
    preds, quotes = market.attach_market_asof(preds, snap, "draftkings", 600)
    store.log_predictions(con, preds)
    store.log_prediction_odds(con, preds, quotes)
    games = GAMES.assign(played=True, home_score=[110, 100], away_score=[100, 108], home_win=[1, 0])
    store.reconcile(con, games, now=pd.Timestamp("2026-10-27T06:00Z"))
    return con


def test_live_benchmark_aligns_games_and_horizons(tmp_path):
    con = _scored_db(tmp_path)
    df = market.live_comparison_frame(con, "draftkings")
    assert sorted(df["horizon"]) == ["2h-12h", "30m-2h"]                    # 4h and 1h before tip
    assert (pd.to_datetime(df["mkt_observed_at"], utc=True) <= pd.to_datetime(df["prediction_ts"], utc=True)).all()
    r = market.benchmark(df, n_boot=20)
    assert r["status"] == "ok" and r["n"] == 2
    y = np.array([1.0, 0.0])
    p_mkt = df.sort_values("game_key")["p_market"].to_numpy()
    expected = -np.mean(y * np.log(p_mkt) + (1 - y) * np.log(1 - p_mkt))
    assert r["market"]["log_loss"] == pytest.approx(expected)
    assert set(r["models"]) >= {"p_elo", "p_logistic_raw", "p_gbm", "p_ensemble", "p_home"}


def test_prediction_odds_are_immutable_records(tmp_path):
    con = _scored_db(tmp_path)
    before = con.execute("SELECT * FROM prediction_odds ORDER BY prediction_id, book").fetchall()
    preds = pd.DataFrame([pred("g_cha", "2026-10-26T22:00:00+00:00", 0.62, "2026-10-26T23:00:00Z")])
    quotes = pd.DataFrame([{"game_key": "g_cha", "book": "draftkings", "observed_at": "2026-10-26T21:59:00Z",
                            "home_decimal": 9.0, "away_decimal": 1.1}])
    assert store.log_prediction_odds(con, preds, quotes) == 0
    assert con.execute("SELECT * FROM prediction_odds ORDER BY prediction_id, book").fetchall() == before


# ------------------------------------------------------------------ CLV
def test_price_clv_formula():
    assert clv.price_clv(2.10, 2.00) == pytest.approx(0.05)
    assert clv.price_clv(1.80, 2.00) == pytest.approx(-0.10)
    with pytest.raises(ValueError):
        clv.price_clv(1.0, 2.0)


def _decision(**kw):
    d = {"id": 1, "game_key": "g_cha", "book": "draftkings", "side": "home", "decimal_odds": 1 + 100 / 150,
         "odds_observed_at": "2026-10-26T22:00:00+00:00", "tipoff_utc": "2026-10-26T23:00:00Z",
         "market_p_side_novig": 0.58}
    d.update(kw)
    return pd.DataFrame([d])


def test_clv_uses_same_book_final_pregame_snapshot(tmp_path):
    con = seeded_db(tmp_path / "p.db", frame(snap_row("2026-10-26T22:00:00Z", h=-150, a=130),
                                              snap_row("2026-10-26T22:50:00Z", h=-170, a=150),
                                              snap_row("2026-10-26T22:55:00Z", book="fanduel", h=-120, a=100),
                                              snap_row("2026-10-26T23:20:00Z", h=-1000, a=650)))   # in-game
    snap = odds_store.load_snapshots(con)
    r = clv.compute(_decision(), snap).iloc[0]
    assert r["status"] == "ok" and r["comparison"] == "same_book"
    assert r["close_decimal"] == pytest.approx(1 + 100 / 170)
    assert r["price_clv"] == pytest.approx((1 + 100 / 150) / (1 + 100 / 170) - 1)
    assert r["price_clv"] > 0                                         # took -150, closed -170
    assert r["novig_prob_move"] == pytest.approx(r["close_novig"] - 0.58)


def test_clv_unavailable_statuses(tmp_path):
    con = seeded_db(tmp_path / "p.db", frame(snap_row("2026-10-26T22:00:00Z"),
                                              snap_row("2026-10-26T22:10:00Z", book="fanduel")))
    snap = odds_store.load_snapshots(con)
    assert clv.compute(_decision(), snap).iloc[0]["status"] == "no_later_snapshot"
    assert clv.compute(_decision(book="betmgm"), snap).iloc[0]["status"] == "no_close_snapshot"
    assert clv.compute(_decision(book="fanduel", odds_observed_at="2026-10-26T21:00:00Z"),
                       snap).iloc[0]["status"] == "close_too_early"           # last quote 50 min before tip
    assert clv.compute(_decision(side=None, decimal_odds=np.nan), snap).iloc[0]["status"] == "no_decision_price"
