"""Live paper-trading ledger: decisions, limits, resolution and bankroll bookkeeping (deterministic)."""
import pandas as pd
import pytest

from nbamodel import market, odds_store, paper, store
from tests.odds_fixtures import GAMES, frame, pred, snap_row

T = pd.Timestamp
CFG = paper.PaperConfig(book="draftkings", ev_threshold=0.02, initial_bankroll=100.0, kelly_multiplier=0.25,
                        max_bet_fraction=0.05, max_daily_exposure=0.10, max_stake_units=50.0, max_model_spread=0.2)


def _setup(tmp_path, snaps, preds_rows):
    con = store.connect(tmp_path / "p.db")
    odds_store.store_snapshots(con, snaps, GAMES)
    preds = pd.DataFrame(preds_rows)
    preds, quotes = market.attach_market_asof(preds, odds_store.load_snapshots(con), "draftkings", 120)
    store.log_predictions(con, preds)
    store.log_prediction_odds(con, preds, quotes)
    return con


def _decide_both(con, cfg):
    """Run the decision step before each game's tip-off, as the live schedule would."""
    return pd.concat([paper.decide(con, cfg, now=T("2026-10-26T22:01Z")),
                      paper.decide(con, cfg, now=T("2026-10-27T01:01Z"))], ignore_index=True)


def _even_snaps():
    # +100 / -120 for CHA game; DEN game +100/-120 as well
    return frame(snap_row("2026-10-26T21:55:00Z", h=100, a=-120),
                 snap_row("2026-10-27T00:55:00Z", event="ev2", home="DEN", away="GSW", h=100, a=-120,
                          commence="2026-10-27T02:00:00Z"))


def test_bet_decision_ev_edge_and_kelly_stake(tmp_path):
    con = _setup(tmp_path, _even_snaps(), [pred("g_cha", "2026-10-26T22:00:00+00:00", 0.55, "2026-10-26T23:00:00Z")])
    d = paper.decide(con, CFG, now=T("2026-10-26T22:01Z")).iloc[0]
    assert d["decision"] == "bet" and d["side"] == "home"
    assert d["decimal_odds"] == pytest.approx(2.0) and d["american_odds"] == 100
    assert d["est_ev"] == pytest.approx(0.55 * 2 - 1)
    assert d["market_p_side_novig"] == pytest.approx((1 / 2) / (1 / 2 + 120 / 220))
    full_kelly = (0.55 * 2 - 1) / (2 - 1)
    assert d["stake_kelly"] == pytest.approx(min(full_kelly * 0.25, 0.05) * 100)
    assert d["stake_flat"] == 1.0 and d["bankroll_at_decision"] == 100.0


def test_decisions_are_written_once_and_outside_window_ignored(tmp_path):
    con = _setup(tmp_path, _even_snaps(), [
        pred("g_cha", "2026-10-26T20:00:00+00:00", 0.55, "2026-10-26T23:00:00Z"),   # 3h out: not in window
        pred("g_cha", "2026-10-26T22:00:00+00:00", 0.55, "2026-10-26T23:00:00Z"),
        pred("g_cha", "2026-10-26T22:20:00+00:00", 0.30, "2026-10-26T23:00:00Z")])  # later, cannot revise
    assert paper.decide(con, CFG, now=T("2026-10-26T19:00Z")).empty                  # nothing in window yet
    first = paper.decide(con, CFG, now=T("2026-10-26T22:30Z"))
    assert len(first) == 1 and first.iloc[0]["decided_at"].startswith("2026-10-26T22:00")
    assert paper.decide(con, CFG, now=T("2026-10-26T22:40Z")).empty
    assert con.execute("SELECT COUNT(*) FROM paper_decisions").fetchone()[0] == 1


def test_abstentions(tmp_path):
    snaps = frame(snap_row("2026-10-26T18:00:00Z", h=100, a=-120))           # 4h old at decision: stale
    con = _setup(tmp_path, snaps, [
        pred("g_cha", "2026-10-26T22:00:00+00:00", 0.55, "2026-10-26T23:00:00Z"),
        pred("g_den", "2026-10-27T01:00:00+00:00", 0.70, "2026-10-27T02:00:00Z", home="DEN", away="GSW")])
    out = _decide_both(con, CFG).set_index("game_key")
    assert out.loc["g_cha", "decision"] == "stale_odds" and out.loc["g_den", "decision"] == "no_odds"

    con2 = _setup(tmp_path / "x", _even_snaps(), [
        pred("g_cha", "2026-10-26T22:00:00+00:00", 0.47, "2026-10-26T23:00:00Z"),               # EV -6% home
        pred("g_den", "2026-10-27T01:00:00+00:00", 0.60, "2026-10-27T02:00:00Z", home="DEN", away="GSW",
             spread=0.30)])
    out = _decide_both(con2, CFG).set_index("game_key")
    assert out.loc["g_cha", "decision"] == "below_threshold"
    assert out.loc["g_den", "decision"] == "model_disagreement"


def test_daily_exposure_cap_across_same_day_games(tmp_path):
    cfg = paper.PaperConfig(book="draftkings", max_bet_fraction=0.08, max_daily_exposure=0.10, max_stake_units=50,
                            initial_bankroll=100.0, kelly_multiplier=1.0, max_abs_edge=None)
    con = _setup(tmp_path, _even_snaps(), [
        pred("g_cha", "2026-10-26T22:00:00+00:00", 0.60, "2026-10-26T23:00:00Z"),
        pred("g_den", "2026-10-27T01:00:00+00:00", 0.60, "2026-10-27T02:00:00Z", home="DEN", away="GSW")])
    paper.decide(con, cfg, now=T("2026-10-26T22:01Z"))
    paper.decide(con, cfg, now=T("2026-10-27T01:01Z"))
    led = paper.ledger(con).set_index("game_key")
    assert led.loc["g_cha", "stake_kelly"] == pytest.approx(8.0)              # 8% cap of 100
    assert led.loc["g_den", "stake_kelly"] == pytest.approx(2.0)              # only 10 - 8 left that day
    assert led["stake_kelly"].sum() <= 10.0 + 1e-9


def test_resolution_pnl_bankroll_and_summary(tmp_path):
    con = _setup(tmp_path, _even_snaps(), [
        pred("g_cha", "2026-10-26T22:00:00+00:00", 0.55, "2026-10-26T23:00:00Z"),
        pred("g_den", "2026-10-27T01:00:00+00:00", 0.55, "2026-10-27T02:00:00Z", home="DEN", away="GSW")])
    _decide_both(con, CFG)
    games = GAMES.assign(played=True, home_score=[110, 100], away_score=[100, 108], home_win=[1, 0])
    store.reconcile(con, games, now=T("2026-10-27T06:00Z"))
    p_before = con.execute("SELECT id, p_home, market_home_odds FROM predictions ORDER BY id").fetchall()
    assert paper.resolve(con, now=T("2026-10-27T06:00Z")) == 2
    assert paper.resolve(con, now=T("2026-10-27T07:00Z")) == 0                 # settled once
    led = paper.ledger(con).set_index("game_key")
    stake = led.loc["g_cha", "stake_kelly"]
    assert led.loc["g_cha", "pnl_flat"] == pytest.approx(1.0) and led.loc["g_cha", "pnl_kelly"] == pytest.approx(stake)
    assert led.loc["g_den", "pnl_flat"] == pytest.approx(-1.0)
    s = paper.summarize(con, CFG, n_boot=20)
    assert s["n_settled"] == 2 and s["win_rate"] == 0.5
    assert s["flat"]["profit"] == pytest.approx(0.0) and s["flat"]["roi"]["estimate"] == pytest.approx(0.0)
    assert s["flat"]["max_drawdown_units"] == pytest.approx(1.0)
    assert s["kelly"]["bankroll"] == pytest.approx(100 + stake - led.loc["g_den", "stake_kelly"])
    # outcomes never rewrite the original probability or the price recorded with it
    assert con.execute("SELECT id, p_home, market_home_odds FROM predictions ORDER BY id").fetchall() == p_before
    assert paper.bankroll_at(con, CFG.strategy, 100.0, T("2026-10-28T00:00Z")) == pytest.approx(s["kelly"]["bankroll"])


def test_model_version_consistency(tmp_path):
    con = _setup(tmp_path, _even_snaps(), [pred("g_cha", "2026-10-26T22:00:00+00:00", 0.55, "2026-10-26T23:00:00Z",
                                                version="vA")])
    assert paper.decide(con, CFG, now=T("2026-10-26T22:01Z"), model_version="vB").empty
    d = paper.decide(con, CFG, now=T("2026-10-26T22:01Z"), model_version="vA").iloc[0]
    pv = con.execute("SELECT model_version FROM predictions WHERE id = ?", (int(d["prediction_id"]),)).fetchone()[0]
    assert d["model_version"] == pv == "vA"


def test_no_decisions_after_tipoff_and_huge_edges_abstain(tmp_path):
    con = _setup(tmp_path, _even_snaps(), [
        pred("g_cha", "2026-10-26T22:00:00+00:00", 0.55, "2026-10-26T23:00:00Z"),
        pred("g_den", "2026-10-27T01:00:00+00:00", 0.75, "2026-10-27T02:00:00Z", home="DEN", away="GSW")])
    # running the decision step after CHA tipped off must not create a (retroactive) CHA decision
    out = paper.decide(con, CFG, now=T("2026-10-27T01:05Z"))
    assert list(out["game_key"]) == ["g_den"]
    assert out.iloc[0]["decision"] == "edge_too_large"              # 0.75 vs ~0.48 no-vig
