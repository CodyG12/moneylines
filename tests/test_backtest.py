import numpy as np
import pandas as pd
import pytest

from nbamodel.backtest import BettingConfig, select_bets, simulate


def _frame():
    return pd.DataFrame({
        "game_id": [0, 1, 2, 3], "date": pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"] * 2),
        "p": [0.60, 0.30, 0.50, 0.55], "home_odds": [2.0, 1.5, 1.9, np.nan], "away_odds": [1.9, 2.8, 1.9, 1.8],
        "home_win": [1.0, 0.0, 1.0, 1.0]})


def test_select_bets_picks_best_side_and_abstains():
    bets = select_bets(_frame(), "p", threshold=0.02)
    # game0: home EV 0.20; game1: away EV 0.7*2.8-1=0.96; game2: EV -0.05 both -> abstain; game3 no odds
    assert bets["game_id"].tolist() == [0, 1]
    assert bets["side"].tolist() == ["home", "away"]
    assert bets["est_ev"].tolist() == pytest.approx([0.20, 0.96])
    assert bets["won"].tolist() == [1.0, 1.0]


def test_flat_and_kelly_accounting():
    bets = select_bets(_frame(), "p", threshold=0.02)
    cfg = BettingConfig(kelly_multiplier=0.5, max_bet_fraction=0.5, max_daily_exposure=1.0, initial_bankroll=100)
    out, s = simulate(bets, cfg, n_boot=200)
    assert s["flat"]["realized_profit_units"] == pytest.approx(1.0 + 1.8)
    # same-date Kelly stakes are both sized from the opening bankroll of 100
    full_kelly = np.array([(0.6 * 2 - 1) / 1, (0.7 * 2.8 - 1) / 1.8])
    assert out["kelly_stake"].to_numpy() == pytest.approx(100 * 0.5 * full_kelly)


def test_daily_exposure_cap():
    bets = select_bets(_frame(), "p", threshold=0.02)
    cfg = BettingConfig(kelly_multiplier=1.0, max_bet_fraction=1.0, max_daily_exposure=0.10)
    out, _ = simulate(bets, cfg, n_boot=50)
    assert out["kelly_stake"].sum() == pytest.approx(10.0)


def test_drawdown_on_losses():
    df = _frame().assign(home_win=[0.0, 1.0, 1.0, 1.0])
    _, s = simulate(select_bets(df, "p", 0.02), BettingConfig(), n_boot=50)
    assert s["flat"]["realized_profit_units"] == -2.0
    assert s["flat"]["max_drawdown_units"] == 2.0
