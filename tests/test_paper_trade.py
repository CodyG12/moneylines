import numpy as np
import pandas as pd
import pytest

from nbamodel.backtest import BettingConfig, decision_table, paper_trade, threshold_sweep


def _preds():
    """Hand-built predictions with archived odds. Prices are illustrative test fixtures, not real lines."""
    d = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3)
    return pd.DataFrame({
        "game_key": list("abcdef"), "date": d, "home_win": [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        "p": [0.60, 0.55, 0.50, 0.70, 0.40, 0.65],
        "home_odds": [2.0, 2.0, 1.9, 2.0, np.nan, 1.8], "away_odds": [1.9, 1.9, 1.9, 1.9, 2.0, 2.1],
        "odds_timestamp": ["2024-01-01T20:00Z", None, "2024-01-01T20:00Z", "2024-01-03T01:00Z",
                           "2024-01-02T20:00Z", "2024-01-02T20:00Z"],
        "prediction_timestamp": pd.to_datetime(["2024-01-01T23:00Z"] * 3 + ["2024-01-02T23:00Z"] * 3),
        "close_home_odds": [1.8, 1.9, 1.9, 1.9, 2.0, 1.7], "close_away_odds": [2.1, 2.0, 1.9, 2.0, 1.9, 2.25],
    })


def test_every_game_gets_a_decision():
    t = decision_table(_preds(), "p", 0.02)
    assert t["decision"].tolist() == ["bet", "no_odds_timestamp", "below_threshold", "odds_after_decision",
                                      "no_odds", "bet"]


def test_untimestamped_odds_allowed_only_when_explicitly_relaxed():
    t = decision_table(_preds(), "p", 0.02, require_timestamps=False)
    assert t.loc[1, "decision"] == "bet"


def test_paper_trade_summary_and_clv():
    bets, s = paper_trade(_preds(), "p", BettingConfig(ev_threshold=0.02), n_boot=100)
    assert s["status"] == "ok" and s["n_bets"] == 2
    assert s["decisions"]["odds_after_decision"] == 1
    assert bets["game_key"].tolist() == ["a", "f"]
    assert s["flat"]["realized_profit_units"] == pytest.approx(1.0 + 0.8)
    assert s["clv"]["n"] == 2 and s["clv"]["mean"] > 0      # both prices beat the close
    assert (bets["edge"] > 0).all()
    assert "SIMULATED" in s["caveat"]
    assert s["overfitting_check"]["n_thresholds_tried"] >= 3


def test_no_valid_odds_means_no_simulation():
    p = _preds().assign(home_odds=np.nan)
    bets, s = paper_trade(p, "p")
    assert bets is None and s["status"] == "no_valid_odds"


def test_threshold_sweep_bonferroni_interval_is_wider():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 100, n), "D"),
                       "est_ev": rng.uniform(0, 0.1, n), "odds": 1.95, "won": rng.integers(0, 2, n)})
    out = threshold_sweep(df, "p", (0.0, 0.02, 0.05), n_boot=300)
    best = next(r for r in out["by_threshold"] if r["threshold"] == out["best_threshold"])
    lo, hi = out["best_roi_bonferroni_ci"]
    assert lo <= best["ci_low"] and hi >= best["ci_high"]
