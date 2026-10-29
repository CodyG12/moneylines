"""Risk-adjusted opportunity ranking: conservative probabilities, every exclusion rule, the no-bet
outcome, sizing and correlated exposure."""
import numpy as np
import pandas as pd
import pytest

from nbamodel import opportunities as op

NOW = pd.Timestamp("2026-11-02T22:00:00Z")


def _table():
    rng = np.random.default_rng(0)
    n = 6000
    p = rng.uniform(0.2, 0.8, n)
    early = rng.random(n) < 0.3
    # later games calibrated; early games over-confident by 8 points above 0.6
    truth = np.where(early & (p > 0.6), p - 0.08, p)
    y = (rng.random(n) < truth).astype(float)
    d = pd.DataFrame({"p": p, "home_win": y, "home_season_games": np.where(early, 3, 30), "away_season_games": 30,
                      "date": "2020-01-01"})
    return op.fit_uncertainty(d, "p", early_games=10)


TABLE = _table()


def test_shortfall_is_measured_and_larger_where_the_model_was_overconfident():
    later, early = op.shortfall(TABLE, 0.72, early=False), op.shortfall(TABLE, 0.72, early=True)
    assert 0 <= later < 0.03                        # calibrated: only sampling uncertainty
    assert early > later + 0.05                     # over-confident early games: bigger haircut
    assert op.shortfall(TABLE, 0.999, early=True) >= 0      # never raises p


def _pred(key, home, away, p_home, h, a, games=30, spread=0.05, ts="2026-11-02T21:45:00Z"):
    nv = (1 / h) / (1 / h + 1 / a)
    return {"game_key": key, "home_team": home, "away_team": away, "tipoff_utc": "2026-11-03T00:00:00Z",
            "date": "2026-11-02", "p_home": p_home, "home_odds": h, "away_odds": a, "market_home_prob_novig": nv,
            "odds_timestamp": ts, "model_spread": spread, "home_season_games": games, "away_season_games": games}


def _flags(keys, **kw):
    base = {"report_age_minutes": 20.0, "report_missing": False, "n_rotation_uncertain": 0, "out_players": "",
            "uncertain_players": ""}
    return pd.DataFrame([{"game_key": k, **base, **kw.get(k, {})} for k in keys])


CFG = op.OpportunityConfig()


def test_eligible_game_is_ranked_with_conservative_numbers_and_stake():
    preds = pd.DataFrame([_pred("g1", "BOS", "NYK", 0.60, 1.85, 2.0)])     # +5.6 pts edge, +11% ROI
    r = op.rank(preds, CFG, TABLE, NOW, _flags(["g1"])).iloc[0]
    assert r["eligible"] and r["rank"] == 1 and r["selection"] == "BOS"
    assert r["roi"] == pytest.approx(0.60 * 1.85 - 1)
    assert r["p_lower"] < r["p"] and r["conservative_roi"] == pytest.approx(r["p_lower"] * 1.85 - 1)
    assert 0 < r["stake_fraction"] <= CFG.max_stake_fraction


@pytest.mark.parametrize("pred,flags,reason", [
    (_pred("g", "BOS", "NYK", 0.52, 1.80, 1.95), {}, "negative_roi"),         # both sides negative
    (_pred("g", "BOS", "NYK", 0.75, 1.85, 2.0), {}, "edge_too_large"),          # kept from Phase 3
    (_pred("g", "BOS", "NYK", 0.60, 1.85, 2.0, ts="2026-11-02T19:00:00Z"), {}, "stale_odds"),
    (_pred("g", "BOS", "NYK", 0.60, 1.85, 2.0, spread=0.3), {}, "model_disagreement"),
    (_pred("g", "BOS", "NYK", 0.60, 1.85, 2.0), {"g": {"n_rotation_uncertain": 1}}, "unresolved_injury"),
    (_pred("g", "BOS", "NYK", 0.60, 1.85, 2.0), {"g": {"report_missing": True}}, "missing_injury"),
    (_pred("g", "BOS", "NYK", 0.60, 1.85, 2.0), {"g": {"report_age_minutes": 400.0}}, "stale_injury"),
    ({**_pred("g", "BOS", "NYK", 0.60, 1.85, 2.0), "home_odds": np.nan}, {}, "no_odds"),
])
def test_each_risk_control_excludes_with_its_reason(pred, flags, reason):
    r = op.rank(pd.DataFrame([pred]), CFG, TABLE, NOW, _flags(["g"], **flags)).iloc[0]
    assert not r["eligible"] and reason in r["reasons"] and op.REASONS[reason] in r["reason_text"]


def test_early_season_needs_a_bigger_conservative_margin():
    cfg = op.OpportunityConfig(early_season_extra_roi=0.10)
    pred = _pred("g", "BOS", "NYK", 0.62, 1.70, 2.3)                         # +5.4% ROI, edge +0.045
    later = op.rank(pd.DataFrame([pred]), cfg, None, NOW, _flags(["g"])).iloc[0]
    early = op.rank(pd.DataFrame([{**pred, "home_season_games": 3}]), cfg, None, NOW, _flags(["g"])).iloc[0]
    assert later["eligible"] and not early["eligible"] and early["reasons"] == ["early_season"]


def test_a_70_percent_favourite_is_not_low_risk_by_itself():
    # 70% on a -300 favourite: expected ROI negative on both sides despite the high win probability
    r = op.rank(pd.DataFrame([_pred("g", "BOS", "NYK", 0.70, 1.333, 3.0)]), CFG, TABLE, NOW, _flags(["g"])).iloc[0]
    assert not r["eligible"] and "negative_roi" in r["reasons"]


def test_no_injury_data_means_no_bet():
    r = op.rank(pd.DataFrame([_pred("g1", "BOS", "NYK", 0.60, 1.85, 2.0)]), CFG, TABLE, NOW, None).iloc[0]
    assert not r["eligible"] and r["reasons"] == ["missing_injury"]


def test_no_bet_day_when_nothing_qualifies():
    preds = pd.DataFrame([_pred("g1", "BOS", "NYK", 0.52, 1.80, 1.95), _pred("g2", "MIA", "ORL", 0.80, 1.5, 2.7)])
    r = op.rank(preds, CFG, TABLE, NOW, _flags(["g1", "g2"]))
    assert not r["eligible"].any() and r["rank"].isna().all()


def test_ranking_order_team_exposure_and_daily_cap():
    preds = pd.DataFrame([_pred("g1", "BOS", "NYK", 0.58, 1.85, 2.0), _pred("g2", "MIA", "ORL", 0.61, 1.85, 2.0),
                          _pred("g3", "BOS", "CHI", 0.60, 1.85, 2.0)])          # BOS twice in one day
    r = op.rank(preds, CFG, TABLE, NOW, _flags(["g1", "g2", "g3"])).set_index("game_key")
    assert r.loc["g2", "rank"] == 1 and r.loc["g3", "rank"] == 2               # by conservative ROI
    assert "team_exposure" in r.loc["g1", "reasons"]                          # BOS already selected
    capped = op.rank(preds.iloc[[1]], CFG, TABLE, NOW, _flags(["g2"]), bankroll=100, used_exposure=10.0).iloc[0]
    assert capped["reasons"] == ["daily_exposure"]


def test_stake_is_sized_on_the_conservative_probability():
    preds = pd.DataFrame([_pred("g1", "BOS", "NYK", 0.60, 1.85, 2.0)])
    loose = op.rank(preds, CFG, None, NOW, _flags(["g1"])).iloc[0]           # no table: p_lower = p
    tight = op.rank(preds, CFG, TABLE, NOW, _flags(["g1"])).iloc[0]
    assert tight["stake_fraction"] <= loose["stake_fraction"]


def test_injury_flags_identify_rotation_players_and_unknown_importance():
    preds = pd.DataFrame([_pred("g1", "BOS", "NYK", 0.60, 1.85, 2.0)])
    statuses = pd.DataFrame({"game_date": ["2026-11-02"] * 3, "matchup": ["NYK@BOS"] * 3, "team": ["BOS", "BOS", "NYK"],
                             "player_key": ["star", "bench", "guard"], "player_name": ["Star", "Bench", "Guard"],
                             "status": ["out", "out", "questionable"]})
    minutes = pd.DataFrame({"team": ["BOS", "BOS", "NYK"], "player_key": ["star", "bench", "guard"],
                            "healthy": [34.0, 8.0, 28.0]})
    cov = {"2026-11-02": {"report_published_at": "2026-11-02T21:30:00+00:00", "report_age_minutes": 30.0,
                          "not_submitted": []}}
    f = op.injury_flags(preds, statuses, cov, minutes, CFG).iloc[0]
    assert f["out_players"] == "Star (BOS)" and f["n_rotation_uncertain"] == 1 and not f["report_missing"]
    assert "Guard (NYK, questionable)" == f["uncertain_players"]
    unknown = op.injury_flags(preds, statuses, cov, minutes.iloc[0:0], CFG).iloc[0]
    assert unknown["report_missing"]                                          # can't judge importance
    pending = op.injury_flags(preds, statuses, {"2026-11-02": {**cov["2026-11-02"], "not_submitted": ["NYK"]}},
                              minutes, CFG).iloc[0]
    assert pending["report_missing"] and pending["not_submitted"] == "NYK"


def test_uncertainty_table_never_uses_the_final_holdout(tmp_path):
    import json
    from nbamodel import cli
    wf = tmp_path / "wf"
    wf.mkdir()
    dev = pd.DataFrame({"p_x": [0.6] * 200, "home_win": [1.0] * 120 + [0.0] * 80, "home_season_games": 30,
                        "away_season_games": 30, "date": "2020-01-01", "phase": "development"})
    final = dev.assign(home_win=0.0, phase="final_test", date="2026-01-01")   # would change the table if used
    pd.concat([dev, final]).to_csv(wf / "predictions.csv", index=False)
    cfg = json.loads(open("config/phase2.json").read())
    cfg["paths"]["walkforward_dir"] = str(wf)
    t = op.fit_uncertainty(dev, "p_x")
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    out = tmp_path / "u.json"
    assert cli.main(["--config", str(tmp_path / "cfg.json"), "uncertainty-fit", "--p-col", "p_x", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["groups"] == json.loads(json.dumps(t["groups"]))
