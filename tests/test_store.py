import json

import numpy as np
import pandas as pd
import pytest

from nbamodel import store
from tests.conftest import tiny


def _pred(game_key, date, p, ts, tip=None, version="m1"):
    return {"game_key": game_key, "date": pd.Timestamp(date), "season": 2023, "home_team": "A", "away_team": "B",
            "tipoff_utc": tip, "prediction_timestamp": ts, "model_version": version, "p_home": p, "p_away": 1 - p,
            "fair_home_american": -150, "fair_away_american": 150, "fair_home_decimal": 1 / p,
            "fair_away_decimal": 1 / (1 - p), "home_odds": np.nan, "away_odds": np.nan, "odds_timestamp": pd.NaT,
            "odds_book": None, "market_home_prob_novig": np.nan, "model_spread": 0.02, "low_history": False,
            "features_asof": "2023-10-23", "feature_snapshot": json.dumps({"elo_diff": 10.0}), "key_factors": "x"}


@pytest.fixture
def con(tmp_path):
    c = store.connect(tmp_path / "p.db")
    yield c
    c.close()


def test_predictions_are_append_only(con):
    df = pd.DataFrame([_pred("g1", "2023-10-24", 0.6, "2023-10-24T12:00:00+00:00")])
    assert store.log_predictions(con, df) == 1
    assert store.log_predictions(con, df.assign(p_home=0.9)) == 0     # same key+version+ts: ignored, not updated
    assert con.execute("SELECT p_home FROM predictions").fetchone()[0] == pytest.approx(0.6)
    snap = json.loads(con.execute("SELECT feature_snapshot FROM predictions").fetchone()[0])
    assert snap == {"elo_diff": 10.0}


def test_reconcile_scores_and_flags_late_predictions(con):
    games = tiny([("2023-10-24", "A", "B", 110, 100), ("2023-10-25", "A", "C", 90, 100)])
    keys = games["game_key"].tolist()
    store.log_predictions(con, pd.DataFrame([
        _pred(keys[0], "2023-10-24", 0.6, "2023-10-24T12:00:00+00:00", tip="2023-10-24T23:30:00Z"),
        _pred(keys[0], "2023-10-24", 0.9, "2023-10-25T01:00:00+00:00", tip="2023-10-24T23:30:00Z"),  # after tip-off
        _pred(keys[1], "2023-10-25", 0.7, "2023-10-25T10:00:00+00:00")]))
    out = store.reconcile(con, games)
    assert out == {"results_recorded": 2, "predictions_scored": 3, "late_predictions": 1}
    assert store.reconcile(con, games)["predictions_scored"] == 0          # idempotent
    ev = store.official_evaluations(con)
    assert len(ev) == 2 and ev["late"].eq(0).all()
    g1 = ev[ev["game_key"] == keys[0]].iloc[0]
    assert g1["p_home"] == pytest.approx(0.6) and g1["correct"] == 1
    assert g1["log_loss"] == pytest.approx(-np.log(0.6))
    m = store.snapshot_metrics(con)
    assert m.loc[m["window"] == "all", "n"].iloc[0] == 2


def _many(con, n, p, wins, version="m1"):
    rows = [(f"2023-11-{1 + i // 10:02d}", f"T{i % 10}a", f"T{i % 10}b", 100 + w, 100 - (1 - w) * 2 - w)
            for i, w in enumerate(wins)]
    g = tiny([(d, h, a, hs, as_ if as_ != hs else as_ - 1) for d, h, a, hs, as_ in rows])
    store.log_predictions(con, pd.DataFrame([_pred(k, d, p, (pd.Timestamp(d) - pd.Timedelta(days=1)).isoformat() + "+00:00",
                                                   version=version)
                                             for k, d in zip(g["game_key"], g["date"])]))
    store.reconcile(con, g)
    return g


def test_degradation_needs_a_significant_sample(con):
    rng = np.random.default_rng(0)
    wins = (rng.uniform(size=200) < 0.6).astype(int)
    _many(con, 200, 0.6, wins)
    ref = {"log_loss": 0.673, "log_loss_sd": 0.2}
    ok = store.detect_degradation(con, "m1", ref, min_n=100)
    assert ok["degraded"] is False
    assert store.detect_degradation(con, "m1", ref, min_n=500) is None   # not enough games: no verdict
    bad = store.detect_degradation(con, "m1", {"log_loss": 0.45, "log_loss_sd": 0.2}, min_n=100)
    assert bad["degraded"] is True
    assert con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] == 1


def test_retrain_due_logic(con):
    now = pd.Timestamp("2024-01-10", tz="UTC")
    assert store.retrain_due(con, 7, None, 5, now) == "no_training_run_recorded"
    meta = {"version": "v1", "trained_at": "2024-01-01T00:00:00+00:00",
            "training": {"start": "2020-01-01", "end": "2023-12-31", "n_rows": 10}}
    store.record_training_run(con, meta, "manual")
    assert store.retrain_due(con, 7, None, 5, now) == "scheduled_every_7_days"
    assert store.retrain_due(con, 7, None, 0, now) is None        # nothing new to learn from
    assert store.retrain_due(con, 30, None, 5, now) is None
    assert store.retrain_due(con, None, {"degraded": True}, 0, now) == "degradation"


def test_elo_snapshot(con):
    store.snapshot_elo(con, {"A": 1510.0, "B": 1490.0}, pd.Timestamp("2024-01-01", tz="UTC"))
    assert con.execute("SELECT COUNT(*) FROM elo_ratings").fetchone()[0] == 2
