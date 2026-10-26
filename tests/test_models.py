import numpy as np
import pandas as pd
import pytest

from nbamodel import registry
from nbamodel.dataset import build_training_dataset
from nbamodel.features import FEATURES_V2
from nbamodel.leakage import outcome_columns
from nbamodel.models import (Calibrator, CalibratedModel, EloModel, GBMModel, LogisticModel, elo_grid_probabilities,
                             ensemble_mean, inner_split)


@pytest.fixture(scope="module")
def ds(multi_season_games):
    return build_training_dataset(multi_season_games)


@pytest.fixture(scope="module")
def split(ds):
    train = ds[ds["train_eligible"] & (ds["season"] < 2022)]
    test = ds[ds["played"] & (ds["season"] == 2022)]
    return train, test


def test_inner_split_is_chronological(split):
    train, _ = split
    fit, val = inner_split(train["date"], 0.2)
    assert train["date"][fit].max() < train["date"][val].min()


def test_elo_selection_uses_only_training_rows(multi_season_games, split):
    train, test = split
    grid = elo_grid_probabilities(multi_season_games)
    a = EloModel(grid).fit(train)
    # Flipping every outcome in the later (test) season cannot change the chosen config.
    from nbamodel.leakage import perturb_outcomes
    tampered = perturb_outcomes(multi_season_games, multi_season_games["season"] == 2022)
    b = EloModel(elo_grid_probabilities(tampered)).fit(train)
    assert a.config == b.config and a.selection == b.selection
    p = a.predict(test.drop(columns=outcome_columns(test)))
    assert ((p > 0) & (p < 1)).all() and len(p) == len(test)


@pytest.mark.parametrize("make", [lambda: LogisticModel(FEATURES_V2), lambda: GBMModel(FEATURES_V2, seed=3)])
def test_models_are_deterministic_and_need_no_outcomes(split, make):
    train, test = split
    X = test.drop(columns=outcome_columns(test))
    p1 = make().fit(train).predict(X)
    p2 = make().fit(train).predict(X)
    np.testing.assert_array_equal(p1, p2)
    assert ((p1 > 0) & (p1 < 1)).all()


def test_logistic_beats_coin_flip_out_of_sample(split):
    from nbamodel.evaluation import score
    train, test = split
    p = LogisticModel(FEATURES_V2).fit(train).predict(test)
    assert score(test["home_win"], p)["log_loss"] < np.log(2)


def test_platt_and_isotonic_fix_overconfidence():
    rng = np.random.default_rng(0)
    true = rng.uniform(0.2, 0.8, 4000)
    y = (rng.uniform(size=4000) < true).astype(float)
    over = 1 / (1 + np.exp(-3 * np.log(true / (1 - true))))   # 3x too extreme
    from nbamodel.evaluation import score
    for method in ("platt", "isotonic"):
        cal = Calibrator(method).fit(over[:2000], y[:2000])
        assert score(y[2000:], cal.transform(over[2000:]))["log_loss"] < score(y[2000:], over[2000:])["log_loss"]
    assert Calibrator("none").fit(over, y).transform([0.3])[0] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        Calibrator("magic").fit(over, y)


def test_calibrated_model_fits_calibrator_on_later_window(split):
    train, test = split
    dates = np.sort(train["date"].unique())
    fit_rows, cal_rows = train[train["date"] < dates[-15]], train[train["date"] >= dates[-15]]
    m = CalibratedModel(LogisticModel(FEATURES_V2), Calibrator("platt")).fit(fit_rows, cal_rows)
    assert m.base.model is not None and "slope" in m.params["calibration"]
    raw, cal = m.predict_raw(test), m.predict(test)
    assert raw.shape == cal.shape and not np.allclose(raw, cal)


def test_ensemble_mean_in_logit_space():
    assert ensemble_mean([0.5], [0.5])[0] == pytest.approx(0.5)
    assert ensemble_mean([0.8], [0.2])[0] == pytest.approx(0.5)


def test_model_serialization_round_trip(tmp_path, split):
    train, test = split
    m = GBMModel(FEATURES_V2).fit(train)
    meta = {"version": "gbm-test-1", "trained_at": "2026-01-01T00:00:00+00:00", "training": {"n_rows": len(train)}}
    registry.save_model(m, meta, tmp_path)
    assert registry.active_version(tmp_path) == "gbm-test-1"
    loaded, loaded_meta = registry.load_model(root=tmp_path)
    np.testing.assert_array_equal(loaded.predict(test), m.predict(test))
    assert loaded_meta["package_version"] and loaded_meta["version"] == "gbm-test-1"
    with pytest.raises(FileExistsError):
        registry.save_model(m, meta, tmp_path)


def test_lightgbm_backend_reports_missing_dependency(split):
    pytest.importorskip("sklearn")
    try:
        import lightgbm  # noqa: F401
        pytest.skip("lightgbm is importable here")
    except (ImportError, OSError):
        pass
    with pytest.raises(RuntimeError, match="lightgbm"):
        GBMModel(FEATURES_V2, backend="lightgbm").fit(split[0])
